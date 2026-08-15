#!/usr/bin/env python3
"""Ticket-bound TensorBoard lifecycle management for a remote GPU profile.

The sidecar itself is CPU-only.  It launches TensorBoard on the remote loopback
interface, records a strong process identity in the local GPU ticket ledger,
and never terminates a process unless that identity still matches exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from credential_guard import contains_secret, normalize_for_secret_scan
from profile import ProfileError, load_profile
from remote_path_guard import RemotePathError, require_managed_remote_path
from ssh_remote import SSHError, ssh_argv
from managed_run import ManagedRunError, build_landlock_command


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
TICKET_HELPER = SCRIPT_DIR / "gpu_ticket.py"

PORT_MIN = 16006
PORT_MAX = 16105
RESERVED_PROXY_PORT = 17890
REMOTE_HOST = "127.0.0.1"
SESSION_PREFIX = "remote-gpu-tb-"
REMOTE_PYTHON = "/usr/bin/python3"
REMOTE_RECORDS_ROOT = "/tmp/remote-gpu-dev-records-unconfigured"
PROFILE: dict[str, Any] | None = None
TICKET_ID_RE = re.compile(r"GPU-[\w-]{1,156}\Z", flags=re.UNICODE)
class SidecarError(RuntimeError):
    """A safe, user-facing sidecar failure."""


class TransientSSHError(SidecarError):
    """A bounded, retryable SSH transport interruption."""


def configure_profile() -> None:
    global PROFILE, PORT_MIN, PORT_MAX, RESERVED_PROXY_PORT
    global REMOTE_PYTHON, REMOTE_RECORDS_ROOT, SESSION_PREFIX
    try:
        PROFILE = load_profile()
    except ProfileError as exc:
        raise SidecarError(str(exc)) from exc
    PORT_MIN = int(PROFILE["dashboard"]["tensorboard_remote_port_start"])
    PORT_MAX = int(PROFILE["dashboard"]["tensorboard_remote_port_end"])
    RESERVED_PROXY_PORT = int(PROFILE["remote"]["proxy_port"])
    REMOTE_PYTHON = str(PROFILE["remote"]["monitor_python"])
    REMOTE_RECORDS_ROOT = str(PROFILE["remote"]["records_root"])
    profile_digest = PROFILE["trust"]["coordination_uid"].split(":", 1)[1][:10]
    SESSION_PREFIX = f"rgpu-{profile_digest}-tb-"


def _run(
    argv: Sequence[str], *, timeout: float = 30, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(argv),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        error_type = TransientSSHError if Path(argv[0]).name == "ssh" else SidecarError
        raise error_type(f"command timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise SidecarError(f"could not execute {argv[0]}: {exc}") from exc
    if check and completed.returncode != 0:
        detail = normalize_for_secret_scan(completed.stderr or completed.stdout)
        if contains_secret(detail):
            detail = "error output was suppressed because it may contain a secret"
        detail = detail[:500]
        suffix = f": {detail}" if detail else ""
        message = f"{Path(argv[0]).name} exited {completed.returncode}{suffix}"
        if Path(argv[0]).name == "ssh" and completed.returncode == 255:
            raise TransientSSHError(message)
        raise SidecarError(message)
    return completed


def _parse_json(text: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SidecarError(f"{source} returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise SidecarError(f"{source} did not return a JSON object")
    return value


def _validate_ticket_id(ticket_id: str) -> str:
    if not TICKET_ID_RE.fullmatch(ticket_id):
        raise SidecarError("ticket ID has an invalid format")
    return ticket_id


def _tensorboard_path_prefix(ticket_id: str) -> str:
    return f"/tb/{quote(_validate_ticket_id(ticket_id), safe='')}"


def _session_name(ticket_id: str) -> str:
    # tmux identifiers stay short ASCII even when a legacy ledger ID contains
    # Unicode.  The full ticket ID remains bound by the ledger and URL prefix.
    digest = hashlib.sha256(_validate_ticket_id(ticket_id).encode("utf-8")).hexdigest()
    return SESSION_PREFIX + digest[:32]


def _ticket_status(ticket_id: str) -> dict[str, Any]:
    completed = _run(
        [sys.executable, str(TICKET_HELPER), "status", ticket_id, "--json"],
        timeout=15,
    )
    ticket = _parse_json(completed.stdout, "gpu_ticket.py status")
    if ticket.get("id") != ticket_id:
        raise SidecarError("ticket status returned a different ticket ID")
    return ticket


def _safe_last_error(message: str) -> str:
    message = normalize_for_secret_scan(str(message))
    if contains_secret(message):
        return "sidecar operation failed; sensitive-looking detail was suppressed"
    return message[:240] or "sidecar operation failed"


def _ticket_tensorboard(
    ticket_id: str, status: str, **fields: str | int | None
) -> dict[str, Any]:
    option_names = {
        "logdir": "--logdir",
        "env_prefix": "--env-prefix",
        "remote_port": "--remote-port",
        "path_prefix": "--path-prefix",
        "session": "--session",
        "pid": "--pid",
        "process_start_ticks": "--process-start-ticks",
        "boot_id": "--boot-id",
        "version": "--version",
        "command_sha256": "--command-sha256",
        "last_error": "--last-error",
        "expected_generation": "--expected-generation",
    }
    argv = [
        sys.executable,
        str(TICKET_HELPER),
        "tensorboard",
        ticket_id,
        "--status",
        status,
    ]
    for key, value in fields.items():
        if value is None:
            continue
        option = option_names.get(key)
        if option is None:
            raise SidecarError(f"internal error: unsupported ledger field {key}")
        argv.extend([option, str(value)])
    completed = _run(argv, timeout=15)
    return _parse_json(completed.stdout, "gpu_ticket.py tensorboard")


def _remote_json(
    code: str,
    *arguments: str,
    timeout: float = 45,
    allow_pty: bool = False,
    retry_transient: bool = False,
) -> dict[str, Any]:
    if PROFILE is None:
        raise SidecarError("profile is not configured")
    remote_argv = [REMOTE_PYTHON, "-c", code, *arguments]
    try:
        remote_command = build_landlock_command(
            PROFILE,
            remote_argv,
            workdir=PROFILE["remote"]["temp_root"],
            allow_pty=allow_pty,
        )
    except (RemotePathError, ManagedRunError) as exc:
        raise SidecarError(str(exc)) from exc
    try:
        argv = ssh_argv(PROFILE, batch=True)
    except SSHError as exc:
        raise SidecarError(str(exc)) from exc
    argv.extend([f"{PROFILE['ssh']['user']}@{PROFILE['ssh']['host']}", remote_command])
    attempts = 5 if retry_transient else 1
    deadline = time.monotonic() + timeout
    last_error: TransientSSHError | None = None
    failed_attempts = 0
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempts_left = attempts - attempt + 1
        attempt_timeout = remaining
        if retry_transient:
            attempt_timeout = min(6.0, attempt_timeout)
        if retry_transient and attempts_left > 1:
            attempt_timeout = min(
                attempt_timeout, max(1.0, remaining / attempts_left)
            )
        try:
            completed = _run(argv, timeout=attempt_timeout)
            return _parse_json(completed.stdout, "remote sidecar helper")
        except TransientSSHError as exc:
            last_error = exc
            failed_attempts += 1
            if attempt == attempts:
                break
            print(
                "tensorboard-sidecar: warning: transient SSH control check "
                f"failed ({attempt}/{attempts}); retrying",
                file=sys.stderr,
            )
            delay = min(0.5, max(0.0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
    if last_error is not None and failed_attempts == attempts:
        raise TransientSSHError(
            f"{last_error} ({attempts} consecutive SSH attempts failed)"
        ) from last_error
    if last_error is not None:
        raise TransientSSHError(
            "SSH retry window expired after "
            f"{failed_attempts} of {attempts} attempts"
        ) from last_error
    raise TransientSSHError(
        f"SSH retry window expired before {attempts} attempts completed"
    )


REMOTE_PREFLIGHT = r'''
import json, os, pathlib, subprocess, sys

def inside(path, root):
    return path == root or root in path.parents

def fail(message):
    print(json.dumps({"ok": False, "error": str(message)}))
    raise SystemExit(0)

try:
    raw_workdir, raw_logdir, raw_env, raw_records, raw_temp, raw_durable = sys.argv[1:7]
    workdir = pathlib.Path(raw_workdir)
    requested = pathlib.Path(raw_logdir)
    env_prefix = pathlib.Path(raw_env)
    if not workdir.is_absolute() or not requested.is_absolute() or not env_prefix.is_absolute():
        fail("workdir, logdir, and env prefix must be absolute")
    roots = [pathlib.Path(raw_temp), pathlib.Path(raw_durable)]
    for root in roots:
        resolved_root = root.resolve(strict=True)
        if resolved_root != root or not root.is_dir():
            fail("managed root is missing, not a directory, or uses a symlink")
    workdir = workdir.resolve(strict=True)
    if not workdir.is_dir():
        fail("ticket remote workdir is not a directory")
    records = pathlib.Path(raw_records).resolve(strict=True)
    if not records.is_dir() or not any(inside(records, root) for root in roots):
        fail("records root is outside the managed roots")
    if not any(inside(workdir, root) for root in roots):
        fail("ticket remote workdir is outside the managed roots")
    allowed = roots

    cursor = requested
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    if not cursor.exists():
        fail("logdir has no existing ancestor")
    ancestor = cursor.resolve(strict=True)
    if not any(inside(ancestor, root) for root in allowed):
        fail("logdir is outside the ticket workdir and configured records root")
    requested.mkdir(parents=True, exist_ok=True)
    logdir = requested.resolve(strict=True)
    if not logdir.is_dir() or not any(inside(logdir, root) for root in allowed):
        fail("resolved logdir is outside the permitted roots")

    env_prefix = env_prefix.resolve(strict=True)
    if not env_prefix.is_dir() or not any(inside(env_prefix, root) for root in roots):
        fail("Conda environment prefix is outside the managed roots")
    executable = env_prefix / "bin" / "tensorboard"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        fail("the experiment Conda environment has no executable bin/tensorboard")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    probe = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True,
        timeout=20, env=environment, check=False,
    )
    if probe.returncode != 0:
        fail("tensorboard --version failed in the experiment Conda environment")
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not lines:
        fail("tensorboard --version returned no version")
    version = lines[-1][:100]
    state_dir = workdir / ".remote-gpu-dev" / "tensorboard"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir = state_dir.resolve(strict=True)
    if not inside(state_dir, workdir):
        fail("sidecar state directory escaped the ticket workdir")
    print(json.dumps({
        "ok": True,
        "workdir": str(workdir),
        "logdir": str(logdir),
        "env_prefix": str(env_prefix),
        "executable": str(executable),
        "version": version,
        "state_dir": str(state_dir),
    }))
except SystemExit:
    raise
except Exception as exc:
    fail(type(exc).__name__ + ": " + str(exc))
'''


REMOTE_LAUNCH = r'''
import hashlib, http.client, json, os, pathlib, re, shlex, signal, socket
import subprocess, sys, time
from urllib.parse import quote, unquote

def emit(value):
    print(json.dumps(value))
    raise SystemExit(0)

def sessions(session):
    result = subprocess.run(
        ["tmux", "-f", "/dev/null", "-L", session, "list-sessions", "-F", "#{session_name}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]

def pane_pids(session):
    result = subprocess.run(
        ["tmux", "-f", "/dev/null", "-L", session, "list-panes", "-t", "=" + session, "-F", "#{pane_pid}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.isdigit()]

def identity(pid):
    proc = pathlib.Path("/proc") / str(pid)
    stat = (proc / "stat").read_text()
    remainder = stat.rsplit(")", 1)[1].split()
    start_ticks = int(remainder[19])
    cmdline = (proc / "cmdline").read_bytes()
    if not cmdline:
        raise RuntimeError("empty process cmdline")
    return {
        "pid": pid,
        "process_start_ticks": start_ticks,
        "boot_id": pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "command_sha256": hashlib.sha256(cmdline).hexdigest(),
        "cmdline": cmdline,
    }

def same_identity(saved):
    try:
        current = identity(saved["pid"])
        return (
            current["process_start_ticks"] == saved["process_start_ticks"]
            and current["boot_id"] == saved["boot_id"]
            and current["command_sha256"] == saved["command_sha256"]
        )
    except Exception:
        return False

def health(port, path_prefix):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
    try:
        connection.request("GET", path_prefix + "/data/plugins_listing")
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        if response.status != 200:
            return False
        decoded = json.loads(body.decode("utf-8"))
        return isinstance(decoded, dict)
    except Exception:
        return False
    finally:
        connection.close()

created = False
saved_identity = None
try:
    def interrupted(signum, frame):
        raise RuntimeError("remote sidecar helper interrupted before live registration")
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env_prefix, logdir, workdir, raw_port, path_prefix, session, raw_timeout, raw_min, raw_max, raw_reserved, session_prefix = sys.argv[1:12]
    port = int(raw_port)
    timeout = float(raw_timeout)
    port_min = int(raw_min)
    port_max = int(raw_max)
    reserved_port = int(raw_reserved)
    if not (port_min <= port <= port_max) or port == reserved_port:
        emit({"ok": False, "error": "remote port is outside the sidecar pool"})
    if not path_prefix.startswith("/tb/"):
        emit({"ok": False, "error": "invalid TensorBoard path prefix"})
    encoded_ticket_id = path_prefix[len("/tb/"):]
    try:
        decoded_ticket_id = unquote(encoded_ticket_id, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        emit({"ok": False, "error": "invalid TensorBoard path prefix encoding"})
    if (
        quote(decoded_ticket_id, safe="") != encoded_ticket_id
        or not re.fullmatch(r"GPU-[\w-]{1,156}", decoded_ticket_id)
    ):
        emit({"ok": False, "error": "invalid TensorBoard path prefix"})
    expected_session = session_prefix + hashlib.sha256(
        decoded_ticket_id.encode("utf-8")
    ).hexdigest()[:32]
    if session != expected_session:
        emit({"ok": False, "error": "invalid tmux session name"})
    env_prefix = pathlib.Path(env_prefix).resolve(strict=True)
    logdir = pathlib.Path(logdir).resolve(strict=True)
    workdir = pathlib.Path(workdir).resolve(strict=True)
    executable = env_prefix / "bin" / "tensorboard"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        emit({"ok": False, "error": "TensorBoard executable disappeared"})
    if not logdir.is_dir():
        emit({"ok": False, "error": "TensorBoard logdir disappeared"})
    if session in sessions(session):
        emit({"ok": False, "error": "exact dedicated tmux server/session already exists"})
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Match normal HTTP-server reuse semantics. A recently closed viewer
        # can leave TCP TIME_WAIT entries, which are not an active listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError:
        emit({"ok": False, "error": "selected remote loopback port is already occupied"})
    finally:
        probe.close()

    state_dir = workdir / ".remote-gpu-dev" / "tensorboard"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_file = state_dir / (session + ".log")
    argv = [
        str(executable), "--logdir", str(logdir), "--host", "127.0.0.1",
        "--port", str(port), "--path_prefix", path_prefix,
    ]
    shell_command = (
        "exec " + shlex.join(["env", "CUDA_VISIBLE_DEVICES=", *argv])
        + " >> " + shlex.quote(str(log_file)) + " 2>&1"
    )
    launched = subprocess.run(
        ["tmux", "-f", "/dev/null", "-L", session, "new-session", "-d", "-s", session, shell_command],
        capture_output=True, text=True, check=False,
    )
    if launched.returncode != 0:
        detail = (launched.stderr or launched.stdout).strip()
        error = "tmux could not create the exact sidecar session (exit %d)" % launched.returncode
        if detail:
            error += ": " + detail[:320]
        emit({"ok": False, "error": error})
    created = True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = pane_pids(session)
        if len(pids) != 1:
            if session not in sessions(session):
                raise RuntimeError("TensorBoard tmux session exited during startup")
            time.sleep(0.1)
            continue
        candidate = identity(pids[0])
        cmdline = candidate["cmdline"]
        required = [
            b"tensorboard", b"--host", b"127.0.0.1", b"--port",
            str(port).encode(), b"--path_prefix", path_prefix.encode(),
        ]
        if not all(item in cmdline for item in required):
            time.sleep(0.1)
            continue
        saved_identity = candidate
        if health(port, path_prefix):
            candidate.pop("cmdline")
            emit({
                "ok": True,
                **candidate,
                "remote_host": "127.0.0.1",
                "remote_port": port,
                "path_prefix": path_prefix,
                "session": session,
                "log_file": str(log_file),
            })
        time.sleep(0.25)
    raise RuntimeError("TensorBoard did not become healthy before the startup timeout")
except SystemExit:
    raise
except Exception as exc:
    cleanup_pending = False
    if created:
        try:
            if saved_identity is not None and same_identity(saved_identity):
                os.kill(saved_identity["pid"], signal.SIGTERM)
                time.sleep(0.5)
            if session in sessions(session):
                subprocess.run(
                    ["tmux", "-f", "/dev/null", "-L", session, "kill-session", "-t", "=" + session],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=5, check=False,
                )
            cleanup_pending = session in sessions(session)
        except Exception:
            cleanup_pending = True
    emit({
        "ok": False,
        "error": type(exc).__name__ + ": " + str(exc),
        "cleanup_pending": cleanup_pending,
    })
'''


REMOTE_INSPECT = r'''
import hashlib, http.client, json, pathlib, subprocess, sys

def pane_pids(session):
    result = subprocess.run(
        ["tmux", "-f", "/dev/null", "-L", session, "list-panes", "-t", "=" + session, "-F", "#{pane_pid}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.isdigit()]

def health(port, prefix):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
    try:
        connection.request("GET", prefix + "/data/plugins_listing")
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        return response.status == 200 and isinstance(json.loads(body.decode()), dict)
    except Exception:
        return False
    finally:
        connection.close()

try:
    pid = int(sys.argv[1])
    expected_ticks = int(sys.argv[2])
    expected_boot, expected_hash, session = sys.argv[3:6]
    port = int(sys.argv[6])
    prefix = sys.argv[7]
    current_boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    result = {
        "reachable": True, "identity_match": False, "session_match": False,
        "healthy": False, "process_state": "unknown",
    }
    if current_boot != expected_boot:
        result["process_state"] = "boot_id_mismatch"
    else:
        proc = pathlib.Path("/proc") / str(pid)
        if not proc.exists():
            result["process_state"] = "absent"
        else:
            stat = (proc / "stat").read_text()
            fields = stat.rsplit(")", 1)[1].split()
            state = fields[0]
            ticks = int(fields[19])
            cmdline = (proc / "cmdline").read_bytes()
            digest = hashlib.sha256(cmdline).hexdigest()
            result["observed_start_ticks"] = ticks
            result["observed_command_sha256"] = digest
            if ticks == expected_ticks and state in {"Z", "X"}:
                result["process_state"] = "terminated"
                result["session_match"] = pid in pane_pids(session)
                print(json.dumps(result))
                raise SystemExit(0)
            result["identity_match"] = ticks == expected_ticks and digest == expected_hash
            result["process_state"] = "matching" if result["identity_match"] else "identity_mismatch"
            result["session_match"] = pid in pane_pids(session)
            if result["identity_match"] and result["session_match"]:
                result["healthy"] = health(port, prefix)
    print(json.dumps(result))
except Exception as exc:
    print(json.dumps({
        "reachable": True, "identity_match": False, "session_match": False,
        "healthy": False, "process_state": "inspection_error",
        "error": type(exc).__name__ + ": " + str(exc),
    }))
'''


REMOTE_STOP = r'''
import hashlib, json, os, pathlib, signal, socket, subprocess, sys, time

def emit(status, message):
    print(json.dumps({"ok": status == "stopped", "status": status, "message": message}))
    raise SystemExit(0)

def pane_pids(session):
    result = subprocess.run(
        ["tmux", "-f", "/dev/null", "-L", session, "list-panes", "-t", "=" + session, "-F", "#{pane_pid}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.isdigit()]

def current_identity(pid):
    proc = pathlib.Path("/proc") / str(pid)
    if not proc.exists():
        return None
    stat = (proc / "stat").read_text()
    fields = stat.rsplit(")", 1)[1].split()
    return {
        "state": fields[0],
        "ticks": int(fields[19]),
        "sha": hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest(),
    }

def port_listening(port):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    finally:
        probe.close()

try:
    pid = int(sys.argv[1])
    expected_ticks = int(sys.argv[2])
    expected_boot, expected_hash, session = sys.argv[3:6]
    port = int(sys.argv[6])
    timeout = float(sys.argv[7])
    current_boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if current_boot != expected_boot:
        if not pane_pids(session) and not port_listening(port):
            emit("stopped", "recorded process belonged to a prior boot and no session or port residue remains")
        emit("cleanup_pending", "remote boot ID changed; no process was signalled")
    observed = current_identity(pid)
    if observed is None:
        if pane_pids(session):
            emit("cleanup_pending", "recorded PID is absent but the exact tmux session still has a pane")
        if port_listening(port):
            emit("cleanup_pending", "recorded PID is absent but the registered loopback port is listening")
        emit("stopped", "recorded process and exact tmux session are absent")
    if observed["ticks"] == expected_ticks and observed["state"] in {"Z", "X"}:
        if pane_pids(session):
            emit("cleanup_pending", "recorded process is terminated but the exact tmux session still has a pane")
        if port_listening(port):
            emit("cleanup_pending", "recorded process is terminated but the registered loopback port is listening")
        emit("stopped", "exact recorded TensorBoard process is terminated")
    if observed["ticks"] != expected_ticks or observed["sha"] != expected_hash:
        if not pane_pids(session) and not port_listening(port):
            emit("stopped", "recorded PID was reused and no session or port residue remains")
        emit("cleanup_pending", "PID identity mismatch; no process was signalled")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observed = current_identity(pid)
        if observed is None or observed["state"] in {"Z", "X"}:
            break
        if observed["ticks"] != expected_ticks or observed["sha"] != expected_hash:
            break
        time.sleep(0.1)
    else:
        observed = current_identity(pid)
        if observed is None or observed["state"] in {"Z", "X"}:
            pass
        elif observed["ticks"] == expected_ticks and observed["sha"] == expected_hash:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        else:
            emit("cleanup_pending", "PID identity changed before SIGKILL; no further signal was sent")

    observed = current_identity(pid)
    if observed is not None and observed["state"] not in {"Z", "X"}:
        if observed["ticks"] == expected_ticks and observed["sha"] == expected_hash:
            emit("cleanup_pending", "exact process remained after TERM and KILL")
        if not pane_pids(session) and not port_listening(port):
            emit("stopped", "recorded PID was reused after stop and no session or port residue remains")
        emit("cleanup_pending", "recorded PID was reused after stop; no further signal was sent")
    remaining = pane_pids(session)
    if remaining and any(item != pid for item in remaining):
        emit("cleanup_pending", "exact tmux session now contains an unrecognized PID")
    if remaining:
        subprocess.run(
            ["tmux", "-f", "/dev/null", "-L", session, "kill-session", "-t", "=" + session],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=False,
        )
    if pane_pids(session):
        emit("cleanup_pending", "exact tmux session remained after process termination")
    deadline = time.monotonic() + 2.0
    while port_listening(port) and time.monotonic() < deadline:
        time.sleep(0.1)
    if port_listening(port):
        emit("cleanup_pending", "registered loopback port remained listening after process termination")
    emit("stopped", "exact recorded TensorBoard process stopped")
except SystemExit:
    raise
except Exception as exc:
    emit("cleanup_pending", type(exc).__name__ + ": " + str(exc))
'''


REMOTE_VERIFY_UNTRACKED_ABSENT = r'''
import json, socket, subprocess, sys

session = sys.argv[1]
port = int(sys.argv[2])
listed = subprocess.run(
    ["tmux", "-f", "/dev/null", "-L", session, "list-sessions", "-F", "#{session_name}"],
    capture_output=True, text=True, check=False,
)
sessions = set(listed.stdout.splitlines()) if listed.returncode == 0 else set()
session_present = session in sessions
probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
probe.settimeout(0.75)
try:
    port_listening = probe.connect_ex(("127.0.0.1", port)) == 0
finally:
    probe.close()
print(json.dumps({
    "ok": not session_present and not port_listening,
    "session_present": session_present,
    "port_listening": port_listening,
}))
'''


def _require_remote_workdir(ticket: dict[str, Any]) -> str:
    workdir = ticket.get("remote_workdir")
    if not isinstance(workdir, str) or not workdir.startswith("/"):
        raise SidecarError(
            "ticket has no absolute remote_workdir; record it before starting TensorBoard"
        )
    if PROFILE is None:
        raise SidecarError("profile is not configured")
    try:
        return require_managed_remote_path(PROFILE, workdir, "ticket remote_workdir")
    except RemotePathError as exc:
        raise SidecarError(str(exc)) from exc


def _metadata(ticket: dict[str, Any]) -> dict[str, Any] | None:
    value = ticket.get("tensorboard")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SidecarError("ticket tensorboard metadata is malformed")
    return value


def _resolve_start_paths(
    existing: dict[str, Any] | None,
    requested_logdir: str | None,
    requested_env_prefix: str | None,
) -> tuple[str, str]:
    """Reuse retained paths from any explicitly stopped configuration."""
    logdir = requested_logdir
    env_prefix = requested_env_prefix
    if logdir is not None and env_prefix is not None:
        return logdir, env_prefix
    if not existing or existing.get("status") != "stopped":
        raise SidecarError(
            "--logdir and --env-prefix are required unless reopening a stopped "
            "TensorBoard configuration"
        )
    if logdir is None:
        retained_logdir = existing.get("logdir")
        if not isinstance(retained_logdir, str) or not retained_logdir:
            raise SidecarError("retained TensorBoard logdir is unavailable")
        logdir = retained_logdir
    if env_prefix is None:
        retained_env = existing.get("env_prefix")
        if not isinstance(retained_env, str) or not retained_env:
            raise SidecarError("retained TensorBoard environment is unavailable")
        env_prefix = retained_env
    return logdir, env_prefix


def _observed_generation(metadata: dict[str, Any] | None) -> int | None:
    value = (metadata or {}).get("generation")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _emit_superseded(
    ticket_id: str, expected_generation: int, observed_generation: int | None
) -> int:
    print(
        json.dumps(
            {
                "ticket_id": ticket_id,
                "status": "superseded",
                "stopped": False,
                "expected_generation": expected_generation,
                "observed_generation": observed_generation,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _return_superseded_if_generation_changed(
    ticket_id: str, expected_generation: int
) -> bool:
    """Report a newer manual generation without mutating or stopping it."""
    try:
        latest = _ticket_status(ticket_id)
        observed = _observed_generation(_metadata(latest))
    except SidecarError:
        return False
    if observed == expected_generation:
        return False
    _emit_superseded(ticket_id, expected_generation, observed)
    return True


def _inspect_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    required = [
        "pid",
        "process_start_ticks",
        "boot_id",
        "command_sha256",
        "session",
        "remote_port",
        "path_prefix",
    ]
    missing = [name for name in required if metadata.get(name) in (None, "")]
    if missing:
        return {
            "reachable": False,
            "identity_match": False,
            "session_match": False,
            "healthy": False,
            "process_state": "identity_incomplete",
            "missing_identity_fields": missing,
        }
    return _remote_json(
        REMOTE_INSPECT,
        str(metadata["pid"]),
        str(metadata["process_start_ticks"]),
        str(metadata["boot_id"]),
        str(metadata["command_sha256"]),
        str(metadata["session"]),
        str(metadata["remote_port"]),
        str(metadata["path_prefix"]),
        timeout=20,
        retry_transient=True,
    )


def _verify_untracked_absent(metadata: dict[str, Any]) -> dict[str, Any]:
    session = metadata.get("session")
    remote_port = metadata.get("remote_port")
    if not isinstance(session, str) or not session:
        raise SidecarError("TensorBoard metadata has no exact tmux session")
    if not isinstance(remote_port, int):
        raise SidecarError("TensorBoard metadata has no remote port")
    return _remote_json(
        REMOTE_VERIFY_UNTRACKED_ABSENT,
        session,
        str(remote_port),
        timeout=20,
        retry_transient=True,
    )


def _best_effort_ticket_update(
    ticket_id: str,
    status: str,
    message: str,
    expected_generation: int | None = None,
    identity: dict[str, str | int] | None = None,
) -> bool:
    """Try twice so a Ctrl-C delivery window cannot skip cleanup recording.

    Repeating the compare-and-set update is safe: terminal/error self-
    transitions are idempotent, while a newer generation still fails closed.
    """
    fields: dict[str, str | int | None] = {
        "last_error": _safe_last_error(message),
        "expected_generation": expected_generation,
        **(identity or {}),
    }
    for _attempt in range(2):
        try:
            _ticket_tensorboard(ticket_id, status, **fields)
            return True
        except (Exception, KeyboardInterrupt):
            continue
    return False


def _best_effort_mark_cleanup(
    ticket_id: str,
    message: str,
    expected_generation: int | None = None,
    identity: dict[str, str | int] | None = None,
) -> bool:
    return _best_effort_ticket_update(
        ticket_id,
        "cleanup_pending",
        message,
        expected_generation,
        identity,
    )


def command_configure(args: argparse.Namespace) -> int:
    """Register a validated TensorBoard source without launching a frontend."""
    ticket_id = _validate_ticket_id(args.ticket_id)
    ticket = _ticket_status(ticket_id)
    existing = _metadata(ticket)
    if existing and existing.get("status") != "stopped":
        raise SidecarError(
            "ticket has an active or unresolved TensorBoard generation; stop it "
            "before configuring the source"
        )
    workdir = _require_remote_workdir(ticket)
    preflight = _remote_json(
        REMOTE_PREFLIGHT,
        workdir,
        args.logdir,
        args.env_prefix,
        REMOTE_RECORDS_ROOT,
        PROFILE["remote"]["temp_root"],
        PROFILE["remote"]["durable_root"],
        timeout=35,
        retry_transient=True,
    )
    if not preflight.get("ok"):
        raise SidecarError(
            _safe_last_error(preflight.get("error", "remote preflight failed"))
        )
    previous_generation = _observed_generation(existing)
    updated = _ticket_tensorboard(
        ticket_id,
        "stopped",
        expected_generation=(
            previous_generation if previous_generation is not None else 0
        ),
        logdir=str(preflight["logdir"]),
        env_prefix=str(preflight["env_prefix"]),
        path_prefix=_tensorboard_path_prefix(ticket_id),
        session=_session_name(ticket_id),
    )
    metadata = _metadata(updated)
    result = {
        "ticket_id": ticket_id,
        "status": "stopped",
        "configured": True,
        "frontend_started": False,
        "tensorboard": metadata,
        "ticket": updated,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_start(args: argparse.Namespace) -> int:
    ticket_id = _validate_ticket_id(args.ticket_id)
    if args.remote_port is not None and not PORT_MIN <= args.remote_port <= PORT_MAX:
        raise SidecarError(f"--remote-port must be between {PORT_MIN} and {PORT_MAX}")
    if args.remote_port == RESERVED_PROXY_PORT:
        raise SidecarError("--remote-port conflicts with the SSH proxy forward")
    ticket = _ticket_status(ticket_id)
    existing = _metadata(ticket)
    if existing and existing.get("status") in {"starting", "live", "cleanup_pending"}:
        observed = _inspect_metadata(existing)
        if existing.get("status") == "live" and observed.get("healthy"):
            result = {
                "ticket_id": ticket_id,
                "status": "live",
                "idempotent": True,
                "tensorboard": existing,
                "observed": observed,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        raise SidecarError(
            "ticket has unresolved TensorBoard state; run sidecar stop before restarting"
        )
    if existing and existing.get("status") == "failed":
        observed = _inspect_metadata(existing)
        if observed.get("identity_match") or observed.get("session_match"):
            raise SidecarError(
                "failed TensorBoard generation still has matching remote residue; "
                "run sidecar stop before restarting"
            )
        absence = _verify_untracked_absent(existing)
        if not absence.get("ok"):
            raise SidecarError(
                "failed TensorBoard generation still owns its exact session or port; "
                "run sidecar stop before restarting"
            )

    logdir, env_prefix = _resolve_start_paths(
        existing, args.logdir, args.env_prefix
    )
    workdir = _require_remote_workdir(ticket)
    preflight = _remote_json(
        REMOTE_PREFLIGHT,
        workdir,
        logdir,
        env_prefix,
        REMOTE_RECORDS_ROOT,
        PROFILE["remote"]["temp_root"],
        PROFILE["remote"]["durable_root"],
        timeout=35,
        retry_transient=True,
    )
    if not preflight.get("ok"):
        raise SidecarError(_safe_last_error(preflight.get("error", "remote preflight failed")))
    path_prefix = _tensorboard_path_prefix(ticket_id)
    session = _session_name(ticket_id)
    starting_fields: dict[str, str | int] = {
        "logdir": str(preflight["logdir"]),
        "env_prefix": str(preflight["env_prefix"]),
        "path_prefix": path_prefix,
        "session": session,
    }
    if args.remote_port is not None:
        starting_fields["remote_port"] = args.remote_port
    previous_generation = (
        int(existing["generation"])
        if existing and isinstance(existing.get("generation"), int)
        else None
    )
    registered = _ticket_tensorboard(
        ticket_id,
        "starting",
        expected_generation=previous_generation,
        **starting_fields,
    )
    metadata = _metadata(registered)
    if not metadata or not isinstance(metadata.get("remote_port"), int):
        _ticket_tensorboard(
            ticket_id,
            "failed",
            last_error="ledger did not assign a remote TensorBoard port",
            expected_generation=(metadata or {}).get("generation"),
        )
        raise SidecarError("ticket ledger did not assign a remote TensorBoard port")
    remote_port = int(metadata["remote_port"])
    generation = int(metadata["generation"])
    try:
        launched = _remote_json(
            REMOTE_LAUNCH,
            str(preflight["env_prefix"]),
            str(preflight["logdir"]),
            str(preflight["workdir"]),
            str(remote_port),
            path_prefix,
            session,
            str(args.startup_timeout),
            str(PORT_MIN),
            str(PORT_MAX),
            str(RESERVED_PROXY_PORT),
            SESSION_PREFIX,
            timeout=args.startup_timeout + 20,
            allow_pty=True,
        )
    except (SidecarError, KeyboardInterrupt) as exc:
        _best_effort_mark_cleanup(
            ticket_id,
            "remote launch outcome is unknown; exact session and port require inspection",
            generation,
        )
        raise SidecarError(
            "remote launch outcome is unknown; cleanup is pending"
        ) from exc
    if not launched.get("ok"):
        error = _safe_last_error(launched.get("error", "remote launch failed"))
        state = "cleanup_pending" if launched.get("cleanup_pending") else "failed"
        _best_effort_ticket_update(
            ticket_id,
            state,
            error,
            generation,
        )
        raise SidecarError(error)
    try:
        identity = {
            "pid": int(launched["pid"]),
            "process_start_ticks": int(launched["process_start_ticks"]),
            "boot_id": str(launched["boot_id"]),
            "version": str(preflight["version"]),
            "command_sha256": str(launched["command_sha256"]),
        }
    except (KeyError, TypeError, ValueError, KeyboardInterrupt) as exc:
        _best_effort_mark_cleanup(
            ticket_id,
            "remote launch returned an incomplete process identity",
            generation,
        )
        raise SidecarError(
            "remote launch returned an incomplete process identity; cleanup is pending"
        ) from exc
    try:
        updated = _ticket_tensorboard(
            ticket_id,
            "live",
            expected_generation=generation,
            **identity,
        )
    except (Exception, KeyboardInterrupt) as exc:
        try:
            rollback = _remote_json(
                REMOTE_STOP,
                str(identity["pid"]),
                str(identity["process_start_ticks"]),
                str(identity["boot_id"]),
                str(identity["command_sha256"]),
                session,
                str(remote_port),
                "5",
                timeout=15,
            )
        except (SidecarError, KeyboardInterrupt) as rollback_exc:
            _best_effort_mark_cleanup(
                ticket_id,
                "ledger live update failed and remote rollback outcome is unknown",
                generation,
                identity,
            )
            raise SidecarError(
                "ledger live update failed and remote rollback outcome is unknown; "
                "cleanup is pending"
            ) from rollback_exc
        state = "failed" if rollback.get("status") == "stopped" else "cleanup_pending"
        rollback_stopped = state == "failed"
        update_message = (
            "ledger live update failed; remote launch was rolled back"
            if rollback_stopped
            else "ledger live update failed; remote rollback did not confirm stop"
        )
        _best_effort_ticket_update(
            ticket_id,
            state,
            update_message,
            generation,
            identity,
        )
        raise SidecarError(update_message) from exc
    result = {
        "ticket_id": ticket_id,
        "status": "live",
        "remote_host": REMOTE_HOST,
        "remote_port": remote_port,
        "path_prefix": path_prefix,
        "url_path": path_prefix + "/",
        "session": session,
        "pid": launched["pid"],
        "process_start_ticks": launched["process_start_ticks"],
        "boot_id": launched["boot_id"],
        "command_sha256": launched["command_sha256"],
        "version": preflight["version"],
        "logdir": preflight["logdir"],
        "log_file": launched["log_file"],
        "ticket": updated,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    ticket_id = _validate_ticket_id(args.ticket_id)
    ticket = _ticket_status(ticket_id)
    metadata = _metadata(ticket)
    if metadata is None:
        result = {
            "ticket_id": ticket_id,
            "ticket_status": ticket.get("status"),
            "status": "not_configured",
            "tensorboard": None,
        }
    else:
        observed = _inspect_metadata(metadata)
        result = {
            "ticket_id": ticket_id,
            "ticket_status": ticket.get("status"),
            "status": metadata.get("status"),
            "tensorboard": metadata,
            "observed": observed,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_stop(args: argparse.Namespace) -> int:
    ticket_id = _validate_ticket_id(args.ticket_id)
    expected_generation = getattr(args, "expected_generation", None)
    if expected_generation is not None and expected_generation <= 0:
        raise SidecarError("--expected-generation must be positive")
    ticket = _ticket_status(ticket_id)
    metadata = _metadata(ticket)
    observed_generation = _observed_generation(metadata)
    if (
        expected_generation is not None
        and observed_generation != expected_generation
    ):
        return _emit_superseded(
            ticket_id, expected_generation, observed_generation
        )
    if metadata is None:
        print(
            json.dumps(
                {"ticket_id": ticket_id, "status": "not_configured", "stopped": True},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    generation = _observed_generation(metadata)
    if generation is None:
        raise SidecarError("TensorBoard metadata has no valid generation")
    if metadata.get("status") == "stopped":
        print(
            json.dumps(
                {"ticket_id": ticket_id, "status": "stopped", "idempotent": True},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    required = [
        "pid",
        "process_start_ticks",
        "boot_id",
        "command_sha256",
        "session",
    ]
    missing = [name for name in required if metadata.get(name) in (None, "")]
    if missing:
        try:
            absence = _verify_untracked_absent(metadata)
        except SidecarError as exc:
            message = "identity is incomplete and remote absence could not be verified"
            updated_cleanup = _best_effort_mark_cleanup(
                ticket_id, message, generation
            )
            if (
                not updated_cleanup
                and _return_superseded_if_generation_changed(ticket_id, generation)
            ):
                return 0
            raise SidecarError(message) from exc
        if absence.get("ok"):
            try:
                updated = _ticket_tensorboard(
                    ticket_id, "stopped", expected_generation=generation
                )
            except SidecarError:
                if _return_superseded_if_generation_changed(ticket_id, generation):
                    return 0
                raise
            print(
                json.dumps(
                    {
                        "ticket_id": ticket_id,
                        "status": "stopped",
                        "message": (
                            "identity was incomplete, but the exact tmux session was "
                            "absent and the registered loopback port was not listening"
                        ),
                        "ticket": updated,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        occupied = []
        if absence.get("session_present"):
            occupied.append("exact tmux session exists")
        if absence.get("port_listening"):
            occupied.append("registered loopback port is listening")
        message = "cannot stop without complete identity; " + ", ".join(occupied)
        try:
            _ticket_tensorboard(
                ticket_id,
                "cleanup_pending",
                last_error=message,
                expected_generation=generation,
            )
        except SidecarError:
            if _return_superseded_if_generation_changed(ticket_id, generation):
                return 0
            raise
        raise SidecarError(message)
    try:
        stopped = _remote_json(
            REMOTE_STOP,
            str(metadata["pid"]),
            str(metadata["process_start_ticks"]),
            str(metadata["boot_id"]),
            str(metadata["command_sha256"]),
            str(metadata["session"]),
            str(metadata["remote_port"]),
            str(args.stop_timeout),
            timeout=args.stop_timeout + 15,
        )
    except (SidecarError, RemotePathError, SSHError, ManagedRunError) as exc:
        message = _safe_last_error(exc)
        try:
            _ticket_tensorboard(
                ticket_id,
                "cleanup_pending",
                last_error=message,
                expected_generation=generation,
            )
        except SidecarError:
            if _return_superseded_if_generation_changed(ticket_id, generation):
                return 0
            raise
        raise
    state = str(stopped.get("status"))
    message = _safe_last_error(stopped.get("message", "remote stop returned no detail"))
    if state == "stopped":
        try:
            updated = _ticket_tensorboard(
                ticket_id, "stopped", expected_generation=generation
            )
        except SidecarError:
            if _return_superseded_if_generation_changed(ticket_id, generation):
                return 0
            raise
        result = {
            "ticket_id": ticket_id,
            "status": "stopped",
            "message": message,
            "ticket": updated,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        _ticket_tensorboard(
            ticket_id,
            "cleanup_pending",
            last_error=message,
            expected_generation=generation,
        )
    except SidecarError:
        if _return_superseded_if_generation_changed(ticket_id, generation):
            return 0
        raise
    raise SidecarError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a ticket-bound, CPU-only TensorBoard sidecar on the configured "
            "remote GPU server."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser(
        "configure",
        help="Validate and register an event source without starting TensorBoard.",
    )
    configure.add_argument("ticket_id")
    configure.add_argument("--env-prefix", required=True, help="Experiment Conda prefix.")
    configure.add_argument("--logdir", required=True, help="Remote TensorBoard event logdir.")

    start = subparsers.add_parser(
        "start", help="Start TensorBoard and mark it live only after a health check."
    )
    start.add_argument("ticket_id")
    start.add_argument(
        "--env-prefix",
        help="Experiment Conda prefix; a stopped configuration may reuse it.",
    )
    start.add_argument(
        "--logdir",
        help="Remote event logdir; a stopped configuration may reuse it.",
    )
    start.add_argument(
        "--remote-port",
        type=int,
        help=f"Optional remote loopback port ({PORT_MIN}..{PORT_MAX}).",
    )
    start.add_argument("--startup-timeout", type=float, default=30.0)

    status = subparsers.add_parser("status", help="Inspect ledger, process identity, and health.")
    status.add_argument("ticket_id")

    stop = subparsers.add_parser(
        "stop", help="Stop only the exact boot/PID/start-ticks/cmdline identity."
    )
    stop.add_argument("ticket_id")
    stop.add_argument("--stop-timeout", type=float, default=8.0)
    stop.add_argument(
        "--expected-generation",
        type=int,
        help=(
            "Stop only this previously observed generation; a newer manual "
            "generation is reported as superseded and left untouched."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "configure": command_configure,
        "start": command_start,
        "status": command_status,
        "stop": command_stop,
    }
    try:
        configure_profile()
        if getattr(args, "startup_timeout", 1) <= 0:
            raise SidecarError("--startup-timeout must be positive")
        if getattr(args, "stop_timeout", 1) <= 0:
            raise SidecarError("--stop-timeout must be positive")
        return commands[args.command](args)
    except SidecarError as exc:
        print(f"tensorboard-sidecar: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
