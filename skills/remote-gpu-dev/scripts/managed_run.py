#!/usr/bin/env python3
"""Run and control ticket-bound Python programs through a structured runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from profile import ProfileError, load_profile
from remote_path_guard import (
    RemotePathError,
    managed_runtime_paths,
    managed_runtime_environment,
    require_managed_remote_path,
)
from ssh_remote import SSHError, ssh_argv


SCRIPT_DIR = Path(__file__).resolve().parent
TICKET_TOOL = SCRIPT_DIR / "gpu_ticket.py"
TICKET_RE = re.compile(r"GPU-[\w-]{1,156}\Z", flags=re.UNICODE)
MODULE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z"
)


class ManagedRunError(RuntimeError):
    pass


def _run(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            env=env,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedRunError(f"could not run structured command: {exc}") from exc


def _ticket(
    profile: dict[str, Any],
    ticket_id: str,
    *,
    allowed_statuses: set[str] | None = None,
) -> dict[str, Any]:
    if not TICKET_RE.fullmatch(ticket_id):
        raise ManagedRunError("ticket ID has an invalid format")
    environment = os.environ.copy()
    environment["REMOTE_GPU_DEV_PROFILE"] = profile["slug"]
    completed = _run(
        [sys.executable, str(TICKET_TOOL), "status", ticket_id, "--json"],
        timeout=15,
        env=environment,
    )
    if completed.returncode != 0:
        raise ManagedRunError("ticket status is unavailable")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ManagedRunError("ticket status returned malformed JSON") from exc
    if not isinstance(value, dict) or value.get("id") != ticket_id:
        raise ManagedRunError("ticket status returned a different ticket")
    allowed = allowed_statuses or {"running"}
    if value.get("status") not in allowed:
        raise ManagedRunError(
            "ticket status is not permitted for this operation: "
            + str(value.get("status"))
        )
    assigned = value.get("assigned_gpus")
    if not isinstance(assigned, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in assigned
    ):
        raise ManagedRunError("ticket has invalid assigned GPUs")
    if value.get("status") in {"running", "stale"} and not assigned:
        raise ManagedRunError("active ticket has no assigned GPUs")
    return value


LANDLOCK_RUNNER = r'''
import ctypes, errno, hashlib, json, os, pathlib, re, signal, stat, sys, time

spec = json.loads(sys.argv[1])

strict_isolation = spec.get("strict_isolation", True)
if not isinstance(strict_isolation, bool):
    print("managed-run: strict_isolation must be a boolean", file=sys.stderr)
    raise SystemExit(126)
allow_pty = spec.get("allow_pty", False)
if not isinstance(allow_pty, bool):
    print("managed-run: allow_pty must be a boolean", file=sys.stderr)
    raise SystemExit(126)

def fail(message):
    print("managed-run: " + str(message), file=sys.stderr)
    raise SystemExit(126)

def inside(path, root):
    return path == root or root in path.parents

roots = [pathlib.Path(item) for item in spec["roots"]]
for root in roots:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        fail("managed root is unavailable: " + str(exc))
    if resolved != root or not root.is_dir():
        fail("managed root is not a canonical directory")
read_exec_roots = [pathlib.Path(item) for item in spec.get("read_exec_roots", [])]
for root in read_exec_roots:
    resolved = root.resolve(strict=True)
    if resolved != root or not root.is_dir() or str(root) in {"/", "/root", "/home", "/opt", "/usr", "/usr/local"}:
        fail("read/execute exception is missing, noncanonical, or too broad")

def managed(raw, kind, canonical=False):
    candidate = pathlib.Path(raw)
    path = candidate.resolve(strict=True)
    if not any(inside(path, root) for root in roots):
        fail(kind + " resolved outside managed roots")
    if canonical and candidate != path:
        fail(kind + " is not a canonical path")
    return path

workdir = managed(spec["workdir"], "workdir", canonical=True)
if not workdir.is_dir():
    fail("workdir is not a directory")
raw_argv = spec.get("argv")
if not isinstance(raw_argv, list) or not raw_argv or not all(isinstance(item, str) for item in raw_argv):
    fail("sandbox argv is invalid")
executable = pathlib.Path(raw_argv[0]).resolve(strict=True)
system_roots = [pathlib.Path(item) for item in (
    "/usr", "/bin", "/sbin", "/lib", "/lib64",
)]
if not any(inside(executable, root) for root in [*roots, *system_roots, *read_exec_roots]):
    fail("executable is outside managed roots and read-only system roots")
if not executable.is_file() or not os.access(executable, os.X_OK):
    fail("sandbox executable is not executable")

required_files = spec.get("required_files", [])
if not isinstance(required_files, list) or any(
    not isinstance(item, str) for item in required_files
):
    fail("required files are invalid")
for raw in required_files:
    required = managed(raw, "required file", canonical=True)
    if not required.is_file():
        fail("required file is not a regular file")

for raw in spec["runtime_dirs"]:
    path = pathlib.Path(raw)
    cursor = path
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    ancestor = cursor.resolve(strict=True)
    if not any(inside(ancestor, root) for root in roots):
        fail("runtime directory ancestor escaped managed roots")
    path.mkdir(parents=True, exist_ok=True)
    managed(str(path), "runtime directory")
condarc = pathlib.Path(spec["environment"]["CONDARC"])
managed(str(condarc.parent), "Conda config parent")
condarc.touch(mode=0o600, exist_ok=True)
managed(str(condarc), "Conda config")

libc = ctypes.CDLL(None, use_errno=True)
SYS_CREATE, SYS_ADD, SYS_RESTRICT = 444, 445, 446
CREATE_VERSION = 1
RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]

class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]

abi = libc.syscall(SYS_CREATE, 0, 0, CREATE_VERSION)
if strict_isolation and abi < 5:
    code = ctypes.get_errno()
    fail("Landlock ABI >=5 is required (observed=%d errno=%d); refusing an unconfined run" % (abi, code))

EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12
REFER = (1 << 13) if abi >= 2 else 0
TRUNCATE = (1 << 14) if abi >= 3 else 0
IOCTL_DEV = (1 << 15) if abi >= 5 else 0
BASE = (
    EXECUTE | WRITE_FILE | READ_FILE | READ_DIR | REMOVE_DIR | REMOVE_FILE
    | MAKE_CHAR | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO | MAKE_BLOCK
    | MAKE_SYM | REFER | TRUNCATE | IOCTL_DEV
)
ruleset_fd = -1
if strict_isolation:
    ruleset_attr = RulesetAttr(BASE)
    ruleset_fd = libc.syscall(
        SYS_CREATE, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0
    )
    if ruleset_fd < 0:
        fail("could not create Landlock ruleset (errno=%d)" % ctypes.get_errno())

def add_path(raw, writable=False, device=False):
    if not strict_isolation:
        return
    try:
        path = pathlib.Path(raw).resolve(strict=True)
    except OSError:
        return
    is_dir = path.is_dir()
    allowed = READ_FILE | EXECUTE
    if is_dir:
        allowed |= READ_DIR
    if writable:
        allowed |= WRITE_FILE | TRUNCATE
        if is_dir:
            allowed |= (
                REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG
                | MAKE_SOCK | MAKE_FIFO | MAKE_SYM | REFER
            )
    if device:
        allowed |= WRITE_FILE | IOCTL_DEV
    allowed &= BASE
    flags = os.O_PATH | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        attr = PathBeneathAttr(allowed, fd, 0)
        result = libc.syscall(
            SYS_ADD, ruleset_fd, RULE_PATH_BENEATH, ctypes.byref(attr), 0
        )
        if result != 0:
            fail("could not add Landlock rule for %s (errno=%d)" % (path, ctypes.get_errno()))
    finally:
        os.close(fd)

# Explicit, read-only runtime exceptions. These contain operating-system
# binaries/libraries, CUDA driver metadata, certificates and proc/sys telemetry;
# user homes, /tmp and broad data mounts are absent.
for path in (
    "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", "/sys", "/run",
):
    add_path(path)
for root in roots:
    add_path(root, writable=True)
for root in read_exec_roots:
    add_path(root)
add_path("/dev/null", writable=True)
add_path("/dev/shm", writable=True)
if allow_pty:
    # tmux needs one pseudoterminal pair even for a detached session.  Keep
    # this opt-in narrower than exposing the whole /dev tree.
    add_path("/dev/ptmx", device=True)
    add_path("/dev/pts", device=True)
gpu_device_names = ({
    "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset",
    *("nvidia" + str(item) for item in spec.get("device_ids", [])),
} if strict_isolation else set())
for name in sorted(gpu_device_names):
    if not re.fullmatch(r"nvidia(?:ctl|-uvm(?:-tools)?|-modeset|[0-9]+)", name):
        fail("invalid GPU device name")
    path = pathlib.Path("/dev") / name
    try:
        metadata = path.stat()
    except FileNotFoundError:
        continue
    if not stat.S_ISCHR(metadata.st_mode):
        fail("GPU device path is not a character device: " + str(path))
    add_path(path, device=True)

if strict_isolation:
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail("could not set no_new_privs (errno=%d)" % ctypes.get_errno())
    if libc.syscall(SYS_RESTRICT, ruleset_fd, 0) != 0:
        fail("could not enter Landlock ruleset (errno=%d)" % ctypes.get_errno())
    os.close(ruleset_fd)

environment = {str(key): str(value) for key, value in spec["environment"].items()}
os.chdir(workdir)
os.umask(0o077)

def atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def process_start_ticks(pid):
    fields = (pathlib.Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[19])

detach = spec.get("detach")
if detach:
    ticket_id = detach.get("ticket_id")
    session = detach.get("session")
    if not isinstance(ticket_id, str) or not re.fullmatch(r"GPU-[\w-]{1,156}", ticket_id):
        fail("detached ticket ID is invalid")
    if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session):
        fail("detached session is invalid")
    job_dir = pathlib.Path(detach["job_dir"])
    parent = job_dir.parent.resolve(strict=True)
    if not any(inside(parent, root) for root in roots):
        fail("detached job parent escaped managed roots")
    try:
        job_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        fail("detached job identity already exists; inspect or stop it before reuse")
    job_dir = managed(str(job_dir), "detached job directory")
    log_file = job_dir / "run.log"
    identity_file = job_dir / "identity.json"
    final_file = job_dir / "final.json"
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    supervisor_pid = os.fork()
    if supervisor_pid:
        os.close(write_fd)
        ready = os.read(read_fd, 65537)
        os.close(read_fd)
        if not ready:
            _pid, wait_status = os.waitpid(supervisor_pid, 0)
            fail("detached supervisor failed before recording identity (wait=%d)" % wait_status)
        if len(ready) > 65536:
            fail("detached supervisor returned oversized identity")
        try:
            launch_value = json.loads(ready)
        except (UnicodeError, ValueError):
            os.waitpid(supervisor_pid, 0)
            fail("detached supervisor returned invalid launch identity")
        if not isinstance(launch_value, dict) or launch_value.get("status") != "launched":
            os.waitpid(supervisor_pid, 0)
            fail("detached supervisor did not confirm launch")
        sys.stdout.write(json.dumps(launch_value, sort_keys=True) + "\n")
        sys.stdout.flush()
        raise SystemExit(0)

    os.close(read_fd)
    try:
        os.setsid()
        log_descriptor = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        null_descriptor = os.open("/dev/null", os.O_RDONLY)
        os.dup2(null_descriptor, 0)
        os.dup2(log_descriptor, 1)
        os.dup2(log_descriptor, 2)
        if null_descriptor > 2:
            os.close(null_descriptor)
        if log_descriptor > 2:
            os.close(log_descriptor)
        termination = {"signal": None}
        worker = {"pid": None}
        def requested(signum, _frame):
            termination["signal"] = int(signum)
            worker_pid = worker["pid"]
            if worker_pid is not None:
                try:
                    os.killpg(worker_pid, signum)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, requested)
        signal.signal(signal.SIGINT, requested)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        supervisor_pid = os.getpid()
        boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        identity = {
            "schema_version": 1,
            "ticket_id": ticket_id,
            "session": session,
            "pid": supervisor_pid,
            "process_start_ticks": process_start_ticks(supervisor_pid),
            "boot_id": boot_id,
            "workdir": str(workdir),
            "job_dir": str(job_dir),
            "log_file": str(log_file),
            "isolation": "strict" if strict_isolation else "compatible",
            "command_sha256": hashlib.sha256(
                json.dumps(raw_argv, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "started_at_unix": int(time.time()),
        }
        worker_read_fd, worker_write_fd = os.pipe2(os.O_CLOEXEC)
        worker_pid = os.fork()
        if worker_pid == 0:
            os.close(worker_read_fd)
            try:
                os.setpgid(0, 0)
                os.write(worker_write_fd, b"1")
                os.close(worker_write_fd)
                os.execve(executable, [str(executable), *raw_argv[1:]], environment)
            except Exception as exc:
                print("managed-run worker exec failed: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
                raise SystemExit(127)
        os.close(worker_write_fd)
        worker_ready = os.read(worker_read_fd, 2)
        os.close(worker_read_fd)
        if worker_ready != b"1":
            os.waitpid(worker_pid, 0)
            fail("detached worker failed before establishing its process group")
        worker["pid"] = worker_pid
        identity["worker_pid"] = worker_pid
        try:
            atomic_json(identity_file, identity)
        except BaseException:
            try:
                os.killpg(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            os.waitpid(worker_pid, 0)
            raise
        os.write(write_fd, (json.dumps({"status": "launched", **identity}, sort_keys=True) + "\n").encode("utf-8"))
        os.close(write_fd)
        while True:
            try:
                _pid, wait_status = os.waitpid(worker_pid, 0)
                break
            except InterruptedError:
                continue
        if os.WIFEXITED(wait_status):
            returncode = os.WEXITSTATUS(wait_status)
            signal_number = None
        else:
            signal_number = os.WTERMSIG(wait_status)
            returncode = 128 + signal_number
        final = {
            "schema_version": 1,
            "ticket_id": ticket_id,
            "session": session,
            "pid": supervisor_pid,
            "worker_pid": worker_pid,
            "process_start_ticks": identity["process_start_ticks"],
            "boot_id": boot_id,
            "status": "stopped" if termination["signal"] is not None else ("completed" if returncode == 0 else "failed"),
            "returncode": returncode,
            "signal": signal_number,
            "stop_signal": termination["signal"],
            "finished_at_unix": int(time.time()),
            "log_file": str(log_file),
            "isolation": "strict" if strict_isolation else "compatible",
        }
        atomic_json(final_file, final)
        raise SystemExit(0)
    except BaseException as exc:
        try:
            os.write(write_fd, (type(exc).__name__ + ": " + str(exc)).encode("utf-8", "replace")[:65536])
        except OSError:
            pass
        raise

os.execve(executable, [str(executable), *raw_argv[1:]], environment)
'''


JOB_CONTROL = r'''
import json, os, pathlib, signal, sys, time

spec = json.loads(sys.argv[1])

def emit(value, code=0):
    print(json.dumps(value, sort_keys=True), flush=True)
    raise SystemExit(code)

def inside(path, root):
    return path == root or root in path.parents

roots = [pathlib.Path(item).resolve(strict=True) for item in spec["roots"]]
job_dir = pathlib.Path(spec["job_dir"]).resolve(strict=True)
if not job_dir.is_dir() or not any(inside(job_dir, root) for root in roots):
    emit({"status": "missing", "error": "managed job directory is unavailable"}, 3)

def read_json(path):
    if path.stat().st_size > 65536:
        raise RuntimeError(path.name + " is oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(path.name + " is not an object")
    return value

try:
    identity = read_json(job_dir / "identity.json")
except (OSError, ValueError, RuntimeError) as exc:
    emit({"status": "invalid", "error": "identity unavailable: " + str(exc)}, 3)

expected = {
    "ticket_id": spec["ticket_id"],
    "session": spec["session"],
    "workdir": spec["workdir"],
    "job_dir": str(job_dir),
}
if any(identity.get(key) != value for key, value in expected.items()):
    emit({"status": "identity_mismatch", "error": "job identity is not bound to this ticket"}, 3)
for key in ("pid", "worker_pid", "process_start_ticks"):
    if isinstance(identity.get(key), bool) or not isinstance(identity.get(key), int) or identity[key] <= 0:
        emit({"status": "invalid", "error": "job identity has invalid " + key}, 3)
boot_id = identity.get("boot_id")
if not isinstance(boot_id, str) or not boot_id:
    emit({"status": "invalid", "error": "job identity has invalid boot_id"}, 3)

final_path = job_dir / "final.json"
def final_value():
    try:
        final = read_json(final_path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RuntimeError) as exc:
        emit({"status": "invalid", "error": "final status is invalid: " + str(exc)}, 3)
    for key in ("ticket_id", "session", "pid", "worker_pid", "process_start_ticks", "boot_id"):
        if final.get(key) != identity.get(key):
            emit({"status": "identity_mismatch", "error": "final status identity changed"}, 3)
    return final

final = final_value()
if final is not None:
    emit({"status": "finished", "identity": identity, "final": final})

def live_identity():
    current_boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if current_boot != boot_id:
        return False, "remote boot ID changed"
    pid = identity["pid"]
    try:
        fields = (pathlib.Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
        state = fields[0]
        process_group = os.getpgid(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return False, "recorded supervisor is absent"
    if ticks != identity["process_start_ticks"]:
        return False, "recorded PID was reused"
    if process_group != pid:
        return False, "recorded supervisor no longer owns its process group"
    if state == "Z":
        return False, "recorded supervisor is a zombie"
    return True, "exact supervisor identity is live"

live, detail = live_identity()
if spec["action"] == "status":
    emit({"status": "running" if live else "not_running", "detail": detail, "identity": identity})
if not live:
    emit({"status": "refused", "error": detail, "identity": identity}, 3)

# Signal only the supervisor whose ticket-bound PID, start ticks, boot ID and
# process-group leadership still match.  Its handler forwards SIGTERM to the
# separately-led worker process group and remains alive to write final.json.
os.kill(identity["pid"], signal.SIGTERM)
deadline = time.monotonic() + 10.0
while time.monotonic() < deadline:
    final = final_value()
    if final is not None:
        emit({"status": "finished", "identity": identity, "final": final})
    still_live, _detail = live_identity()
    if not still_live:
        break
    time.sleep(0.1)
emit({"status": "stopping", "detail": "SIGTERM sent to exact ticket-bound process group", "identity": identity})
'''


def build_landlock_command(
    profile: dict[str, Any],
    argv: list[str],
    *,
    workdir: str,
    environment: dict[str, str] | None = None,
    runtime_directories: list[str] | None = None,
    device_ids: list[int] | None = None,
    read_exec_roots: list[str] | None = None,
    required_files: list[str] | None = None,
    detach: dict[str, str] | None = None,
    strict_isolation: bool = True,
    allow_pty: bool = False,
) -> str:
    """Return one structured remote command, optionally with Landlock."""

    require_managed_remote_path(profile, workdir, "sandbox workdir")
    runtime = managed_runtime_paths(profile)
    directories = runtime_directories or [
        value for key, value in runtime.items() if key != "condarc"
    ]
    for path in directories:
        require_managed_remote_path(profile, path, "sandbox runtime directory")
    if required_files is None:
        files: list[str] = []
    elif not isinstance(required_files, list) or any(
        not isinstance(item, str) for item in required_files
    ):
        raise ManagedRunError("required_files must be a list of paths")
    else:
        files = list(required_files)
    if not isinstance(strict_isolation, bool):
        raise ManagedRunError("strict_isolation must be a boolean")
    if not isinstance(allow_pty, bool):
        raise ManagedRunError("allow_pty must be a boolean")
    for path in files:
        require_managed_remote_path(profile, path, "required file")
    if not argv or not all(isinstance(item, str) and "\0" not in item for item in argv):
        raise ManagedRunError("sandbox argv is invalid")
    sandbox_environment = managed_runtime_environment(profile)
    sandbox_environment.update(environment or {})
    sandbox_environment.setdefault(
        "PATH", "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    )
    spec = {
        "roots": [profile["remote"]["temp_root"], profile["remote"]["durable_root"]],
        "workdir": workdir,
        "argv": argv,
        "runtime_dirs": sorted(set(directories)),
        "environment": sandbox_environment,
        "device_ids": sorted(set(device_ids or [])),
        "read_exec_roots": list(read_exec_roots or []),
        "required_files": files,
        "detach": detach,
        "strict_isolation": strict_isolation,
        "allow_pty": allow_pty,
    }
    return shlex.join(
        ["/usr/bin/python3", "-c", LANDLOCK_RUNNER, json.dumps(spec)]
    )


def build_spec(
    profile: dict[str, Any],
    ticket: dict[str, Any],
    *,
    workdir: str,
    env_prefix: str,
    script: str | None = None,
    module: str | None = None,
    arguments: list[str],
) -> dict[str, Any]:
    for field, value in (("workdir", workdir), ("env_prefix", env_prefix)):
        require_managed_remote_path(profile, value, field)
    if (script is None) == (module is None):
        raise ManagedRunError("launch requires exactly one of --script or --module")
    required_files: list[str] = []
    if script is not None:
        require_managed_remote_path(profile, script, "script")
        python_target = [script]
        required_files.append(script)
    else:
        assert module is not None
        if (
            not isinstance(module, str)
            or len(module) > 255
            or not MODULE_RE.fullmatch(module)
        ):
            raise ManagedRunError("--module must be a dotted ASCII Python module name")
        python_target = ["-m", module]
    if ticket.get("remote_workdir") != workdir:
        raise ManagedRunError("--workdir must exactly equal the ticket remote_workdir")
    if not isinstance(arguments, list) or any(
        not isinstance(item, str) for item in arguments
    ):
        raise ManagedRunError("Python arguments must be a list of strings")
    if len(arguments) > 256 or sum(len(item) for item in arguments) > 32_768:
        raise ManagedRunError("script arguments exceed the structured-run limit")
    for item in arguments:
        if "\0" in item or any(
            ord(character) < 32 and character not in "\t" for character in item
        ):
            raise ManagedRunError("script arguments contain control characters")
        candidate = (
            item.partition("=")[2]
            if item.startswith("--") and "=" in item
            else item
        )
        if candidate.startswith("/"):
            require_managed_remote_path(
                profile, candidate, "absolute Python argument"
            )
        if candidate == ".." or candidate.startswith("../"):
            raise ManagedRunError(
                "relative Python arguments may not escape the ticket workdir"
            )
    runtime = managed_runtime_paths(profile)
    run_root = runtime["base"] + "/runs/" + ticket["id"]
    paths = {**runtime, "run": run_root}
    for key, value in paths.items():
        if key == "condarc":
            continue
        require_managed_remote_path(profile, value, "runtime path")
    environment = managed_runtime_environment(profile)
    environment.update({
        "PATH": f"{env_prefix}/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": ",".join(str(item) for item in ticket["assigned_gpus"]),
        "REMOTE_GPU_DEV_TICKET": ticket["id"],
    })
    environment.update(profile["gpu"]["environment"])
    if profile["network"].get("hf_endpoint"):
        environment["HF_ENDPOINT"] = profile["network"]["hf_endpoint"]
    if profile["network"].get("pip_index_url"):
        environment["PIP_INDEX_URL"] = profile["network"]["pip_index_url"]
    if profile["network"].get("pip_extra_index_urls"):
        environment["PIP_EXTRA_INDEX_URL"] = " ".join(
            profile["network"]["pip_extra_index_urls"]
        )
    return {
        "workdir": workdir,
        "argv": [f"{env_prefix}/bin/python", *python_target, *arguments],
        "required_files": required_files,
        "runtime_directories": sorted(
            value for key, value in paths.items() if key != "condarc"
        ),
        "environment": environment,
    }


def _session(value: str | None) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,100}", value
    ):
        raise ManagedRunError(
            "--session must use 1-100 ASCII letters, digits, _ or -"
        )
    return value


def _module(value: str) -> str:
    if len(value) > 255 or not MODULE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "module must be a dotted ASCII Python module name"
        )
    return value


def _ticket_workdir(
    profile: dict[str, Any], ticket: dict[str, Any], explicit: str | None
) -> str:
    recorded = ticket.get("remote_workdir")
    if not isinstance(recorded, str):
        raise ManagedRunError("ticket has no remote_workdir")
    recorded = require_managed_remote_path(
        profile, recorded, "ticket remote_workdir"
    )
    if explicit is not None:
        provided = require_managed_remote_path(profile, explicit, "--workdir")
        if provided != recorded:
            raise ManagedRunError("--workdir must exactly equal the ticket remote_workdir")
    return recorded


def _ticket_session(ticket: dict[str, Any], explicit: str | None) -> str:
    recorded = _session(ticket.get("session"))
    if explicit is not None and _session(explicit) != recorded:
        raise ManagedRunError("--session must exactly equal the ticket session")
    return recorded


def _job_directory(
    profile: dict[str, Any], ticket_id: str, session: str
) -> str:
    path = (
        managed_runtime_paths(profile)["base"]
        + "/runs/"
        + ticket_id
        + "/jobs/"
        + session
    )
    return require_managed_remote_path(profile, path, "managed job directory")


def _control_job(
    profile: dict[str, Any],
    ticket: dict[str, Any],
    *,
    session: str,
    action: str,
    workdir: str | None = None,
) -> int:
    if ticket.get("session") != session:
        raise ManagedRunError("--session must exactly equal the ticket session")
    workdir = _ticket_workdir(profile, ticket, workdir)
    job_dir = _job_directory(profile, ticket["id"], session)
    control_spec = {
        "action": action,
        "roots": [
            profile["remote"]["temp_root"],
            profile["remote"]["durable_root"],
        ],
        "ticket_id": ticket["id"],
        "session": session,
        "workdir": workdir,
        "job_dir": job_dir,
    }
    remote_command = build_landlock_command(
        profile,
        [
            "/usr/bin/python3",
            "-I",
            "-B",
            "-c",
            JOB_CONTROL,
            json.dumps(control_spec, separators=(",", ":")),
        ],
        workdir=profile["remote"]["temp_root"],
        device_ids=ticket.get("assigned_gpus", []),
    )
    argv = ssh_argv(profile, batch=True)
    argv.extend(
        [f"{profile['ssh']['user']}@{profile['ssh']['host']}", remote_command]
    )
    return subprocess.run(argv, check=False).returncode


def execute(profile: dict[str, Any], args: argparse.Namespace) -> int:
    action = "stop" if args.stop else ("status" if args.status else "launch")
    if action == "status":
        ticket = _ticket(
            profile,
            args.ticket_id,
            allowed_statuses={
                "running", "stale", "completed", "failed", "cancelled"
            },
        )
    elif action == "stop":
        ticket = _ticket(
            profile, args.ticket_id, allowed_statuses={"running", "stale"}
        )
    else:
        ticket = _ticket(profile, args.ticket_id)
    if action in {"status", "stop"}:
        session = _ticket_session(ticket, args.session)
        if any(value is not None for value in (args.env_prefix, args.script, args.module)):
            raise ManagedRunError("status/stop do not accept launch target options")
        if args.arguments:
            raise ManagedRunError("status/stop do not accept Python arguments")
        return _control_job(
            profile, ticket, session=session, action=action, workdir=args.workdir
        )
    missing = [
        option
        for option, value in (
            ("--env-prefix", args.env_prefix),
        )
        if value is None
    ]
    if missing:
        raise ManagedRunError("launch requires " + ", ".join(missing))
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    workdir = _ticket_workdir(profile, ticket, args.workdir)
    spec = build_spec(
        profile,
        ticket,
        workdir=workdir,
        env_prefix=args.env_prefix,
        script=args.script,
        module=args.module,
        arguments=arguments,
    )
    spec["environment"]["REMOTE_GPU_DEV_ISOLATION"] = "compatible"
    detach = None
    if args.session is not None:
        session = _session(args.session)
        if ticket.get("session") != session:
            raise ManagedRunError("--session must exactly equal the ticket session")
        detach = {
            "ticket_id": ticket["id"],
            "session": session,
            "job_dir": _job_directory(profile, ticket["id"], session),
        }
    remote_command = build_landlock_command(
        profile,
        spec["argv"],
        workdir=spec["workdir"],
        environment=spec["environment"],
        runtime_directories=spec["runtime_directories"],
        device_ids=ticket["assigned_gpus"],
        required_files=spec["required_files"],
        detach=detach,
        strict_isolation=False,
    )
    argv = ssh_argv(profile, batch=True)
    argv.extend(
        [
            f"{profile['ssh']['user']}@{profile['ssh']['host']}",
            remote_command,
        ]
    )
    return subprocess.run(argv, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticket_id")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--status", action="store_true", help="inspect an exact detached job")
    operation.add_argument("--stop", action="store_true", help="SIGTERM an exact detached job")
    parser.add_argument("--workdir")
    parser.add_argument("--env-prefix")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--script")
    target.add_argument(
        "--module",
        type=_module,
        help="run a dotted Python module name with the selected environment",
    )
    parser.add_argument(
        "--session",
        help="launch or control the exact ticket-bound detached job identity",
    )
    parser.add_argument(
        "arguments",
        nargs="*",
        help="script arguments; put -- before option-like training arguments",
    )
    return parser


def parse_command_line(argv: list[str] | None = None) -> argparse.Namespace:
    """Allow launch options after the ticket while preserving `--` arguments."""

    return build_parser().parse_intermixed_args(argv)


def main() -> int:
    try:
        profile = load_profile()
        return execute(profile, parse_command_line())
    except (ProfileError, RemotePathError, SSHError, ManagedRunError) as exc:
        print(f"managed-run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
