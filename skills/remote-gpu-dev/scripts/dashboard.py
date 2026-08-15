#!/usr/bin/env python3
"""Local dashboard for remote GPU telemetry, tickets, and TensorBoard."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import selectors
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from profile import ProfileError, dashboard_runtime_dir, load_profile, public_profile
from remote_path_guard import managed_runtime_paths
from ssh_remote import SSHError, ssh_argv
from managed_run import ManagedRunError, build_landlock_command


SCRIPT = Path(__file__).resolve()
SKILL_ROOT = SCRIPT.parent.parent
PROJECT_ROOT = Path.home()
ASSET_ROOT = SKILL_ROOT / "assets" / "dashboard"
TICKET_TOOL = SKILL_ROOT / "scripts" / "gpu_ticket.py"
TENSORBOARD_SIDECAR = SKILL_ROOT / "scripts" / "tensorboard_sidecar.py"
REMOTE_PYTHON = "/usr/bin/python3"
DEFAULT_PORT = 8765
SAMPLE_INTERVAL_SECONDS = 2.0
PROFILE: dict[str, Any] | None = None
ACTIVE_STATES = {"reserved", "running", "stale"}
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}
MAX_LINE_BYTES = 1_000_000
MAX_PROXY_REQUEST_PATH_BYTES = 16_384
MAX_TENSORBOARD_POST_BYTES = 1_048_576
MAX_CONTROL_POST_BYTES = 1024
SIDECAR_COMMAND_TIMEOUT_SECONDS = 45
OWNED_SIDECAR_CLEANUP_SECONDS = 25
LOCAL_PROCESS_CLEANUP_SECONDS = 5
STATE_THREAD_JOIN_SECONDS = 2
DASHBOARD_STOP_GRACE_SECONDS = 90
MAX_PARALLEL_SIDECAR_CLEANUPS = 128
TENSORBOARD_TICKET_RE = re.compile(r"GPU-[\w-]{1,156}\Z", flags=re.UNICODE)
TENSORBOARD_EXPERIMENT_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
MULTIPART_FORM_DATA_RE = re.compile(
    r"multipart/form-data[ \t]*;[ \t]*boundary=(?:"
    r"(?P<bare>[0-9A-Za-z'()+_,./:=?-]{1,70})|"
    r'"(?P<quoted>[0-9A-Za-z\'()+_,./:=?-]{1,70})")'
    r"[ \t]*\Z",
    flags=re.IGNORECASE,
)
TENSORBOARD_STATES = {
    "starting",
    "live",
    "stopped",
    "failed",
    "cleanup_pending",
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


REMOTE_COLLECTOR = r'''
import json
import math
import os
import socket
import time

import nvitop
from nvitop import Device


def scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def call(obj, method):
    try:
        return getattr(obj, method)()
    except Exception:
        return None


def text(value, limit=120):
    if value is None:
        return None
    rendered = str(value).replace("\x00", "").strip()
    return rendered[:limit]


devices = Device.all()
hello_gpus = []
for device in devices:
    hello_gpus.append(
        {
            "index": int(device.physical_index),
            "uuid": text(call(device, "uuid")),
            "name": text(call(device, "name")),
        }
    )

hello = {
    "type": "hello",
    "protocol": 1,
    "nvitop_version": nvitop.__version__,
    "hostname": socket.gethostname(),
    "boot_id": open("/proc/sys/kernel/random/boot_id", encoding="utf-8").read().strip(),
    "gpus": hello_gpus,
}
print(json.dumps(hello, ensure_ascii=False, allow_nan=False), flush=True)

interval = float("__REMOTE_GPU_SAMPLE_INTERVAL__")
sequence = 0
while True:
    started = time.monotonic()
    rows = []
    for device in devices:
        row = {
            "index": int(device.physical_index),
            "uuid": text(call(device, "uuid")),
            "name": text(call(device, "name")),
            "utilization": scalar(call(device, "gpu_utilization")),
            "memory_percent": scalar(call(device, "memory_percent")),
            "temperature_c": scalar(call(device, "temperature")),
            "fan_percent": scalar(call(device, "fan_speed")),
            "power_mw": scalar(call(device, "power_usage")),
            "power_limit_mw": scalar(call(device, "power_limit")),
            "processes": [],
        }
        memory = call(device, "memory_info")
        if memory is not None:
            row.update(
                {
                    "memory_total_bytes": scalar(getattr(memory, "total", None)),
                    "memory_used_bytes": scalar(getattr(memory, "used", None)),
                    "memory_free_bytes": scalar(getattr(memory, "free", None)),
                    "memory_reserved_bytes": scalar(getattr(memory, "reserved", None)),
                }
            )
        processes = call(device, "processes") or {}
        for process in list(processes.values())[:128]:
            try:
                pid = int(process.pid)
            except (TypeError, ValueError):
                continue
            row["processes"].append(
                {
                    "pid": pid,
                    "username": text(call(process, "username"), 80),
                    "type": text(call(process, "type"), 32),
                    "gpu_memory_bytes": scalar(call(process, "gpu_memory")),
                    "gpu_sm_utilization": scalar(call(process, "gpu_sm_utilization")),
                }
            )
        row["processes"].sort(key=lambda item: item["pid"])
        rows.append(row)

    sample = {
        "type": "sample",
        "protocol": 1,
        "seq": sequence,
        "sampled_at": time.time(),
        "gpus": rows,
    }
    sequence += 1
    try:
        print(json.dumps(sample, ensure_ascii=False, allow_nan=False), flush=True)
    except BrokenPipeError:
        break
    time.sleep(max(0.0, interval - (time.monotonic() - started)))
'''


def configure_profile() -> None:
    global PROFILE, PROJECT_ROOT, REMOTE_PYTHON, DEFAULT_PORT, SAMPLE_INTERVAL_SECONDS
    try:
        PROFILE = load_profile()
    except ProfileError as exc:
        raise RuntimeError(str(exc)) from exc
    PROJECT_ROOT = Path(PROFILE["local"]["projects_root"])
    REMOTE_PYTHON = str(PROFILE["remote"]["monitor_python"])
    DEFAULT_PORT = int(PROFILE["dashboard"]["local_port"])
    SAMPLE_INTERVAL_SECONDS = float(PROFILE["dashboard"]["sample_interval_seconds"])


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def runtime_dir() -> Path:
    if PROFILE is None:
        raise RuntimeError("dashboard profile is not configured")
    base = dashboard_runtime_dir(PROFILE)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    return base


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def process_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [item.decode("utf-8", "replace") for item in raw.split(b"\0") if item]


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        SCRIPT,
        ASSET_ROOT / "index.html",
        ASSET_ROOT / "app.css",
        ASSET_ROOT / "app.js",
    ):
        digest.update(str(path.relative_to(SKILL_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if PROFILE is None:
        raise RuntimeError("dashboard profile is not configured")
    digest.update(
        json.dumps(
            public_profile(PROFILE),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def metadata_path() -> Path:
    return runtime_dir() / "runtime.json"


def load_metadata() -> dict[str, Any] | None:
    try:
        value = json.loads(metadata_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write_metadata(value: dict[str, Any]) -> None:
    target = metadata_path()
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=".runtime-", delete=False
    )
    temporary = Path(handle.name)
    try:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, target)
    finally:
        try:
            handle.close()
        except OSError:
            pass
        if temporary.exists():
            temporary.unlink()


def dashboard_url(meta: dict[str, Any]) -> str:
    return f"http://127.0.0.1:{int(meta['port'])}/{meta['token']}/"


def process_identity_matches(meta: dict[str, Any]) -> bool:
    try:
        pid = int(meta["pid"])
        expected_ticks = int(meta["starttime_ticks"])
        expected_uid = int(meta["uid"])
    except (KeyError, TypeError, ValueError):
        return False
    if expected_uid != os.getuid() or meta.get("boot_id") != boot_id():
        return False
    if process_start_ticks(pid) != expected_ticks:
        return False
    cmdline = process_cmdline(pid)
    return str(SCRIPT) in cmdline and "_serve" in cmdline


def health_matches(meta: dict[str, Any], timeout: float = 1.0) -> bool:
    try:
        port = int(meta["port"])
        token = str(meta["token"])
        expected_instance = str(meta["instance_id"])
        # Never let ambient proxy variables receive the capability token.
        # HTTPConnection is a direct loopback socket, unlike urlopen's default
        # environment-aware proxy handler.
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            connection.request(
                "GET",
                f"/{token}/health",
                headers={"Host": f"127.0.0.1:{port}"},
            )
            response = connection.getresponse()
            if response.status != 200:
                return False
            payload = json.loads(response.read(64 * 1024))
        finally:
            connection.close()
        return payload.get("instance_id") == expected_instance
    except (KeyError, TypeError, ValueError, OSError, http.client.HTTPException, json.JSONDecodeError):
        return False


def is_live(meta: dict[str, Any] | None) -> bool:
    return bool(meta and process_identity_matches(meta) and health_matches(meta))


def clipped_text(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    rendered = str(value).replace("\x00", "").strip()
    return rendered[:limit]


def finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        converted = float(value)
        if math.isfinite(converted):
            return int(value) if isinstance(value, int) else converted
    return None


def integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted


def tensorboard_path_prefix(ticket_id: str) -> str | None:
    if not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
        return None
    return f"/tb/{quote(ticket_id, safe='')}"


def decode_tensorboard_ticket_segment(segment: str) -> str | None:
    """Decode exactly one canonical UTF-8 path segment into a ledger ID."""
    try:
        ticket_id = unquote(segment, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if quote(ticket_id, safe="") != segment:
        return None
    return ticket_id if TENSORBOARD_TICKET_RE.fullmatch(ticket_id) else None


def decode_tensorboard_experiment_segment(segment: str) -> str | None:
    try:
        experiment_id = unquote(segment, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if quote(experiment_id, safe="") != segment:
        return None
    return (
        experiment_id
        if TENSORBOARD_EXPERIMENT_RE.fullmatch(experiment_id)
        else None
    )


def tensorboard_timeseries_post_target(path: str) -> tuple[str, str] | None:
    pieces = path.split("/")
    if len(pieces) != 9 or pieces[:2] != ["", "tb"]:
        return None
    if pieces[3] != "experiment" or pieces[5:] != [
        "data",
        "plugin",
        "timeseries",
        "timeSeries",
    ]:
        return None
    ticket_id = decode_tensorboard_ticket_segment(pieces[2])
    experiment_id = decode_tensorboard_experiment_segment(pieces[4])
    if ticket_id is None or experiment_id is None:
        return None
    return ticket_id, experiment_id


def tensorboard_scalars_multirun_post_ticket(path: str) -> str | None:
    """Return the ticket for TensorBoard's classic read-only scalar batch API."""
    pieces = path.split("/")
    if len(pieces) != 7 or pieces[:2] != ["", "tb"]:
        return None
    if pieces[3:] != ["data", "plugin", "scalars", "scalars_multirun"]:
        return None
    return decode_tensorboard_ticket_segment(pieces[2])


def multipart_form_data_boundary(value: str) -> str | None:
    if not isinstance(value, str) or len(value) > 160:
        return None
    match = MULTIPART_FORM_DATA_RE.fullmatch(value)
    return (match.group("bare") or match.group("quoted")) if match else None


def sanitize_tensorboard(value: Any, ticket_id: str | None) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not ticket_id or not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
        return None
    status = clipped_text(value.get("status"), 32)
    path_prefix = clipped_text(value.get("path_prefix"), 4096)
    remote_port = integer(value.get("remote_port"))
    generation = integer(value.get("generation"))
    expected_prefix = tensorboard_path_prefix(ticket_id)
    if status not in TENSORBOARD_STATES:
        return None
    if path_prefix != expected_prefix:
        path_prefix = None
    if remote_port is not None and not 1024 <= remote_port <= 65535:
        remote_port = None
    if generation is not None and generation <= 0:
        generation = None
    return {
        "status": status,
        "path_prefix": path_prefix,
        "remote_port": remote_port,
        "generation": generation,
        "logdir": clipped_text(value.get("logdir"), 320),
        "version": clipped_text(value.get("version"), 64),
        "started_at": clipped_text(value.get("started_at"), 64),
        "last_health_at": clipped_text(value.get("last_health_at"), 64),
        "stopped_at": clipped_text(value.get("stopped_at"), 64),
        "last_error": clipped_text(value.get("last_error"), 240),
        "viewer_status": "idle" if status == "live" and path_prefix and remote_port else status,
    }


def sanitize_ticket(ticket: Any) -> dict[str, Any] | None:
    if not isinstance(ticket, dict):
        return None
    raw_ticket_id = ticket.get("id")
    ticket_id = clipped_text(raw_ticket_id, 160)
    tensorboard_ticket_id = (
        raw_ticket_id
        if isinstance(raw_ticket_id, str)
        and TENSORBOARD_TICKET_RE.fullmatch(raw_ticket_id)
        else None
    )
    assigned = [item for item in (integer(value) for value in ticket.get("assigned_gpus", [])) if item is not None]
    return {
        "id": ticket_id,
        "status": clipped_text(ticket.get("status"), 32),
        "project": clipped_text(ticket.get("project"), 120),
        "owner": clipped_text(ticket.get("owner"), 120),
        "purpose": clipped_text(ticket.get("purpose"), 240),
        "assigned_gpus": assigned,
        "requested_gpus": integer(ticket.get("requested_gpus")),
        "created_at": clipped_text(ticket.get("created_at"), 64),
        "started_at": clipped_text(ticket.get("started_at"), 64),
        "updated_at": clipped_text(ticket.get("updated_at"), 64),
        "last_heartbeat_at": clipped_text(ticket.get("last_heartbeat_at"), 64),
        "expected_end_at": clipped_text(ticket.get("expected_end_at"), 64),
        "result": clipped_text(ticket.get("result"), 320),
        "tensorboard": sanitize_tensorboard(
            ticket.get("tensorboard"), tensorboard_ticket_id
        ),
    }


def sanitize_ticket_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ticket status is not an object")
    tickets = []
    for raw in value.get("tickets", []):
        item = sanitize_ticket(raw)
        if item and item["id"]:
            tickets.append(item)
    active = [item for item in tickets if item["status"] in ACTIVE_STATES]
    queued = [item for item in tickets if item["status"] == "queued"]
    terminal = [item for item in tickets if item["status"] in TERMINAL_STATES]
    terminal.sort(key=lambda item: (item.get("updated_at") or "", item.get("id") or ""), reverse=True)
    pending = []
    for item in value.get("pending_reconcile", []):
        if isinstance(item, dict):
            pending.append({str(key)[:80]: clipped_text(val, 200) for key, val in list(item.items())[:12]})
        else:
            rendered = clipped_text(item, 200)
            if rendered:
                pending.append(rendered)
    gpu_ids = [item for item in (integer(value) for value in value.get("gpu_ids", [])) if item is not None]
    occupied = [item for item in (integer(value) for value in value.get("occupied_gpus", [])) if item is not None]
    return {
        "server": clipped_text(value.get("server"), 160),
        "profile": clipped_text(value.get("profile"), 48),
        "ledger_profile": clipped_text(value.get("ledger_profile"), 48),
        "coordination_uid": clipped_text(value.get("coordination_uid"), 80),
        "gpu_ids": sorted(set(gpu_ids)),
        "occupied_gpus": sorted(set(occupied)),
        "pending_reconcile": pending[:32],
        "active": active,
        "queued": queued,
        "recent": terminal[:12],
        "history": terminal,
        "snapshot_note": clipped_text(value.get("snapshot_note"), 200),
    }


def sanitize_gpu(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    index = integer(value.get("index"))
    if index is None:
        return None
    processes = []
    for raw in value.get("processes", [])[:128]:
        if not isinstance(raw, dict):
            continue
        pid = integer(raw.get("pid"))
        if pid is None:
            continue
        processes.append(
            {
                "pid": pid,
                "username": clipped_text(raw.get("username"), 80),
                "type": clipped_text(raw.get("type"), 32),
                "gpu_memory_bytes": finite_number(raw.get("gpu_memory_bytes")),
                "gpu_sm_utilization": finite_number(raw.get("gpu_sm_utilization")),
            }
        )
    return {
        "index": index,
        "uuid": clipped_text(value.get("uuid"), 120),
        "name": clipped_text(value.get("name"), 120),
        "utilization": finite_number(value.get("utilization")),
        "memory_percent": finite_number(value.get("memory_percent")),
        "memory_total_bytes": finite_number(value.get("memory_total_bytes")),
        "memory_used_bytes": finite_number(value.get("memory_used_bytes")),
        "memory_free_bytes": finite_number(value.get("memory_free_bytes")),
        "memory_reserved_bytes": finite_number(value.get("memory_reserved_bytes")),
        "temperature_c": finite_number(value.get("temperature_c")),
        "fan_percent": finite_number(value.get("fan_percent")),
        "power_mw": finite_number(value.get("power_mw")),
        "power_limit_mw": finite_number(value.get("power_limit_mw")),
        "processes": processes,
    }


def sanitize_remote(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("protocol") != 1:
        raise ValueError("unsupported remote collector message")
    message_type = value.get("type")
    if message_type == "hello":
        return {
            "type": "hello",
            "protocol": 1,
            "nvitop_version": clipped_text(value.get("nvitop_version"), 32),
            "hostname": clipped_text(value.get("hostname"), 120),
            "boot_id": clipped_text(value.get("boot_id"), 80),
            "gpus": [item for item in (sanitize_gpu(raw) for raw in value.get("gpus", [])) if item],
        }
    if message_type == "sample":
        return {
            "type": "sample",
            "protocol": 1,
            "seq": integer(value.get("seq")),
            "sampled_at": finite_number(value.get("sampled_at")),
            "gpus": [item for item in (sanitize_gpu(raw) for raw in value.get("gpus", [])) if item],
        }
    raise ValueError("unknown remote collector message")


class DashboardState:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ticket: dict[str, Any] = {"connected": False, "error": "waiting for ticket snapshot"}
        self.remote: dict[str, Any] = {"connected": False, "error": "waiting for nvitop"}
        self.ssh_process: subprocess.Popen[str] | None = None
        self.tensorboard_tunnels: dict[str, dict[str, Any]] = {}
        self.tensorboard_owned_generations: dict[str, int] = {}
        self.tensorboard_control_lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for target, name in ((self._ticket_loop, "ticket-poller"), (self._remote_loop, "nvitop-stream")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.stop_event.set()
        with self.tensorboard_control_lock:
            with self.lock:
                process = self.ssh_process
                tunnel_processes = [
                    item.get("process") for item in self.tensorboard_tunnels.values()
                ]
                owned_generations = dict(self.tensorboard_owned_generations)
                self.tensorboard_tunnels = {}
                self.tensorboard_owned_generations = {}
            local_processes = [
                candidate
                for candidate in [process, *tunnel_processes]
                if isinstance(candidate, subprocess.Popen)
            ]
            self._terminate_processes(
                local_processes,
                timeout=LOCAL_PROCESS_CLEANUP_SECONDS,
            )
            join_deadline = time.monotonic() + STATE_THREAD_JOIN_SECONDS
            for thread in self.threads:
                thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
            self._stop_owned_tensorboards(owned_generations)

    @staticmethod
    def _terminate_processes(
        processes: list[subprocess.Popen[Any]], timeout: float
    ) -> None:
        live = [process for process in processes if process.poll() is None]
        for process in live:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while live and time.monotonic() < deadline:
            live = [process for process in live if process.poll() is None]
            if live:
                time.sleep(0.05)
        for process in live:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _terminate_process(process: subprocess.Popen[Any], timeout: float = 2) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _ticket_items(snapshot: Any) -> list[dict[str, Any]]:
        if not isinstance(snapshot, dict):
            return []
        items: dict[str, dict[str, Any]] = {}
        for group in ("active", "queued", "history"):
            for item in snapshot.get(group, []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    items[item["id"]] = item
        return list(items.values())

    def _ticket_item(self, ticket_id: str) -> dict[str, Any] | None:
        with self.lock:
            snapshot = copy.deepcopy(self.ticket.get("snapshot"))
        for item in self._ticket_items(snapshot):
            if item.get("id") == ticket_id:
                return item
        return None

    @staticmethod
    def _sidecar_generation(payload: dict[str, Any]) -> int | None:
        candidates = [payload.get("tensorboard")]
        ticket = payload.get("ticket")
        if isinstance(ticket, dict):
            candidates.append(ticket.get("tensorboard"))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            generation = integer(candidate.get("generation"))
            if generation is not None and generation > 0:
                return generation
        return None

    @staticmethod
    def _sidecar_message(value: Any, fallback: str) -> str:
        rendered = clipped_text(value, 300)
        return rendered or fallback

    def _run_sidecar(
        self,
        arguments: list[str],
        *,
        timeout: int = SIDECAR_COMMAND_TIMEOUT_SECONDS,
    ) -> tuple[int, dict[str, Any]]:
        try:
            completed = subprocess.run(
                [sys.executable, str(TENSORBOARD_SIDECAR), *arguments],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 504, {"ok": False, "error": "TensorBoard 操作超时。"}
        except OSError:
            return 502, {"ok": False, "error": "无法启动 TensorBoard 管理程序。"}

        payload: dict[str, Any] | None = None
        try:
            decoded = json.loads(completed.stdout)
            if isinstance(decoded, dict):
                payload = decoded
        except (json.JSONDecodeError, TypeError):
            pass
        if completed.returncode != 0:
            message = self._sidecar_message(
                completed.stderr,
                "TensorBoard 管理程序拒绝了该操作。",
            )
            return 409, {"ok": False, "error": message}
        if payload is None:
            return 502, {"ok": False, "error": "TensorBoard 管理程序返回了无效响应。"}
        return 200, payload

    def open_tensorboard(self, ticket_id: str) -> tuple[int, dict[str, Any]]:
        if not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
            return 400, {"ok": False, "error": "无效的工单 ID。"}
        if self.stop_event.is_set():
            return 503, {"ok": False, "error": "看板正在停止。"}
        if not self.tensorboard_control_lock.acquire(blocking=False):
            return 409, {"ok": False, "error": "另一个 TensorBoard 操作正在进行。"}
        try:
            if self.stop_event.is_set():
                return 503, {"ok": False, "error": "看板正在停止。"}
            item = self._ticket_item(ticket_id)
            if item is None:
                return 404, {"ok": False, "error": "工单不在当前账本快照中。"}
            metadata = item.get("tensorboard")
            if not isinstance(metadata, dict):
                return 409, {"ok": False, "error": "该工单尚未配置 TensorBoard。"}
            status, payload = self._run_sidecar(["start", ticket_id])
            if status != 200:
                return status, payload
            if payload.get("status") != "live":
                return 502, {"ok": False, "error": "TensorBoard 未返回 live 状态。"}
            generation = self._sidecar_generation(payload)
            if generation is None:
                return 502, {"ok": False, "error": "TensorBoard 响应缺少 generation。"}
            created = not bool(payload.get("idempotent"))
            if created:
                with self.lock:
                    self.tensorboard_owned_generations[ticket_id] = generation
            return 200, {
                "ok": True,
                "ticket_id": ticket_id,
                "status": "live",
                "generation": generation,
                "created_by_dashboard": created,
            }
        finally:
            self.tensorboard_control_lock.release()

    def close_tensorboard(self, ticket_id: str) -> tuple[int, dict[str, Any]]:
        if not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
            return 400, {"ok": False, "error": "无效的工单 ID。"}
        if self.stop_event.is_set():
            return 503, {"ok": False, "error": "看板正在停止。"}
        if not self.tensorboard_control_lock.acquire(blocking=False):
            return 409, {"ok": False, "error": "另一个 TensorBoard 操作正在进行。"}
        try:
            if self.stop_event.is_set():
                return 503, {"ok": False, "error": "看板正在停止。"}
            item = self._ticket_item(ticket_id)
            if item is None:
                return 404, {"ok": False, "error": "工单不在当前账本快照中。"}
            metadata = item.get("tensorboard")
            if not isinstance(metadata, dict):
                return 409, {"ok": False, "error": "该工单尚未配置 TensorBoard。"}
            with self.lock:
                expected_generation = self.tensorboard_owned_generations.get(ticket_id)
            if expected_generation is None:
                expected_generation = integer(metadata.get("generation"))
            if expected_generation is None or expected_generation <= 0:
                return 409, {"ok": False, "error": "当前快照缺少有效 generation。"}
            status, payload = self._run_sidecar(
                [
                    "stop",
                    ticket_id,
                    "--expected-generation",
                    str(expected_generation),
                ]
            )
            observed_status = payload.get("status")
            if status == 200 and observed_status == "superseded":
                with self.lock:
                    if self.tensorboard_owned_generations.get(ticket_id) == expected_generation:
                        self.tensorboard_owned_generations.pop(ticket_id, None)
                self.drop_tensorboard_tunnel(ticket_id)
                return 409, {
                    "ok": False,
                    "ticket_id": ticket_id,
                    "status": "superseded",
                    "error": "TensorBoard 已出现更新 generation；未停止新实例。",
                }
            if status != 200:
                return status, payload
            if observed_status not in {"stopped", "not_configured"}:
                return 502, {"ok": False, "error": "TensorBoard 未确认停止。"}
            with self.lock:
                if self.tensorboard_owned_generations.get(ticket_id) == expected_generation:
                    self.tensorboard_owned_generations.pop(ticket_id, None)
            self.drop_tensorboard_tunnel(ticket_id)
            return 200, {
                "ok": True,
                "ticket_id": ticket_id,
                "status": "stopped",
                "generation": expected_generation,
            }
        finally:
            self.tensorboard_control_lock.release()

    def _stop_owned_tensorboards(self, owned_generations: dict[str, int]) -> None:
        if not owned_generations:
            return

        deadline = time.monotonic() + OWNED_SIDECAR_CLEANUP_SECONDS

        def stop_one(item: tuple[str, int]) -> tuple[str, int, dict[str, Any]]:
            ticket_id, generation = item
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ticket_id, 504, {
                    "ok": False,
                    "error": "TensorBoard cleanup deadline elapsed before dispatch.",
                }
            status, payload = self._run_sidecar(
                [
                    "stop",
                    ticket_id,
                    "--expected-generation",
                    str(generation),
                    "--stop-timeout",
                    "5",
                ],
                timeout=max(1, int(remaining + 0.999)),
            )
            return ticket_id, status, payload

        workers = min(MAX_PARALLEL_SIDECAR_CLEANUPS, len(owned_generations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(stop_one, owned_generations.items()))
        for ticket_id, status, payload in results:
            if status == 200 and payload.get("status") in {
                "stopped",
                "not_configured",
                "superseded",
            }:
                continue
            message = self._sidecar_message(
                payload.get("error"),
                "TensorBoard 清理未确认。",
            )
            print(
                f"dashboard TensorBoard cleanup failed for {ticket_id}: {message}",
                file=sys.stderr,
                flush=True,
            )

    def _prune_tensorboard_tunnels(self, snapshot: Any) -> None:
        desired: dict[str, int] = {}
        observed_generations: dict[str, int] = {}
        observed_statuses: dict[str, str] = {}
        for item in self._ticket_items(snapshot):
            tensorboard = item.get("tensorboard")
            if not isinstance(tensorboard, dict):
                continue
            remote_port = integer(tensorboard.get("remote_port"))
            generation = integer(tensorboard.get("generation"))
            status = tensorboard.get("status")
            path_prefix = tensorboard.get("path_prefix")
            if generation is not None and generation > 0:
                observed_generations[item["id"]] = generation
                if isinstance(status, str):
                    observed_statuses[item["id"]] = status
            if (
                status == "live"
                and remote_port
                and path_prefix == tensorboard_path_prefix(item["id"])
            ):
                desired[item["id"]] = remote_port
        stale: list[subprocess.Popen[Any]] = []
        with self.lock:
            for ticket_id, generation in list(
                self.tensorboard_owned_generations.items()
            ):
                observed_generation = observed_generations.get(ticket_id)
                if (
                    observed_generation is not None
                    and (
                        observed_generation > generation
                        or (
                            observed_generation == generation
                            and observed_statuses.get(ticket_id) == "stopped"
                        )
                    )
                ):
                    self.tensorboard_owned_generations.pop(ticket_id, None)
            for ticket_id, tunnel in list(self.tensorboard_tunnels.items()):
                process = tunnel.get("process")
                if (
                    desired.get(ticket_id) != tunnel.get("remote_port")
                    or not isinstance(process, subprocess.Popen)
                    or process.poll() is not None
                ):
                    self.tensorboard_tunnels.pop(ticket_id, None)
                    if isinstance(process, subprocess.Popen) and process.poll() is None:
                        stale.append(process)
        for process in stale:
            self._terminate_process(process)

    @staticmethod
    def _unused_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    @staticmethod
    def _port_accepting(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.15):
                return True
        except OSError:
            return False

    def _find_tensorboard(self, ticket_id: str) -> dict[str, Any] | None:
        with self.lock:
            snapshot = copy.deepcopy(self.ticket.get("snapshot"))
        for item in self._ticket_items(snapshot):
            if item.get("id") != ticket_id:
                continue
            tensorboard = item.get("tensorboard")
            if not isinstance(tensorboard, dict):
                return None
            if tensorboard.get("status") != "live":
                return None
            if tensorboard.get("path_prefix") != tensorboard_path_prefix(ticket_id):
                return None
            remote_port = integer(tensorboard.get("remote_port"))
            if remote_port is None or not 1024 <= remote_port <= 65535:
                return None
            return tensorboard
        return None

    def tensorboard_upstream(self, ticket_id: str) -> tuple[int | None, str | None]:
        if not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
            return None, "invalid TensorBoard ticket"
        tensorboard = self._find_tensorboard(ticket_id)
        if tensorboard is None:
            return None, "TensorBoard is not registered as live for this ticket"
        remote_port = int(tensorboard["remote_port"])
        with self.lock:
            existing = self.tensorboard_tunnels.get(ticket_id)
            if existing:
                process = existing.get("process")
                if (
                    existing.get("remote_port") == remote_port
                    and isinstance(process, subprocess.Popen)
                    and process.poll() is None
                    and self._port_accepting(int(existing["local_port"]))
                ):
                    return int(existing["local_port"]), None

        last_error = "SSH tunnel failed to start"
        for _attempt in range(5):
            local_port = self._unused_loopback_port()
            process = subprocess.Popen(
                [
                    str(SSH_HELPER),
                    "--batch",
                    "--local-forward",
                    f"{local_port}:{remote_port}",
                    "--no-command",
                ],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and process.poll() is None:
                if self._port_accepting(local_port):
                    tunnel = {
                        "process": process,
                        "local_port": local_port,
                        "remote_port": remote_port,
                        "connected_at": utc_now(),
                    }
                    with self.lock:
                        winner = self.tensorboard_tunnels.get(ticket_id)
                        if winner:
                            winner_process = winner.get("process")
                            if isinstance(winner_process, subprocess.Popen) and winner_process.poll() is None:
                                self._terminate_process(process)
                                return int(winner["local_port"]), None
                        self.tensorboard_tunnels[ticket_id] = tunnel
                    return local_port, None
                time.sleep(0.1)
            if process.poll() is None:
                self._terminate_process(process)
            else:
                last_error = f"SSH tunnel exited with status {process.returncode}"
        return None, last_error

    def drop_tensorboard_tunnel(self, ticket_id: str) -> None:
        with self.lock:
            tunnel = self.tensorboard_tunnels.pop(ticket_id, None)
        if tunnel:
            process = tunnel.get("process")
            if isinstance(process, subprocess.Popen):
                self._terminate_process(process)

    def _ticket_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    [sys.executable, str(TICKET_TOOL), "status", "--json"],
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=True,
                )
                snapshot = sanitize_ticket_snapshot(json.loads(completed.stdout))
                with self.lock:
                    self.ticket = {
                        "connected": True,
                        "error": None,
                        "received_at": utc_now(),
                        "snapshot": snapshot,
                    }
                self._prune_tensorboard_tunnels(snapshot)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
                with self.lock:
                    previous = self.ticket.get("snapshot")
                    self.ticket = {
                        "connected": False,
                        "error": f"ticket snapshot unavailable: {type(exc).__name__}",
                        "received_at": self.ticket.get("received_at"),
                        "snapshot": previous,
                    }
            self.stop_event.wait(1.5)

    def _terminate_ssh(self, process: subprocess.Popen[str]) -> None:
        self._terminate_process(process)

    def _remote_loop(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            process: subprocess.Popen[str] | None = None
            try:
                if PROFILE is None:
                    raise RuntimeError("dashboard profile is not configured")
                runtime = managed_runtime_paths(PROFILE)
                remote_argv = [REMOTE_PYTHON, "-u", "-"]
                remote_command = build_landlock_command(
                    PROFILE,
                    remote_argv,
                    workdir=PROFILE["remote"]["temp_root"],
                    environment={"CUDA_VISIBLE_DEVICES": ""},
                    device_ids=PROFILE["gpu"]["ids"],
                )
                argv = ssh_argv(PROFILE, batch=True)
                argv.extend(
                    [
                        f"{PROFILE['ssh']['user']}@{PROFILE['ssh']['host']}",
                        remote_command,
                    ]
                )
                process = subprocess.Popen(
                    argv,
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
                with self.lock:
                    self.ssh_process = process
                assert process.stdin is not None
                process.stdin.write(
                    REMOTE_COLLECTOR.replace(
                        "__REMOTE_GPU_SAMPLE_INTERVAL__",
                        format(SAMPLE_INTERVAL_SECONDS, ".6g"),
                    )
                )
                process.stdin.close()
                assert process.stdout is not None
                received_sample = False
                last_message = time.monotonic()
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                try:
                    while not self.stop_event.is_set():
                        ready = selector.select(timeout=1.0)
                        if not ready:
                            if process.poll() is not None:
                                break
                            if time.monotonic() - last_message > 8.0:
                                raise TimeoutError("remote collector sample timeout")
                            continue
                        line = process.stdout.readline()
                        if not line:
                            break
                        last_message = time.monotonic()
                        if len(line.encode("utf-8", "replace")) > MAX_LINE_BYTES:
                            raise ValueError("remote collector line too large")
                        message = sanitize_remote(json.loads(line))
                        now_mono = time.monotonic()
                        with self.lock:
                            if message["type"] == "hello":
                                self.remote["hello"] = message
                                self.remote["error"] = None
                            else:
                                self.remote.update(
                                    {
                                        "connected": True,
                                        "error": None,
                                        "sample": message,
                                        "received_at": utc_now(),
                                        "received_monotonic": now_mono,
                                    }
                                )
                                received_sample = True
                        if received_sample:
                            backoff = 1.0
                finally:
                    selector.close()
                exit_code = process.wait(timeout=2)
                if not self.stop_event.is_set():
                    raise ConnectionError(f"collector exited with status {exit_code}")
            except (OSError, SSHError, ManagedRunError, subprocess.SubprocessError, json.JSONDecodeError, ValueError, ConnectionError, TimeoutError) as exc:
                with self.lock:
                    self.remote["connected"] = False
                    self.remote["error"] = f"nvitop stream unavailable: {type(exc).__name__}"
            finally:
                if process is not None:
                    self._terminate_ssh(process)
                with self.lock:
                    if self.ssh_process is process:
                        self.ssh_process = None
            if not self.stop_event.is_set():
                self.stop_event.wait(backoff + secrets.randbelow(250) / 1000.0)
                backoff = min(backoff * 2.0, 30.0)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            ticket = copy.deepcopy(self.ticket)
            remote = copy.deepcopy(self.remote)
            tunnel_snapshot = {
                ticket_id: {
                    "local_port": item.get("local_port"),
                    "remote_port": item.get("remote_port"),
                    "connected": isinstance(item.get("process"), subprocess.Popen)
                    and item["process"].poll() is None,
                    "connected_at": item.get("connected_at"),
                }
                for ticket_id, item in self.tensorboard_tunnels.items()
            }
        snapshot = ticket.get("snapshot")
        for item in self._ticket_items(snapshot):
            tensorboard = item.get("tensorboard")
            if not isinstance(tensorboard, dict):
                continue
            tunnel = tunnel_snapshot.get(item.get("id"))
            if tensorboard.get("status") == "live":
                tensorboard["viewer_status"] = "connected" if tunnel and tunnel["connected"] else "ready"
        received_mono = remote.pop("received_monotonic", None)
        if received_mono is not None:
            remote["age_seconds"] = round(max(0.0, time.monotonic() - received_mono), 2)
            if remote["age_seconds"] > 5.0:
                remote["connected"] = False
                remote["error"] = "nvitop sample is stale"
        else:
            remote["age_seconds"] = None
        warnings = derive_warnings(ticket.get("snapshot"), remote.get("sample"))
        return {
            "generated_at": utc_now(),
            "profile": {
                "slug": PROFILE["slug"] if PROFILE else None,
                "name": PROFILE["name"] if PROFILE else None,
                "coordination_uid": PROFILE["trust"]["coordination_uid"] if PROFILE else None,
                "ssh_trust_uid": PROFILE["trust"]["server_uid"] if PROFILE else None,
            },
            "ticket": ticket,
            "remote": remote,
            "warnings": warnings,
        }


def derive_warnings(ticket: Any, sample: Any) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if not isinstance(ticket, dict):
        warnings.append({"level": "error", "message": "工单快照不可用；不能据此判断 GPU 是否空闲。"})
        return warnings
    if ticket.get("pending_reconcile"):
        warnings.append({"level": "warn", "message": "存在待 reconcile 的时间状态；看板不会自动修改账本。"})
    holders: dict[int, list[dict[str, Any]]] = {}
    for item in ticket.get("active", []):
        for gpu_id in item.get("assigned_gpus", []):
            holders.setdefault(gpu_id, []).append(item)
    for gpu_id, entries in holders.items():
        if len(entries) > 1:
            warnings.append({"level": "error", "message": f"GPU {gpu_id} 同时被 {len(entries)} 个持有态工单占用。"})
    if not isinstance(sample, dict):
        warnings.append({"level": "warn", "message": "远端 nvitop 快照不可用；保留工单占用状态，不把 GPU 显示为空闲。"})
        return warnings
    seen: set[int] = set()
    for gpu in sample.get("gpus", []):
        gpu_id = gpu.get("index")
        if not isinstance(gpu_id, int):
            continue
        if gpu_id in seen:
            warnings.append({"level": "error", "message": f"远端快照重复报告 GPU {gpu_id}。"})
        seen.add(gpu_id)
        processes = gpu.get("processes", [])
        if processes and gpu_id not in holders:
            warnings.append({"level": "warn", "message": f"GPU {gpu_id} 有 {len(processes)} 个进程，但账本没有持有态工单。"})
        if not processes and gpu_id in holders:
            warnings.append({"level": "info", "message": f"GPU {gpu_id} 有持有态工单但当前未采到进程；这不自动释放工单。"})
    missing = set(ticket.get("gpu_ids", [])) - seen
    if missing:
        warnings.append({"level": "error", "message": f"远端快照缺少账本 GPU：{sorted(missing)}。"})
    return warnings


def make_handler(
    state: DashboardState,
    instance_id: str,
    port: int,
    token: str,
    session_token: str,
) -> type[BaseHTTPRequestHandler]:
    expected_host = f"127.0.0.1:{port}"
    expected_origin = f"http://{expected_host}"
    viewer_host = f"localhost:{port}"
    viewer_origin = f"http://{viewer_host}"
    cookie_name = "remote_gpu_dev_dashboard"
    capability_root = f"/{token}/"
    capability_health = f"/{token}/health"
    static = {
        "/ui/": ("text/html; charset=utf-8", (ASSET_ROOT / "index.html").read_bytes()),
        "/ui/app.css": ("text/css; charset=utf-8", (ASSET_ROOT / "app.css").read_bytes()),
        "/ui/app.js": ("text/javascript; charset=utf-8", (ASSET_ROOT / "app.js").read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "remote-gpu-dev-dashboard"
        sys_version = ""

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _request_scope(self, path: str) -> str | None:
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if host == expected_host:
                allowed_origin = expected_origin
                scope = "dashboard"
            elif host == viewer_host and path.startswith("/tb/"):
                allowed_origin = viewer_origin
                scope = "tensorboard"
            elif host == viewer_host:
                self._error(404, "viewer origin serves only TensorBoard", tensorboard=True)
                return None
            else:
                self._error(421, "unexpected Host header")
                return None
            if origin is not None and origin != allowed_origin:
                self._error(403, "cross-origin request denied")
                return None
            return scope

        def _authenticated(self) -> bool:
            raw = self.headers.get("Cookie")
            if not raw:
                return False
            try:
                parsed = SimpleCookie(raw)
                morsel = parsed.get(cookie_name)
                return bool(morsel and hmac.compare_digest(morsel.value, session_token))
            except (KeyError, ValueError):
                return False

        def _headers(
            self,
            status: int,
            content_type: str,
            length: int | None,
            *,
            tensorboard: bool = False,
            extra: list[tuple[str, str]] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            if self.close_connection:
                self.send_header("Connection", "close")
            elif length is None:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            if tensorboard:
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self' blob: data:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                    "connect-src 'self'; worker-src 'self' blob:; "
                    f"frame-ancestors {expected_origin}; "
                    "base-uri 'self'; form-action 'none'",
                )
            else:
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                    f"img-src 'self' data:; frame-src {viewer_origin}; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'none'",
                )
            for name, value in extra or []:
                self.send_header(name, value)
            self.end_headers()

        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            head_only: bool = False,
            *,
            tensorboard: bool = False,
            extra: list[tuple[str, str]] | None = None,
        ) -> None:
            self._headers(
                status,
                content_type,
                len(body),
                tensorboard=tensorboard,
                extra=extra,
            )
            if not head_only:
                self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send(status, "application/json; charset=utf-8", body)

        def _error(self, status: int, message: str, *, tensorboard: bool = False) -> None:
            body = (message.strip() + "\n").encode("utf-8", "replace")
            self._send(status, "text/plain; charset=utf-8", body, tensorboard=tensorboard)

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            del explain
            self._error(code, message or self.responses.get(code, ("Error",))[0])

        def _bootstrap(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/ui/")
            self.send_header(
                "Set-Cookie",
                f"{cookie_name}={session_token}; Path=/; HttpOnly; SameSite=Strict",
            )
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
            self.end_headers()

        def _read_tensorboard_control_ticket(self, parsed: Any) -> str | None:
            if parsed.query:
                self._send_json(405, {"ok": False, "error": "控制 API 不接受查询参数。"})
                return None
            if not self._authenticated():
                self._send_json(401, {"ok": False, "error": "请先打开看板 capability URL。"})
                return None
            if self.headers.get_all("Host", []) != [expected_host] or self.headers.get_all(
                "Origin", []
            ) != [expected_origin]:
                self._send_json(403, {"ok": False, "error": "控制 API 仅接受同源请求。"})
                return None
            forbidden = (
                "Transfer-Encoding",
                "TE",
                "Authorization",
                "Proxy-Authorization",
                "Proxy-Connection",
                "Content-Encoding",
                "Expect",
                "Trailer",
                "Upgrade",
            )
            if any(self.headers.get_all(name, []) for name in forbidden):
                self._send_json(400, {"ok": False, "error": "控制请求包含禁止的请求头。"})
                return None
            if len(self.headers.get_all("Cookie", [])) != 1:
                self._send_json(400, {"ok": False, "error": "控制请求必须携带一个会话 Cookie。"})
                return None
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].strip().lower() != "application/json":
                self._send_json(415, {"ok": False, "error": "控制 API 仅接受 application/json。"})
                return None
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                self._send_json(411, {"ok": False, "error": "控制 API 需要 Content-Length。"})
                return None
            raw_length = lengths[0].strip()
            if not re.fullmatch(r"[0-9]{1,4}", raw_length):
                self._send_json(400, {"ok": False, "error": "无效的 Content-Length。"})
                return None
            content_length = int(raw_length)
            if content_length < 1:
                self._send_json(400, {"ok": False, "error": "控制请求体不能为空。"})
                return None
            if content_length > MAX_CONTROL_POST_BYTES:
                self._send_json(413, {"ok": False, "error": "控制请求体过大。"})
                return None

            previous_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(5)
                request_body = self.rfile.read(content_length)
            except (OSError, TimeoutError):
                self._send_json(408, {"ok": False, "error": "控制请求体读取超时。"})
                return None
            finally:
                try:
                    self.connection.settimeout(previous_timeout)
                except OSError:
                    pass
            if len(request_body) != content_length:
                self._send_json(400, {"ok": False, "error": "控制请求体不完整。"})
                return None

            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                value: dict[str, Any] = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("duplicate JSON key")
                    value[key] = item
                return value

            try:
                payload = json.loads(
                    request_body.decode("utf-8", "strict"),
                    object_pairs_hook=unique_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                self._send_json(400, {"ok": False, "error": "控制请求必须是规范 JSON 对象。"})
                return None
            if not isinstance(payload, dict) or set(payload) != {"ticket_id"}:
                self._send_json(400, {"ok": False, "error": "控制请求只能包含 ticket_id。"})
                return None
            ticket_id = payload.get("ticket_id")
            if not isinstance(ticket_id, str) or not TENSORBOARD_TICKET_RE.fullmatch(ticket_id):
                self._send_json(400, {"ok": False, "error": "无效的工单 ID。"})
                return None
            return ticket_id

        def _proxy_tensorboard(
            self,
            parsed: Any,
            method: str,
            request_body: bytes | None = None,
            request_content_type: str | None = None,
        ) -> None:
            path = parsed.path
            head_only = method == "HEAD"
            if len(self.path.encode("utf-8", "replace")) > MAX_PROXY_REQUEST_PATH_BYTES:
                self._error(414, "TensorBoard request path is too long", tensorboard=True)
                return
            pieces = path.split("/", 3)
            if len(pieces) < 4 or pieces[1] != "tb":
                self._error(404, "TensorBoard route not found", tensorboard=True)
                return
            ticket_segment = pieces[2]
            ticket_id = decode_tensorboard_ticket_segment(ticket_segment)
            if ticket_id is None:
                self._error(404, "invalid TensorBoard ticket", tensorboard=True)
                return
            upstream_port, error = state.tensorboard_upstream(ticket_id)
            if upstream_port is None:
                self._error(503, error or "TensorBoard is unavailable", tensorboard=True)
                return
            upstream_path = path
            if parsed.query:
                upstream_path += "?" + parsed.query
            request_headers = {
                "Host": f"127.0.0.1:{upstream_port}",
                "User-Agent": "remote-gpu-dev-dashboard-tensorboard-proxy/1",
            }
            for name in ("Accept", "Accept-Encoding", "Range", "If-None-Match", "If-Modified-Since"):
                value = self.headers.get(name)
                if value:
                    request_headers[name] = value[:4096]
            if method == "POST":
                if request_body is None or request_content_type is None:
                    self._error(500, "TensorBoard POST proxy state is invalid", tensorboard=True)
                    return
                request_headers["Content-Type"] = request_content_type
                request_headers["Content-Length"] = str(len(request_body))
            connection = http.client.HTTPConnection("127.0.0.1", upstream_port, timeout=10)
            response_started = False
            try:
                connection.request(
                    method,
                    upstream_path,
                    body=request_body if method == "POST" else None,
                    headers=request_headers,
                )
                response = connection.getresponse()
                response_headers: list[tuple[str, str]] = []
                content_type = response.getheader("Content-Type") or "application/octet-stream"
                content_length: int | None = None
                raw_length = response.getheader("Content-Length")
                if raw_length and raw_length.isdigit():
                    content_length = int(raw_length)
                for name, value in response.getheaders():
                    lower = name.lower()
                    if lower in HOP_BY_HOP_HEADERS or lower in {
                        "content-length",
                        "content-type",
                        "content-security-policy",
                        "x-frame-options",
                        "set-cookie",
                        "cache-control",
                        "referrer-policy",
                        "x-content-type-options",
                    }:
                        continue
                    if lower == "location":
                        location = urlsplit(value)
                        if location.scheme or location.netloc:
                            if location.hostname not in {"127.0.0.1", "localhost"}:
                                continue
                            value = location.path + (("?" + location.query) if location.query else "")
                        expected_prefix = tensorboard_path_prefix(ticket_id)
                        if not expected_prefix or not (
                            value == expected_prefix
                            or value.startswith(expected_prefix + "/")
                        ):
                            continue
                    if lower in {"content-encoding", "etag", "last-modified", "accept-ranges", "location", "vary"}:
                        response_headers.append((name, value[:8192]))
                self._headers(
                    response.status,
                    content_type,
                    content_length,
                    tensorboard=True,
                    extra=response_headers,
                )
                response_started = True
                if not head_only:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (OSError, http.client.HTTPException, TimeoutError):
                state.drop_tensorboard_tunnel(ticket_id)
                if not response_started and not self.wfile.closed:
                    try:
                        self._error(502, "TensorBoard upstream connection failed", tensorboard=True)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.close_connection = True
            finally:
                connection.close()

        def _proxy_tensorboard_readonly_post(self, parsed: Any) -> None:
            if len(self.path.encode("utf-8", "replace")) > MAX_PROXY_REQUEST_PATH_BYTES:
                self._error(414, "TensorBoard request path is too long", tensorboard=True)
                return
            allowed_target = (
                tensorboard_timeseries_post_target(parsed.path) is not None
                or tensorboard_scalars_multirun_post_ticket(parsed.path) is not None
            )
            if parsed.query or not allowed_target:
                self._error(405, "TensorBoard POST endpoint is not allowed", tensorboard=True)
                return
            if self.headers.get_all("Host", []) != [viewer_host] or self.headers.get_all(
                "Origin", []
            ) != [viewer_origin]:
                self._error(403, "TensorBoard POST requires the viewer origin", tensorboard=True)
                return

            forbidden = (
                "Transfer-Encoding",
                "TE",
                "Cookie",
                "Authorization",
                "Proxy-Authorization",
                "Proxy-Connection",
                "Content-Encoding",
                "Expect",
                "Trailer",
                "Upgrade",
            )
            if any(self.headers.get_all(name, []) for name in forbidden):
                self._error(400, "forbidden TensorBoard POST header", tensorboard=True)
                return

            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1:
                self._error(415, "TensorBoard POST requires one Content-Type", tensorboard=True)
                return
            content_type = content_types[0].strip()
            boundary = multipart_form_data_boundary(content_type)
            if boundary is None:
                self._error(415, "TensorBoard POST requires multipart/form-data", tensorboard=True)
                return

            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                self._error(411, "TensorBoard POST requires Content-Length", tensorboard=True)
                return
            raw_length = lengths[0].strip()
            if not re.fullmatch(r"[0-9]{1,10}", raw_length):
                self._error(400, "invalid TensorBoard POST Content-Length", tensorboard=True)
                return
            content_length = int(raw_length)
            if content_length < 1:
                self._error(400, "TensorBoard POST body must not be empty", tensorboard=True)
                return
            if content_length > MAX_TENSORBOARD_POST_BYTES:
                self._error(413, "TensorBoard POST body is too large", tensorboard=True)
                return

            previous_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(10)
                request_body = self.rfile.read(content_length)
            except (OSError, TimeoutError):
                self._error(408, "TensorBoard POST body could not be read", tensorboard=True)
                return
            finally:
                try:
                    self.connection.settimeout(previous_timeout)
                except OSError:
                    pass
            if len(request_body) != content_length:
                self._error(400, "incomplete TensorBoard POST body", tensorboard=True)
                return
            marker = b"--" + boundary.encode("ascii")
            if not request_body.startswith(marker + b"\r\n") or not request_body.rstrip(
                b"\r\n"
            ).endswith(marker + b"--"):
                self._error(400, "malformed TensorBoard multipart body", tensorboard=True)
                return
            self._proxy_tensorboard(
                parsed,
                "POST",
                request_body=request_body,
                request_content_type=content_type,
            )

        def _dispatch(self, head_only: bool = False) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            scope = self._request_scope(path)
            if scope is None:
                return
            if scope == "tensorboard":
                self._proxy_tensorboard(parsed, "HEAD" if head_only else "GET")
                return
            if path == capability_health:
                body = json.dumps({"ok": True, "instance_id": instance_id}).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body, head_only)
                return
            if path == capability_root:
                if head_only:
                    self._send(200, "text/plain; charset=utf-8", b"", True)
                else:
                    self._bootstrap()
                return
            if not self._authenticated():
                self._error(401, "open the dashboard capability URL first")
                return
            if path == "/api/status":
                body = json.dumps(state.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body, head_only)
                return
            asset = static.get(path)
            if asset is None:
                self._error(404, "not found")
                return
            content_type, body = asset
            self._send(200, content_type, body, head_only)

        def do_GET(self) -> None:
            self._dispatch(False)

        def do_HEAD(self) -> None:
            self._dispatch(True)

        def do_POST(self) -> None:
            self.close_connection = True
            parsed = urlsplit(self.path)
            scope = self._request_scope(parsed.path)
            if scope is None:
                return
            if scope == "tensorboard":
                self._proxy_tensorboard_readonly_post(parsed)
                return
            controls = {
                "/api/tensorboard/open": state.open_tensorboard,
                "/api/tensorboard/close": state.close_tensorboard,
            }
            operation = controls.get(parsed.path)
            if operation is None:
                self._error(405, "method not allowed")
                return
            ticket_id = self._read_tensorboard_control_ticket(parsed)
            if ticket_id is None:
                return
            status, payload = operation(ticket_id)
            self._send_json(status, payload)

        def _reject_mutating_method(self) -> None:
            self.close_connection = True
            self._error(405, "method not allowed")

        do_PUT = _reject_mutating_method
        do_DELETE = _reject_mutating_method
        do_PATCH = _reject_mutating_method

    return Handler


def open_control_lock() -> Any:
    path = runtime_dir() / "control.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(path, 0o600)
    handle = os.fdopen(descriptor, "r+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def ensure_dashboard(open_browser: bool) -> int:
    with open_control_lock():
        meta = load_metadata()
        if meta and is_live(meta):
            if meta.get("code_hash") != code_hash():
                print("dashboard is running older code; stop it before restarting", file=sys.stderr)
                return 1
            url = dashboard_url(meta)
            print(f"dashboard={url} pid={meta['pid']} status=running")
            if open_browser:
                webbrowser.open(url)
            return 0
        if meta and process_identity_matches(meta):
            print("dashboard process exists but health verification failed; refusing to replace it", file=sys.stderr)
            return 1
        try:
            metadata_path().unlink()
        except FileNotFoundError:
            pass
        log_path = runtime_dir() / "dashboard.log"
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.chmod(log_path, 0o600)
        try:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "_serve"],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(log_fd)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print(f"dashboard failed to start; inspect {log_path}", file=sys.stderr)
                return 1
            meta = load_metadata()
            if meta and is_live(meta):
                url = dashboard_url(meta)
                print(f"dashboard={url} pid={meta['pid']} status=started")
                if open_browser:
                    webbrowser.open(url)
                return 0
            time.sleep(0.1)
        print(f"dashboard start timed out; inspect {log_path}", file=sys.stderr)
        return 1


def dashboard_status() -> int:
    meta = load_metadata()
    if meta and is_live(meta):
        print(f"dashboard={dashboard_url(meta)} pid={meta['pid']} status=running")
        return 0
    print("dashboard status=stopped")
    return 1


def stop_dashboard() -> int:
    with open_control_lock():
        meta = load_metadata()
        if not meta:
            print("dashboard status=stopped")
            return 0
        if not process_identity_matches(meta) or not health_matches(meta):
            print("dashboard identity could not be verified; refusing to signal any process", file=sys.stderr)
            return 1
        pid = int(meta["pid"])
        os.kill(pid, signal.SIGTERM)
        # One in-flight 45 s control call may finish before the daemon performs
        # its own bounded 30 s generation-pinned cleanup.
        deadline = time.monotonic() + DASHBOARD_STOP_GRACE_SECONDS
        while time.monotonic() < deadline and process_start_ticks(pid) is not None:
            time.sleep(0.1)
        if process_start_ticks(pid) is not None:
            if not process_identity_matches(meta):
                print("dashboard process identity changed; refusing SIGKILL", file=sys.stderr)
                return 1
            os.kill(pid, signal.SIGKILL)
        current = load_metadata()
        if current and current.get("instance_id") == meta.get("instance_id"):
            try:
                metadata_path().unlink()
            except FileNotFoundError:
                pass
        print("dashboard status=stopped")
        return 0


def serve_dashboard() -> int:
    daemon_lock_path = runtime_dir() / "daemon.lock"
    daemon_fd = os.open(daemon_lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(daemon_lock_path, 0o600)
    daemon_lock = os.fdopen(daemon_fd, "r+")
    try:
        fcntl.flock(daemon_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 1

    requested_port = integer(os.environ.get("REMOTE_GPU_DEV_DASHBOARD_PORT")) or DEFAULT_PORT
    try:
        server = ThreadingHTTPServer(("127.0.0.1", requested_port), BaseHTTPRequestHandler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    server.daemon_threads = True
    server.timeout = 0.5
    port = int(server.server_address[1])
    token = secrets.token_urlsafe(24)
    session_token = secrets.token_urlsafe(32)
    instance_id = secrets.token_hex(16)
    state = DashboardState()
    server.RequestHandlerClass = make_handler(
        state,
        instance_id,
        port,
        token,
        session_token,
    )
    pid = os.getpid()
    start_ticks = process_start_ticks(pid)
    if start_ticks is None:
        return 1
    meta = {
        "pid": pid,
        "uid": os.getuid(),
        "boot_id": boot_id(),
        "starttime_ticks": start_ticks,
        "instance_id": instance_id,
        "port": port,
        "token": token,
        "code_hash": code_hash(),
        "started_at": utc_now(),
    }
    atomic_write_metadata(meta)
    stopping = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state.start()
    try:
        while not stopping.is_set():
            server.handle_request()
    finally:
        server.server_close()
        state.stop()
        current = load_metadata()
        if current and current.get("instance_id") == instance_id:
            try:
                metadata_path().unlink()
            except FileNotFoundError:
                pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure = subparsers.add_parser("ensure", help="start the dashboard or reuse the verified singleton")
    ensure.add_argument("--open", action="store_true", help="open the verified loopback URL in the default browser")
    subparsers.add_parser("status", help="show the verified dashboard status and URL")
    subparsers.add_parser("stop", help="stop only the exact verified dashboard process")
    subparsers.add_parser("_serve", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        configure_profile()
        if args.command == "ensure":
            return ensure_dashboard(args.open)
        if args.command == "status":
            return dashboard_status()
        if args.command == "stop":
            return stop_dashboard()
        return serve_dashboard()
    except (ProfileError, RuntimeError, OSError) as exc:
        print(f"remote-gpu-dashboard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
