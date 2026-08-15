#!/usr/bin/env python3
"""Atomic, file-backed GPU tickets for one remote-gpu-dev server profile."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import posixpath
import re
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from credential_guard import contains_secret
from profile import ProfileError, load_profile
from remote_path_guard import RemotePathError, require_managed_remote_path


SCRIPT_PATH = Path(__file__).resolve()
PROFILE: dict[str, Any] | None = None
TICKET_ROOT = Path("/nonexistent/remote-gpu-dev-ticket-root")
CONFIG_PATH = TICKET_ROOT / "config.json"
STATE_PATH = TICKET_ROOT / "state.json"
BOARD_PATH = TICKET_ROOT / "BOARD.md"
EVENTS_PATH = TICKET_ROOT / "events.jsonl"
LOCK_PATH = TICKET_ROOT / ".lock"
TICKET_DIR = TICKET_ROOT / "tickets"


def configure_profile_paths() -> None:
    global PROFILE, TICKET_ROOT, CONFIG_PATH, STATE_PATH, BOARD_PATH
    global EVENTS_PATH, LOCK_PATH, TICKET_DIR
    try:
        PROFILE = load_profile()
    except ProfileError as exc:
        raise TicketError(str(exc)) from exc
    TICKET_ROOT = Path(PROFILE["local"]["ticket_root"])
    CONFIG_PATH = TICKET_ROOT / "config.json"
    STATE_PATH = TICKET_ROOT / "state.json"
    BOARD_PATH = TICKET_ROOT / "BOARD.md"
    EVENTS_PATH = TICKET_ROOT / "events.jsonl"
    LOCK_PATH = TICKET_ROOT / ".lock"
    TICKET_DIR = TICKET_ROOT / "tickets"

HOLDING_STATES = {"reserved", "running", "stale"}
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}
ALL_STATES = {"queued", *HOLDING_STATES, *TERMINAL_STATES}
TENSORBOARD_STATES = {
    "starting",
    "live",
    "stopped",
    "failed",
    "cleanup_pending",
}
TENSORBOARD_PORT_HOLDING_STATES = {
    "starting",
    "live",
    "failed",
    "cleanup_pending",
}
# Schema-1 ledgers created before the ASCII-only slug rule may contain
# Unicode letters/digits and underscores.  Keep those IDs operable while
# excluding path separators, whitespace, controls, and punctuation that is
# unsafe in ticket filenames or URL routing.
TICKET_ID_RE = re.compile(r"GPU-[\w-]{1,156}\Z", flags=re.UNICODE)
TENSORBOARD_TRANSITIONS = {
    None: {"starting", "stopped"},
    "starting": {"live", "stopped", "failed", "cleanup_pending"},
    "live": {"live", "stopped", "failed", "cleanup_pending"},
    "cleanup_pending": {"cleanup_pending", "live", "stopped", "failed"},
    "failed": {"failed", "cleanup_pending", "stopped", "starting"},
    "stopped": {"stopped", "starting"},
}
DEFAULT_TENSORBOARD_PORT_START = 16006
DEFAULT_TENSORBOARD_PORT_END = 16105
TENSORBOARD_FIELDS = {
    "status",
    "logdir",
    "env_prefix",
    "remote_port",
    "path_prefix",
    "session",
    "pid",
    "process_start_ticks",
    "boot_id",
    "version",
    "command_sha256",
    "last_error",
    "generation",
    "registered_at",
    "live_at",
    "stopped_at",
    "updated_at",
}
class TicketError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def timestamp(value: dt.datetime | None = None) -> str:
    value = value or utc_now()
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_duration(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([mhd])", value.strip().lower())
    if not match:
        raise TicketError("duration must look like 30m, 2h, or 1d")
    amount = int(match.group(1))
    factor = {"m": 1, "h": 60, "d": 1440}[match.group(2)]
    minutes = amount * factor
    if minutes > 525_600:
        raise TicketError("duration must not exceed 365d")
    return minutes


def clean_text(value: str, field: str, limit: int) -> str:
    value = " ".join(value.strip().split())
    if not value:
        raise TicketError(f"{field} must not be empty")
    if len(value) > limit:
        raise TicketError(f"{field} must not exceed {limit} characters")
    if contains_secret(value):
        raise TicketError(f"{field} looks secret-bearing; store only a sanitized summary")
    if any(ord(char) < 32 for char in value):
        raise TicketError(f"{field} contains control characters")
    return value


def clean_tensorboard_text(value: str, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TicketError(f"{field} must be a string")
    value = clean_text(value, field, limit)
    if contains_secret(value):
        raise TicketError(f"{field} looks secret-bearing; store only sanitized metadata")
    return value


def clean_absolute_remote_path(value: str, field: str, limit: int = 512) -> str:
    if not isinstance(value, str):
        raise TicketError(f"{field} must be a string")
    if value != value.strip():
        raise TicketError(f"{field} must not have surrounding whitespace")
    if not value or len(value) > limit:
        raise TicketError(f"{field} must be between 1 and {limit} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise TicketError(f"{field} contains control characters")
    if contains_secret(value):
        raise TicketError(f"{field} looks secret-bearing; use a sanitized absolute path")
    if not value.startswith("/") or value.startswith("//"):
        raise TicketError(f"{field} must be an absolute remote POSIX path")
    normalized = posixpath.normpath(value)
    if normalized != value or value == "/":
        raise TicketError(
            f"{field} must be normalized, absolute, and more specific than /"
        )
    if any(part in {".", ".."} for part in value.split("/")):
        raise TicketError(f"{field} must not contain . or .. components")
    return value


def clean_managed_remote_path(value: str, field: str, limit: int = 512) -> str:
    """Validate a ledger path against the selected profile's two roots."""

    cleaned = clean_absolute_remote_path(value, field, limit)
    if PROFILE is None:
        raise TicketError("no selected server profile is configured")
    try:
        return require_managed_remote_path(PROFILE, cleaned, field)
    except RemotePathError as exc:
        raise TicketError(str(exc)) from exc


def clean_positive_integer(
    value: Any, field: str, maximum: int = (2**63 - 1)
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TicketError(f"{field} must be an integer")
    if value < 1 or value > maximum:
        raise TicketError(f"{field} must be between 1 and {maximum}")
    return value


def clean_remote_port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TicketError("remote_port must be an integer")
    if value < 1024 or value > 65535:
        raise TicketError("remote_port must be between 1024 and 65535")
    return value


def clean_boot_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
        value,
    ):
        raise TicketError("boot_id must be a canonical UUID")
    return value.lower()


def clean_command_sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", value):
        raise TicketError("command_sha256 must be exactly 64 hexadecimal characters")
    return value.lower()


def clean_ticket_id(value: Any) -> str:
    if not isinstance(value, str) or not TICKET_ID_RE.fullmatch(value):
        raise TicketError("ticket ID has an invalid format")
    return value


def tensorboard_path_prefix(ticket_id: str) -> str:
    # A ticket ID is an opaque ledger key.  Encode it as exactly one URL path
    # segment so legacy Unicode IDs cannot be confused with route syntax.
    return f"/tb/{quote(clean_ticket_id(ticket_id), safe='')}"


def clean_path_prefix(value: str, ticket_id: str) -> str:
    expected = tensorboard_path_prefix(ticket_id)
    if value != expected:
        raise TicketError(f"path_prefix must exactly equal {expected}")
    return value


def tensorboard_port_bounds(config: dict[str, Any]) -> tuple[int, int]:
    start = config.get("tensorboard_port_start", DEFAULT_TENSORBOARD_PORT_START)
    end = config.get("tensorboard_port_end", DEFAULT_TENSORBOARD_PORT_END)
    start = clean_remote_port(start)
    end = clean_remote_port(end)
    if start > end:
        raise TicketError("tensorboard_port_start must not exceed tensorboard_port_end")
    if end - start + 1 > 10_000:
        raise TicketError("TensorBoard port pool must not exceed 10000 ports")
    return start, end


def clean_tensorboard_timestamp(value: Any, field: str, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise TicketError(f"TensorBoard {field} is required")
        return None
    if not isinstance(value, str) or len(value) > 40:
        raise TicketError(f"TensorBoard {field} must be a short ISO-8601 timestamp")
    try:
        parsed = parse_timestamp(value)
    except (TypeError, ValueError) as exc:
        raise TicketError(f"TensorBoard {field} is not a valid ISO-8601 timestamp") from exc
    if parsed is None or parsed.tzinfo is None:
        raise TicketError(f"TensorBoard {field} must include a timezone")
    return value


def validate_tensorboard_record(ticket_id: str, value: Any) -> dict[str, Any] | None:
    """Validate the optional schema-1 TensorBoard extension without upgrading state."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TicketError(f"TensorBoard metadata for {ticket_id} must be an object or null")
    unknown = set(value) - TENSORBOARD_FIELDS
    if unknown:
        raise TicketError(
            f"unknown TensorBoard fields for {ticket_id}: {', '.join(sorted(unknown))}"
        )
    status = value.get("status")
    if status not in TENSORBOARD_STATES:
        raise TicketError(f"unknown TensorBoard state for {ticket_id}: {status}")

    required_base = ("logdir", "env_prefix", "path_prefix", "session")
    for field in required_base:
        if value.get(field) is None:
            raise TicketError(f"TensorBoard {field} is required for {ticket_id}")
    clean_managed_remote_path(value["logdir"], "logdir")
    clean_managed_remote_path(value["env_prefix"], "env_prefix")
    remote_port = value.get("remote_port")
    if remote_port is None:
        if status != "stopped":
            raise TicketError(
                f"TensorBoard remote_port is required for {ticket_id} in state {status}"
            )
    else:
        clean_remote_port(remote_port)
    clean_path_prefix(value["path_prefix"], ticket_id)
    if clean_tensorboard_text(value["session"], "session", 128) != value["session"]:
        raise TicketError(f"TensorBoard session for {ticket_id} is not normalized")

    identity_fields = ("pid", "process_start_ticks", "boot_id")
    identity_present = [value.get(field) is not None for field in identity_fields]
    if any(identity_present) and not all(identity_present):
        raise TicketError(
            f"TensorBoard process identity for {ticket_id} must include pid, "
            "process_start_ticks, and boot_id together"
        )
    if all(identity_present):
        clean_positive_integer(value["pid"], "pid", 2**31 - 1)
        clean_positive_integer(value["process_start_ticks"], "process_start_ticks")
        if clean_boot_id(value["boot_id"]) != value["boot_id"]:
            raise TicketError(f"TensorBoard boot_id for {ticket_id} must be lowercase")

    version = value.get("version")
    command_sha256 = value.get("command_sha256")
    if version is not None:
        if clean_tensorboard_text(version, "version", 128) != version:
            raise TicketError(f"TensorBoard version for {ticket_id} is not normalized")
    if command_sha256 is not None:
        if clean_command_sha256(command_sha256) != command_sha256:
            raise TicketError(
                f"TensorBoard command_sha256 for {ticket_id} must be lowercase"
            )
    if status == "live":
        missing = [
            field
            for field in (*identity_fields, "version", "command_sha256")
            if value.get(field) is None
        ]
        if missing:
            raise TicketError(
                f"live TensorBoard metadata for {ticket_id} is missing "
                + ", ".join(missing)
            )

    last_error = value.get("last_error")
    if last_error is not None:
        if clean_tensorboard_text(last_error, "last_error", 500) != last_error:
            raise TicketError(f"TensorBoard last_error for {ticket_id} is not normalized")
    if status in {"failed", "cleanup_pending"} and not last_error:
        raise TicketError(f"TensorBoard {status} requires last_error for {ticket_id}")

    clean_positive_integer(value.get("generation"), "generation", 1_000_000)
    clean_tensorboard_timestamp(value.get("registered_at"), "registered_at", True)
    clean_tensorboard_timestamp(value.get("updated_at"), "updated_at", True)
    clean_tensorboard_timestamp(value.get("live_at"), "live_at")
    clean_tensorboard_timestamp(value.get("stopped_at"), "stopped_at")
    if status == "live" and value.get("live_at") is None:
        raise TicketError(f"live TensorBoard metadata for {ticket_id} requires live_at")
    if status in {"stopped", "failed"} and value.get("stopped_at") is None:
        raise TicketError(f"TensorBoard {status} metadata for {ticket_id} requires stopped_at")
    return value


def slugify(value: str) -> str:
    # Ticket IDs are also URL path and tmux-session identifiers.  Keep newly
    # generated slugs in one portable alphabet shared by every consumer.
    value = re.sub(r"[^a-z0-9]+", "-", value.lower(), flags=re.ASCII).strip("-")
    return (value or "task")[:28]


def parse_gpu_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise TicketError("GPU IDs must be comma-separated integers") from exc
    if not ids or len(set(ids)) != len(ids):
        raise TicketError("GPU IDs must be a non-empty unique list")
    return ids


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise TicketError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TicketError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TicketError(f"expected a JSON object in {path}")
    return value


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    if config.get("schema_version") != 1:
        raise TicketError("unsupported config schema version")
    gpu_ids = config.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(item, int) or item < 0 for item in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise TicketError("config gpu_ids must be unique non-negative integers")
    tensorboard_port_bounds(config)
    if PROFILE is None:
        raise TicketError("no selected server profile is configured")
    expected = {
        "coordination_uid": PROFILE["trust"]["coordination_uid"],
        "gpu_ids": PROFILE["gpu"]["ids"],
        "gpu_devices": PROFILE["gpu"]["devices"],
        "reservation_ttl_minutes": PROFILE["gpu"]["reservation_ttl_minutes"],
        "heartbeat_grace_minutes": PROFILE["gpu"]["heartbeat_grace_minutes"],
        "recent_terminal_limit": 12,
        "tensorboard_port_start": PROFILE["dashboard"]["tensorboard_remote_port_start"],
        "tensorboard_port_end": PROFILE["dashboard"]["tensorboard_remote_port_end"],
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise TicketError(
                f"ticket config {field} differs from the selected profile's "
                "coordination contract"
            )
    # ``profile`` and ``server`` identify the alias that created the ledger.
    # They are display metadata: another verified alias may operate the same
    # ledger when the coordination identity and exact GPU mapping match.
    return config


def initial_state() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": timestamp(), "tickets": {}}


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return initial_state()
    state = load_json(STATE_PATH)
    if state.get("schema_version") != 1 or not isinstance(state.get("tickets"), dict):
        raise TicketError("unsupported or malformed state.json")
    for ticket_id, ticket in state["tickets"].items():
        clean_ticket_id(ticket_id)
        if not isinstance(ticket, dict) or ticket.get("id") != ticket_id:
            raise TicketError(f"malformed ticket record: {ticket_id}")
        if ticket.get("status") not in ALL_STATES:
            raise TicketError(f"unknown ticket state for {ticket_id}")
        # ``tensorboard`` is an optional schema-1 extension.  Materialize null
        # in memory so old ledgers remain readable and status stays zero-write.
        ticket.setdefault("tensorboard", None)
        remote_workdir = ticket.get("remote_workdir")
        if remote_workdir is not None:
            clean_managed_remote_path(remote_workdir, "remote_workdir")
        validate_tensorboard_record(ticket_id, ticket["tensorboard"])
    return state


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_event(action: str, ticket: dict[str, Any], detail: str = "") -> None:
    event = {
        "at": timestamp(),
        "action": action,
        "ticket_id": ticket["id"],
        "status": ticket["status"],
        "assigned_gpus": ticket.get("assigned_gpus", []),
        "detail": detail,
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, list):
        value = ",".join(str(item) for item in value) or "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def ticket_markdown(ticket: dict[str, Any]) -> str:
    assigned = ", ".join(str(item) for item in ticket.get("assigned_gpus", [])) or "none"
    requested = ticket.get("requested_gpu_ids")
    requested_text = (
        ", ".join(str(item) for item in requested)
        if requested
        else f"any {ticket['requested_gpus']}"
    )
    tensorboard = ticket.get("tensorboard") or {}
    tensorboard_process = "-"
    if tensorboard.get("pid") is not None:
        tensorboard_process = (
            f"PID {tensorboard['pid']} @ ticks "
            f"{tensorboard.get('process_start_ticks', '-')}"
        )
    return f"""---
id: {json.dumps(ticket['id'], ensure_ascii=False)}
status: {json.dumps(ticket['status'])}
project: {json.dumps(ticket['project'], ensure_ascii=False)}
owner: {json.dumps(ticket['owner'], ensure_ascii=False)}
assigned_gpus: {json.dumps(ticket.get('assigned_gpus', []))}
created_at: {json.dumps(ticket['created_at'])}
updated_at: {json.dumps(ticket['updated_at'])}
---

# {ticket['id']}: {ticket['project']}

| Field | Value |
|---|---|
| Status | `{ticket['status']}` |
| Owner | {markdown_cell(ticket['owner'])} |
| Purpose | {markdown_cell(ticket['purpose'])} |
| Requested GPUs | {markdown_cell(requested_text)} |
| Assigned physical GPUs | {markdown_cell(assigned)} |
| Expected duration | {ticket['expected_duration_minutes']} minutes |
| Reservation expires | {markdown_cell(ticket.get('reservation_expires_at'))} |
| Heartbeat due | {markdown_cell(ticket.get('heartbeat_due_at'))} |
| Remote workdir | {markdown_cell(ticket.get('remote_workdir'))} |
| Session | {markdown_cell(ticket.get('session'))} |
| Command summary | {markdown_cell(ticket.get('command_summary'))} |
| Result | {markdown_cell(ticket.get('result'))} |
| TensorBoard status | `{markdown_cell(tensorboard.get('status'))}` |
| TensorBoard logdir | {markdown_cell(tensorboard.get('logdir'))} |
| TensorBoard environment | {markdown_cell(tensorboard.get('env_prefix'))} |
| TensorBoard remote port | {markdown_cell(tensorboard.get('remote_port'))} |
| TensorBoard path prefix | {markdown_cell(tensorboard.get('path_prefix'))} |
| TensorBoard session | {markdown_cell(tensorboard.get('session'))} |
| TensorBoard process | {markdown_cell(tensorboard_process)} |
| TensorBoard remote boot ID | {markdown_cell(tensorboard.get('boot_id'))} |
| TensorBoard version | {markdown_cell(tensorboard.get('version'))} |
| TensorBoard command SHA256 | {markdown_cell(tensorboard.get('command_sha256'))} |
| TensorBoard generation | {markdown_cell(tensorboard.get('generation'))} |
| TensorBoard became live | {markdown_cell(tensorboard.get('live_at'))} |
| TensorBoard stopped | {markdown_cell(tensorboard.get('stopped_at'))} |
| TensorBoard last error | {markdown_cell(tensorboard.get('last_error'))} |
| TensorBoard updated | {markdown_cell(tensorboard.get('updated_at'))} |

This file is generated from `state.json`. Use `gpu_ticket.py` for updates.
"""


def board_markdown(state: dict[str, Any], config: dict[str, Any]) -> str:
    tickets = list(state["tickets"].values())
    active = sorted(
        (ticket for ticket in tickets if ticket["status"] in HOLDING_STATES),
        key=lambda ticket: ticket["created_at"],
    )
    queued = sorted(
        (ticket for ticket in tickets if ticket["status"] == "queued"),
        key=lambda ticket: ticket["created_at"],
    )
    terminal = sorted(
        (ticket for ticket in tickets if ticket["status"] in TERMINAL_STATES),
        key=lambda ticket: ticket["updated_at"],
        reverse=True,
    )[: int(config.get("recent_terminal_limit", 12))]

    def table(items: list[dict[str, Any]]) -> str:
        if not items:
            return "_None._\n"
        lines = [
            "| Ticket | Status | Project | Owner | GPUs | TensorBoard | Updated |",
            "|---|---|---|---|---:|---|---|",
        ]
        for ticket in items:
            link = f"[#{ticket['id']}](tickets/{ticket['id']}.md)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        link,
                        markdown_cell(ticket["status"]),
                        markdown_cell(ticket["project"]),
                        markdown_cell(ticket["owner"]),
                        markdown_cell(ticket.get("assigned_gpus") or "-"),
                        markdown_cell((ticket.get("tensorboard") or {}).get("status")),
                        markdown_cell(ticket["updated_at"]),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    return (
        "# Remote GPU Ticket Board\n\n"
        f"Generated at `{state['updated_at']}` for `{config['server']}`. "
        "Use `gpu_ticket.py`; do not edit this board.\n\n"
        "## Active allocations\n\n"
        + table(active)
        + "\n## Queue\n\n"
        + table(queued)
        + "\n## Recent terminal tickets\n\n"
        + table(terminal)
    )


def persist(state: dict[str, Any], config: dict[str, Any]) -> None:
    state["updated_at"] = timestamp()
    atomic_write(
        STATE_PATH,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    for ticket in state["tickets"].values():
        atomic_write(TICKET_DIR / f"{ticket['id']}.md", ticket_markdown(ticket))
    atomic_write(BOARD_PATH, board_markdown(state, config))


def occupied_gpus(state: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for ticket in state["tickets"].values():
        if ticket["status"] in HOLDING_STATES:
            result.update(ticket.get("assigned_gpus", []))
    return result


def choose_gpus(
    state: dict[str, Any], config: dict[str, Any], count: int, requested: list[int] | None
) -> list[int] | None:
    configured = list(config["gpu_ids"])
    if requested and any(item not in configured for item in requested):
        raise TicketError(f"requested GPU is outside configured set {configured}")
    free = [item for item in configured if item not in occupied_gpus(state)]
    if requested:
        return requested if all(item in free for item in requested) else None
    return free[:count] if len(free) >= count else None


def reconcile_state(state: dict[str, Any], config: dict[str, Any]) -> list[str]:
    current = utc_now()
    changes: list[str] = []
    for ticket in state["tickets"].values():
        if ticket["status"] == "reserved":
            expiry = parse_timestamp(ticket.get("reservation_expires_at"))
            if expiry and expiry <= current:
                ticket["status"] = "expired"
                ticket["assigned_gpus"] = []
                ticket["updated_at"] = timestamp(current)
                ticket["result"] = "reservation launch window expired"
                changes.append(f"expired {ticket['id']}")
                append_event("auto-expire", ticket)
        elif ticket["status"] == "running":
            heartbeat_due = parse_timestamp(ticket.get("heartbeat_due_at"))
            if heartbeat_due and heartbeat_due <= current:
                ticket["status"] = "stale"
                ticket["updated_at"] = timestamp(current)
                changes.append(f"stale {ticket['id']}")
                append_event("auto-stale", ticket, "heartbeat overdue; GPUs remain held")

    queued = sorted(
        (ticket for ticket in state["tickets"].values() if ticket["status"] == "queued"),
        key=lambda ticket: ticket["created_at"],
    )
    for ticket in queued:
        assigned = choose_gpus(
            state,
            config,
            int(ticket["requested_gpus"]),
            ticket.get("requested_gpu_ids"),
        )
        if assigned is None:
            continue
        ticket["status"] = "reserved"
        ticket["assigned_gpus"] = assigned
        ticket["reservation_expires_at"] = timestamp(
            current + dt.timedelta(minutes=int(config["reservation_ttl_minutes"]))
        )
        ticket["updated_at"] = timestamp(current)
        changes.append(f"reserved {ticket['id']} on {assigned}")
        append_event("auto-reserve", ticket, "promoted from queue")
    return changes


def pending_time_transitions(state: dict[str, Any]) -> list[str]:
    """Describe time-based transitions without mutating files or state."""
    current = utc_now()
    pending: list[str] = []
    for ticket in state["tickets"].values():
        if ticket["status"] == "reserved":
            expiry = parse_timestamp(ticket.get("reservation_expires_at"))
            if expiry and expiry <= current:
                pending.append(f"would expire {ticket['id']}")
        elif ticket["status"] == "running":
            heartbeat_due = parse_timestamp(ticket.get("heartbeat_due_at"))
            if heartbeat_due and heartbeat_due <= current:
                pending.append(f"would mark stale {ticket['id']}")
    return pending


def require_ticket(state: dict[str, Any], ticket_id: str) -> dict[str, Any]:
    ticket_id = clean_ticket_id(ticket_id)
    try:
        return state["tickets"][ticket_id]
    except KeyError as exc:
        raise TicketError(f"unknown ticket: {ticket_id}") from exc


def result_payload(ticket: dict[str, Any], board: bool = True) -> dict[str, Any]:
    payload = dict(ticket)
    if board:
        payload["board"] = str(BOARD_PATH)
        payload["ticket_file"] = str(TICKET_DIR / f"{ticket['id']}.md")
    return payload


def occupied_tensorboard_ports(
    state: dict[str, Any], exclude_ticket_id: str | None = None
) -> set[int]:
    ports: set[int] = set()
    for ticket_id, ticket in state["tickets"].items():
        if ticket_id == exclude_ticket_id:
            continue
        tensorboard = ticket.get("tensorboard")
        if not tensorboard or tensorboard.get("status") not in TENSORBOARD_PORT_HOLDING_STATES:
            continue
        port = tensorboard.get("remote_port")
        if isinstance(port, int) and not isinstance(port, bool):
            ports.add(port)
    return ports


def choose_tensorboard_port(
    state: dict[str, Any], config: dict[str, Any], ticket_id: str
) -> int:
    occupied = occupied_tensorboard_ports(state, exclude_ticket_id=ticket_id)
    start, end = tensorboard_port_bounds(config)
    for port in range(start, end + 1):
        if port not in occupied:
            return port
    raise TicketError(f"no free TensorBoard port in configured pool {start}..{end}")


def tensorboard_argument_updates(args: argparse.Namespace) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if args.logdir is not None:
        updates["logdir"] = clean_managed_remote_path(args.logdir, "logdir")
    if args.env_prefix is not None:
        updates["env_prefix"] = clean_managed_remote_path(
            args.env_prefix, "env_prefix"
        )
    if args.remote_port is not None:
        updates["remote_port"] = clean_remote_port(args.remote_port)
    if args.path_prefix is not None:
        updates["path_prefix"] = clean_path_prefix(args.path_prefix, args.ticket_id)
    if args.session is not None:
        updates["session"] = clean_tensorboard_text(args.session, "session", 128)
    if args.pid is not None:
        updates["pid"] = clean_positive_integer(args.pid, "pid", 2**31 - 1)
    if args.process_start_ticks is not None:
        updates["process_start_ticks"] = clean_positive_integer(
            args.process_start_ticks, "process_start_ticks"
        )
    if args.boot_id is not None:
        updates["boot_id"] = clean_boot_id(args.boot_id)
    if args.version is not None:
        updates["version"] = clean_tensorboard_text(args.version, "version", 128)
    if args.command_sha256 is not None:
        updates["command_sha256"] = clean_command_sha256(args.command_sha256)
    if args.last_error is not None:
        updates["last_error"] = clean_tensorboard_text(
            args.last_error, "last_error", 500
        )
    return updates


def command_tensorboard(
    state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace
) -> None:
    """Atomically register or advance a ticket's independent TensorBoard sidecar."""
    ticket = require_ticket(state, args.ticket_id)
    existing = ticket.get("tensorboard")
    previous_status = existing.get("status") if existing else None
    desired_status = args.status
    expected_generation = args.expected_generation
    if expected_generation is not None:
        actual_generation = (existing or {}).get("generation")
        if expected_generation == 0:
            generation_matches = existing is None
        else:
            expected_generation = clean_positive_integer(
                expected_generation, "expected_generation", 1_000_000
            )
            generation_matches = actual_generation == expected_generation
        if not generation_matches:
            raise TicketError(
                "TensorBoard generation changed: expected "
                f"{'unconfigured' if expected_generation == 0 else expected_generation}, "
                f"got {actual_generation or 'null'}"
            )
    if desired_status not in TENSORBOARD_TRANSITIONS.get(previous_status, set()):
        raise TicketError(
            f"invalid TensorBoard transition {previous_status or 'null'} -> "
            f"{desired_status}"
        )
    current = utc_now()
    now = timestamp(current)
    updates = tensorboard_argument_updates(args)
    restarting = desired_status == "starting" and previous_status in {"stopped", "failed"}
    reconfiguring = (
        desired_status == "stopped"
        and previous_status == "stopped"
        and any(
            field in updates
            for field in ("logdir", "env_prefix", "path_prefix", "session")
        )
    )
    creating = existing is None or restarting or reconfiguring

    if creating:
        required_arguments = ("logdir", "env_prefix", "path_prefix", "session")
        missing = [field for field in required_arguments if field not in updates]
        if missing:
            raise TicketError(
                "creating a TensorBoard source or generation requires explicit "
                + ", ".join(missing)
            )
        remote_port = updates.get("remote_port")
        if remote_port is None and desired_status != "stopped":
            remote_port = choose_tensorboard_port(state, config, ticket["id"])
        metadata: dict[str, Any] = {
            "status": desired_status,
            "logdir": updates["logdir"],
            "env_prefix": updates["env_prefix"],
            "remote_port": remote_port,
            "path_prefix": updates["path_prefix"],
            "session": updates["session"],
            "pid": updates.get("pid"),
            "process_start_ticks": updates.get("process_start_ticks"),
            "boot_id": updates.get("boot_id"),
            "version": updates.get("version"),
            "command_sha256": updates.get("command_sha256"),
            "last_error": None,
            "generation": int((existing or {}).get("generation", 0)) + 1,
            "registered_at": now,
            "live_at": None,
            "stopped_at": None,
            "updated_at": now,
        }
    else:
        metadata = dict(existing)
        metadata.update(updates)
        metadata["status"] = desired_status
        metadata["updated_at"] = now

    if desired_status == "starting":
        metadata["last_error"] = None
        metadata["live_at"] = None
        metadata["stopped_at"] = None
    elif desired_status == "live":
        metadata["last_error"] = None
        metadata["live_at"] = metadata.get("live_at") or now
        metadata["stopped_at"] = None
    elif desired_status == "stopped":
        metadata["last_error"] = None
        metadata["stopped_at"] = now
    elif desired_status == "failed":
        metadata["stopped_at"] = now

    # Once a process has gone live, its launch identity is immutable.  A new
    # identity must be represented by stopped/failed -> starting generation.
    if existing and previous_status in {"live", "cleanup_pending"}:
        immutable = {
            "logdir",
            "env_prefix",
            "remote_port",
            "path_prefix",
            "session",
            "pid",
            "process_start_ticks",
            "boot_id",
            "version",
            "command_sha256",
        }
        changed = [field for field in immutable if metadata.get(field) != existing.get(field)]
        if changed:
            raise TicketError(
                "live TensorBoard identity is immutable; changed "
                + ", ".join(sorted(changed))
            )
    elif existing and previous_status == "failed" and not restarting:
        base_identity = {"logdir", "env_prefix", "remote_port", "path_prefix", "session"}
        changed = [
            field for field in base_identity if metadata.get(field) != existing.get(field)
        ]
        refinements = {
            "pid",
            "process_start_ticks",
            "boot_id",
            "version",
            "command_sha256",
        }
        changed.extend(
            field
            for field in refinements
            if existing.get(field) is not None
            and metadata.get(field) != existing.get(field)
        )
        if changed:
            raise TicketError(
                "failed TensorBoard identity may only fill missing process metadata; "
                "changed " + ", ".join(sorted(changed))
            )

    remote_port = metadata.get("remote_port")
    if remote_port is not None:
        remote_port = clean_remote_port(remote_port)
    occupied = occupied_tensorboard_ports(state, exclude_ticket_id=ticket["id"])
    if (
        desired_status in TENSORBOARD_PORT_HOLDING_STATES
        and remote_port in occupied
    ):
        raise TicketError(f"TensorBoard remote port {remote_port} is already registered")

    validate_tensorboard_record(ticket["id"], metadata)
    ticket["tensorboard"] = metadata
    ticket["updated_at"] = now
    append_event(
        "tensorboard",
        ticket,
        f"{previous_status or 'null'} -> {desired_status}; generation "
        f"{metadata['generation']}",
    )
    persist(state, config)
    print(json.dumps(result_payload(ticket), ensure_ascii=False, indent=2, sort_keys=True))


def command_init(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    reconcile_state(state, config)
    persist(state, config)
    print(f"initialized {TICKET_ROOT}")


def command_reserve(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    reconcile_state(state, config)
    project = clean_text(args.project, "project", 80)
    owner = clean_text(args.owner, "owner", 100)
    purpose = clean_text(args.purpose, "purpose", 240)
    requested = parse_gpu_ids(args.gpu_ids)
    count = args.gpus
    if requested:
        if args.gpus is not None and args.gpus != len(requested):
            raise TicketError("--gpus must match the number of --gpu-ids")
        count = len(requested)
    if count is None:
        count = 1
    if count < 1 or count > len(config["gpu_ids"]):
        raise TicketError(f"--gpus must be between 1 and {len(config['gpu_ids'])}")
    expected_minutes = parse_duration(args.expected)
    current = utc_now()
    ticket_id = (
        f"GPU-{current.strftime('%Y%m%d-%H%M%S')}-"
        f"{secrets.token_hex(2)}-{slugify(project)}"
    )
    assigned = choose_gpus(state, config, count, requested)
    status = "reserved" if assigned is not None else "queued"
    ticket = {
        "id": ticket_id,
        "status": status,
        "project": project,
        "owner": owner,
        "purpose": purpose,
        "requested_gpus": count,
        "requested_gpu_ids": requested,
        "assigned_gpus": assigned or [],
        "expected_duration_minutes": expected_minutes,
        "created_at": timestamp(current),
        "updated_at": timestamp(current),
        "reservation_expires_at": (
            timestamp(current + dt.timedelta(minutes=int(config["reservation_ttl_minutes"])))
            if assigned is not None
            else None
        ),
        "started_at": None,
        "last_heartbeat_at": None,
        "heartbeat_due_at": None,
        "expected_end_at": None,
        "finished_at": None,
        "session": None,
        "remote_workdir": None,
        "command_summary": None,
        "result": None,
        "tensorboard": None,
    }
    state["tickets"][ticket_id] = ticket
    append_event("reserve" if status == "reserved" else "queue", ticket)
    persist(state, config)
    payload = result_payload(ticket)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"{ticket_id} status={status} assigned={ticket['assigned_gpus']} "
            f"ticket={payload['ticket_file']}"
        )


def command_start(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    reconcile_state(state, config)
    ticket = require_ticket(state, args.ticket_id)
    if ticket["status"] != "reserved":
        raise TicketError(f"ticket must be reserved, got {ticket['status']}")
    confirmed = parse_gpu_ids(args.confirmed_idle)
    if confirmed != ticket["assigned_gpus"]:
        raise TicketError(
            f"--confirmed-idle must exactly match assigned GPUs {ticket['assigned_gpus']}"
        )
    session = clean_text(args.session, "session", 100)
    remote_workdir = clean_managed_remote_path(
        args.remote_workdir, "remote_workdir", 512
    )
    summary = clean_text(args.summary, "summary", 240)
    current = utc_now()
    expected_minutes = (
        parse_duration(args.expected)
        if args.expected
        else int(ticket["expected_duration_minutes"])
    )
    ticket.update(
        {
            "status": "running",
            "started_at": ticket.get("started_at") or timestamp(current),
            "updated_at": timestamp(current),
            "last_heartbeat_at": timestamp(current),
            "heartbeat_due_at": timestamp(
                current + dt.timedelta(minutes=int(config["heartbeat_grace_minutes"]))
            ),
            "expected_end_at": timestamp(current + dt.timedelta(minutes=expected_minutes)),
            "session": session,
            "remote_workdir": remote_workdir,
            "command_summary": summary,
            "reservation_expires_at": None,
        }
    )
    append_event("start", ticket, "remote GPU state confirmed by caller")
    persist(state, config)
    print(json.dumps(result_payload(ticket), ensure_ascii=False, indent=2, sort_keys=True))


def command_heartbeat(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    reconcile_state(state, config)
    ticket = require_ticket(state, args.ticket_id)
    if ticket["status"] not in {"running", "stale"}:
        raise TicketError(f"ticket must be running or stale, got {ticket['status']}")
    current = utc_now()
    ticket["status"] = "running"
    ticket["updated_at"] = timestamp(current)
    ticket["last_heartbeat_at"] = timestamp(current)
    ticket["heartbeat_due_at"] = timestamp(
        current + dt.timedelta(minutes=int(config["heartbeat_grace_minutes"]))
    )
    if args.expected:
        ticket["expected_end_at"] = timestamp(
            current + dt.timedelta(minutes=parse_duration(args.expected))
        )
    append_event("heartbeat", ticket)
    persist(state, config)
    print(
        f"{ticket['id']} status=running heartbeat_due={ticket['heartbeat_due_at']} "
        f"assigned={ticket['assigned_gpus']}"
    )


def command_release(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    reconcile_state(state, config)
    ticket = require_ticket(state, args.ticket_id)
    if ticket["status"] in TERMINAL_STATES:
        raise TicketError(f"ticket is already terminal: {ticket['status']}")
    if ticket["status"] == "queued" and args.outcome != "cancelled":
        raise TicketError("a queued ticket can only be released as cancelled")
    confirmed_stopped = parse_gpu_ids(args.confirmed_stopped)
    if ticket["status"] in {"running", "stale"}:
        if confirmed_stopped != ticket["assigned_gpus"]:
            raise TicketError(
                "--confirmed-stopped must exactly match assigned GPUs "
                f"{ticket['assigned_gpus']} after checking remote processes"
            )
    elif confirmed_stopped is not None:
        raise TicketError(
            "--confirmed-stopped is only valid for a running or stale ticket"
        )
    result = clean_text(args.result, "result", 500)
    previous_status = ticket["status"]
    current = utc_now()
    ticket.update(
        {
            "status": args.outcome,
            "assigned_gpus": [],
            "updated_at": timestamp(current),
            "finished_at": timestamp(current),
            "heartbeat_due_at": None,
            "reservation_expires_at": None,
            "result": result,
        }
    )
    append_event("release", ticket, f"from {previous_status}: {result}")
    changes = reconcile_state(state, config)
    persist(state, config)
    suffix = f"; {'; '.join(changes)}" if changes else ""
    print(f"{ticket['id']} status={ticket['status']} released{suffix}")


def command_status(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    pending = pending_time_transitions(state)
    if args.ticket_id:
        payload: Any = result_payload(require_ticket(state, args.ticket_id))
        payload["profile"] = PROFILE["slug"] if PROFILE else None
        payload["ledger_profile"] = config.get("profile")
        payload["coordination_uid"] = config.get("coordination_uid")
    else:
        selected_server = (
            f"{PROFILE['ssh']['user']}@{PROFILE['ssh']['host']}:{PROFILE['ssh']['port']}"
            if PROFILE
            else config["server"]
        )
        payload = {
            "server": selected_server,
            "profile": PROFILE["slug"] if PROFILE else None,
            "ledger_profile": config.get("profile"),
            "coordination_uid": config.get("coordination_uid"),
            "gpu_ids": config["gpu_ids"],
            "gpu_devices": config.get("gpu_devices", []),
            "occupied_gpus": sorted(occupied_gpus(state)),
            "tensorboard_port_range": list(tensorboard_port_bounds(config)),
            "occupied_tensorboard_ports": sorted(occupied_tensorboard_ports(state)),
            "tickets": sorted(
                state["tickets"].values(), key=lambda ticket: ticket["created_at"]
            ),
            "board": str(BOARD_PATH),
            "pending_reconcile": pending,
            "snapshot_note": "read-only; run reconcile to apply pending transitions",
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.ticket_id:
        tensorboard_status = (payload.get("tensorboard") or {}).get("status", "none")
        print(
            f"{payload['id']} status={payload['status']} "
            f"assigned={payload.get('assigned_gpus', [])} "
            f"tensorboard={tensorboard_status} file={payload['ticket_file']}"
        )
        return
    active = [
        ticket
        for ticket in payload["tickets"]
        if ticket["status"] in {"queued", *HOLDING_STATES}
    ]
    print(
        f"server={config['server']} occupied={payload['occupied_gpus']} "
        f"tensorboard_ports={payload['occupied_tensorboard_ports']}"
    )
    for ticket in active:
        tensorboard_status = (ticket.get("tensorboard") or {}).get("status", "none")
        print(
            f"{ticket['id']} status={ticket['status']} project={ticket['project']} "
            f"owner={ticket['owner']} assigned={ticket['assigned_gpus']} "
            f"tensorboard={tensorboard_status}"
        )
    if not active:
        print("no active or queued tickets")
    print(f"board={BOARD_PATH}")
    if pending:
        print("pending_reconcile=" + "; ".join(pending))


def command_reconcile(state: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> None:
    changes = reconcile_state(state, config)
    persist(state, config)
    print("; ".join(changes) if changes else "no state changes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate project GPU leases with an atomic local ticket ledger."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize or re-render the ledger.")

    reserve = subparsers.add_parser("reserve", help="Reserve GPUs or enter the FIFO queue.")
    reserve.add_argument("--project", required=True)
    reserve.add_argument("--owner", required=True)
    reserve.add_argument("--purpose", required=True)
    reserve.add_argument("--gpus", type=int)
    reserve.add_argument("--gpu-ids", help="Explicit physical IDs, for example 0,2.")
    reserve.add_argument("--expected", default="1h")
    reserve.add_argument("--json", action="store_true")

    start = subparsers.add_parser("start", help="Mark a checked reservation running.")
    start.add_argument("ticket_id")
    start.add_argument("--confirmed-idle", required=True)
    start.add_argument("--session", required=True)
    start.add_argument("--remote-workdir", required=True)
    start.add_argument("--summary", required=True)
    start.add_argument("--expected")

    heartbeat = subparsers.add_parser("heartbeat", help="Refresh a running or stale ticket.")
    heartbeat.add_argument("ticket_id")
    heartbeat.add_argument("--expected")

    release = subparsers.add_parser("release", help="Release a queued or allocated ticket.")
    release.add_argument("ticket_id")
    release.add_argument(
        "--outcome", choices=["completed", "failed", "cancelled"], required=True
    )
    release.add_argument(
        "--confirmed-stopped",
        help="Exact assigned GPU IDs after verifying no tracked process remains.",
    )
    release.add_argument("--result", required=True)

    tensorboard = subparsers.add_parser(
        "tensorboard", help="Register or update a ticket's TensorBoard sidecar."
    )
    tensorboard.add_argument("ticket_id")
    tensorboard.add_argument("--status", choices=sorted(TENSORBOARD_STATES), required=True)
    tensorboard.add_argument("--logdir")
    tensorboard.add_argument("--env-prefix")
    tensorboard.add_argument(
        "--remote-port",
        type=int,
        help=(
            "Remote loopback port; starting auto-allocates from the configured "
            f"pool (default {DEFAULT_TENSORBOARD_PORT_START}.."
            f"{DEFAULT_TENSORBOARD_PORT_END}) when omitted."
        ),
    )
    tensorboard.add_argument("--path-prefix")
    tensorboard.add_argument("--session")
    tensorboard.add_argument("--pid", type=int)
    tensorboard.add_argument("--process-start-ticks", type=int)
    tensorboard.add_argument("--boot-id")
    tensorboard.add_argument("--version")
    tensorboard.add_argument("--command-sha256")
    tensorboard.add_argument("--last-error")
    tensorboard.add_argument(
        "--expected-generation",
        type=int,
        help=(
            "Compare-and-set guard for a previously read TensorBoard generation; "
            "zero requires an unconfigured source."
        ),
    )

    status = subparsers.add_parser("status", help="Show the board or one ticket.")
    status.add_argument("ticket_id", nargs="?")
    status.add_argument("--json", action="store_true")

    subparsers.add_parser("reconcile", help="Expire launch windows and mark overdue jobs stale.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        configure_profile_paths()
        config = load_config()
        if args.command == "status":
            command_status(load_state(), config, args)
            return 0

        TICKET_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        TICKET_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(TICKET_ROOT, 0o700)
        os.chmod(TICKET_DIR, 0o700)
        events_fd = os.open(EVENTS_PATH, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(events_fd)
        lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(lock_fd)
        with LOCK_PATH.open("r+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            state = load_state()
            commands = {
                "init": command_init,
                "reserve": command_reserve,
                "start": command_start,
                "heartbeat": command_heartbeat,
                "release": command_release,
                "tensorboard": command_tensorboard,
                "status": command_status,
                "reconcile": command_reconcile,
            }
            commands[args.command](state, config, args)
    except TicketError as exc:
        print(f"gpu-ticket: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
