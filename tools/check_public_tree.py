#!/usr/bin/env python3
"""Fail closed when a public release tree contains secrets or local state."""

from __future__ import annotations

import argparse
import ast
import binascii
import hashlib
import io
import json
import re
import stat
import struct
import sys
import tokenize
import zlib
from pathlib import Path


SCRIPT_ROOT = (
    Path(__file__).resolve().parents[1] / "skills" / "remote-gpu-dev" / "scripts"
)
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
sys.dont_write_bytecode = True

from credential_guard import (  # noqa: E402
    ASSIGNMENT_CANDIDATE_RE,
    FIELD_NAME_BODY,
    PRIVATE_KEY_RE,
    SECRET_VALUE_RE,
    StructuredSecretScanError,
    contains_private_jwk,
    contains_secret,
    contains_structured_secret,
    decode_json_for_secret_scan,
    is_credential_field_name,
    normalize_for_secret_scan,
)

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".gitignore", "LICENSE"}
FORBIDDEN_NAMES = {
    ".env",
    "auth.json",
    "state.json",
    "events.jsonl",
    "known_hosts",
}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".ckpt", ".pth", ".safetensors", ".pyc"}
MAX_PUBLIC_SOURCE_BYTES = 2 * 1024 * 1024
MAX_PUBLIC_IMAGE_EDGE = 4096
MAX_PUBLIC_IMAGE_PIXELS = 8_294_400
# Python legitimately uses names such as ``token`` for transient capability
# values.  In source files, reject only a credential-labelled hard-coded
# string/bytes literal; non-Python configuration remains fail-closed on the
# label and assignment operator alone.
PYTHON_CREDENTIAL_LITERAL_RE = re.compile(
    rf"(?ix)(?<![A-Za-z0-9_])(?P<label_quote>['\"]?)"
    rf"(?P<field>{FIELD_NAME_BODY})(?P=label_quote)[ \t]*"
    r"(?:"
    r"=[ \t]*(?:\([ \t]*){0,4}[rubf]{0,2}['\"]"
    r"|:[ \t]*(?:"
    r"[rubf]{0,2}['\"]"
    r"|[A-Za-z_][A-Za-z0-9_.\[\], |]{0,127}[ \t]*="
    r"[ \t]*(?:\([ \t]*){0,4}[rubf]{0,2}['\"]"
    r")"
    r")"
)
SECRET_PATTERNS = [SECRET_VALUE_RE]
MAX_STATIC_LITERAL_BYTES = MAX_PUBLIC_SOURCE_BYTES
MAX_RECONSTRUCTED_BYTES = MAX_PUBLIC_SOURCE_BYTES
MAX_STATIC_CONTAINER_NODES = 10_000
_NOT_STATIC = object()


def _contains_credential_assignment(
    text: str,
    pattern: re.Pattern[str],
    *,
    allow_browser_credentials_mode: bool = False,
) -> bool:
    for match in pattern.finditer(text):
        field = match.group("field")
        if not is_credential_field_name(field):
            continue
        if (
            allow_browser_credentials_mode
            and tuple(part.upper() for part in re.findall(r"[A-Za-z0-9]+", field))
            in {("CREDENTIAL",), ("CREDENTIALS",)}
            and re.match(
                r"[ \t]*['\"](?:include|omit|same-origin)['\"]",
                text[match.end() :],
                flags=re.IGNORECASE,
            )
        ):
            continue
        return True
    return False


def _static_joined_text(node: ast.JoinedStr) -> str | None:
    """Evaluate a constant f-string without evaluating user code."""

    parts: list[str] = []
    total = 0
    for child in node.values:
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            part = child.value
        elif isinstance(child, ast.FormattedValue) and isinstance(
            child.value, ast.Constant
        ):
            value = child.value.value
            if not isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
                return None
            if child.conversion == -1:
                converted = value
            elif child.conversion == ord("s"):
                converted = str(value)
            elif child.conversion == ord("r"):
                converted = repr(value)
            elif child.conversion == ord("a"):
                converted = ascii(value)
            else:
                return None
            if child.format_spec is None:
                format_spec = ""
            else:
                format_spec = _static_joined_text(child.format_spec)
                if format_spec is None or format_spec:
                    return None
            try:
                part = format(converted, format_spec)
            except (TypeError, ValueError):
                return None
        else:
            return None
        total += len(part)
        if total > MAX_STATIC_LITERAL_BYTES:
            return None
        parts.append(part)
    return "".join(parts)


def _static_text_value(node: ast.AST | None) -> str | bytes | None:
    """Evaluate bounded literal text, constant f-strings, and ``+`` chains."""

    if node is None:
        return None
    pending = [node]
    parts: list[str] | list[bytes] = []
    result_type: type[str] | type[bytes] | None = None
    total = 0
    while pending:
        current = pending.pop()
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            pending.append(current.right)
            pending.append(current.left)
            continue
        if isinstance(current, ast.Constant) and isinstance(
            current.value, (str, bytes)
        ):
            part = current.value
        elif isinstance(current, ast.JoinedStr):
            part = _static_joined_text(current)
            if part is None:
                return None
        else:
            return None
        if result_type is None:
            result_type = type(part)
        elif type(part) is not result_type:
            return None
        total += len(part)
        if total > MAX_STATIC_LITERAL_BYTES:
            return None
        parts.append(part)
    if result_type is bytes:
        return b"".join(parts)  # type: ignore[arg-type]
    return "".join(parts)  # type: ignore[arg-type]


def _is_static_text(node: ast.AST | None) -> bool:
    return _static_text_value(node) is not None


def _python_target_fields(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Subscript):
        key = _static_text_value(node.slice)
        if isinstance(key, str):
            return (key,)
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(
            field
            for child in node.elts
            for field in _python_target_fields(child)
        )
    return ()


def _call_argument(
    node: ast.Call, position: int, keyword_name: str
) -> ast.AST | None:
    if len(node.args) > position:
        return node.args[position]
    return next(
        (
            keyword.value
            for keyword in node.keywords
            if keyword.arg == keyword_name
        ),
        None,
    )


def _credential_environment_default(node: ast.Call) -> bool:
    """Detect non-empty static defaults in standard ``os.environ`` access."""

    function = node.func
    os_getenv = (
        isinstance(function, ast.Attribute)
        and function.attr == "getenv"
        and isinstance(function.value, ast.Name)
        and function.value.id == "os"
    )
    environ_get = (
        isinstance(function, ast.Attribute)
        and function.attr == "get"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "environ"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "os"
    )
    environ_setdefault = (
        isinstance(function, ast.Attribute)
        and function.attr == "setdefault"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "environ"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "os"
    )
    if not (os_getenv or environ_get or environ_setdefault):
        return False
    key = _static_text_value(_call_argument(node, 0, "key"))
    default = _static_text_value(_call_argument(node, 1, "default"))
    return (
        isinstance(key, str)
        and is_credential_field_name(key)
        and isinstance(default, (str, bytes))
        and bool(default.strip())
    )


def _dict_pair_credential_literal(node: ast.Call) -> bool:
    """Detect credential pairs passed directly to the built-in ``dict`` form."""

    if not (
        isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        return False
    for pair in node.args[0].elts:
        if not isinstance(pair, (ast.List, ast.Tuple)) or len(pair.elts) != 2:
            continue
        field = _static_text_value(pair.elts[0])
        if (
            isinstance(field, str)
            and is_credential_field_name(field)
            and _is_static_text(pair.elts[1])
        ):
            return True
    return False


def _python_tree_contains_credential_literal(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        targets: tuple[ast.AST, ...] = ()
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = (node.target,)
            value = node.value
        if _is_static_text(value) and any(
            is_credential_field_name(field)
            for target in targets
            for field in _python_target_fields(target)
        ):
            return True

        if isinstance(node, ast.Dict):
            for key, item in zip(node.keys, node.values):
                field = _static_text_value(key)
                if (
                    isinstance(field, str)
                    and is_credential_field_name(field)
                    and _is_static_text(item)
                ):
                    return True

        if isinstance(node, ast.Call):
            if (
                _credential_environment_default(node)
                or _dict_pair_credential_literal(node)
            ):
                return True
            if any(
                keyword.arg is not None
                and is_credential_field_name(keyword.arg)
                and _is_static_text(keyword.value)
                for keyword in node.keywords
            ):
                return True

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            positional = (*node.args.posonlyargs, *node.args.args)
            positional_defaults = (
                zip(positional[-len(node.args.defaults) :], node.args.defaults)
                if node.args.defaults
                else ()
            )
            default_pairs = (
                *positional_defaults,
                *zip(node.args.kwonlyargs, node.args.kw_defaults),
            )
            if any(
                default is not None
                and is_credential_field_name(argument.arg)
                and _is_static_text(default)
                for argument, default in default_pairs
            ):
                return True
    return False


def _static_json_literal(
    node: ast.AST,
    memo: dict[int, object],
    *,
    depth: int = 0,
) -> object:
    """Evaluate only expression-local JSON-like literals used in source."""

    identity = id(node)
    if identity in memo:
        return memo[identity]
    if depth > 32 or len(memo) > MAX_STATIC_CONTAINER_NODES:
        raise ValueError("static container scan budget exceeded")

    text = _static_text_value(node)
    if text is not None:
        result: object = text
    elif isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, float))
    ):
        result = node.value
    elif isinstance(node, (ast.List, ast.Tuple)):
        children: list[object] = []
        for child in node.elts:
            value = _static_json_literal(child, memo, depth=depth + 1)
            if value is _NOT_STATIC:
                result = _NOT_STATIC
                break
            children.append(value)
        else:
            result = children
    elif isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
        mapping: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            assert key_node is not None
            key = _static_json_literal(key_node, memo, depth=depth + 1)
            value = _static_json_literal(value_node, memo, depth=depth + 1)
            if not isinstance(key, str) or value is _NOT_STATIC or key in mapping:
                result = _NOT_STATIC
                break
            mapping[key] = value
        else:
            result = mapping
    else:
        result = _NOT_STATIC
    memo[identity] = result
    return result


def _python_static_container_findings(
    tree: ast.AST,
) -> tuple[bool, str | None]:
    """Scan maximal literal dict/list values without interpreting Python."""

    memo: dict[int, object] = {}
    secret = False
    artifact_kind: str | None = None
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call):
            keyword_mapping: dict[str, object] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                value = _static_json_literal(keyword.value, memo)
                if value is not _NOT_STATIC:
                    keyword_mapping[keyword.arg] = value
            if contains_private_jwk(keyword_mapping):
                secret = True
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
            value = _static_json_literal(node, memo)
            if value is not _NOT_STATIC:
                secret = secret or contains_structured_secret(value)
                artifact_kind = artifact_kind or _private_json_artifact_kind(value)
                continue
        pending.extend(ast.iter_child_nodes(node))
    return secret, artifact_kind


def _static_text_node_info(
    tree: ast.AST,
) -> tuple[list[ast.AST], dict[int, tuple[type[str] | type[bytes], int]]]:
    """Annotate static text nodes once, bottom-up, without materializing chains."""

    nodes = list(ast.walk(tree))
    info: dict[int, tuple[type[str] | type[bytes], int]] = {}
    for node in reversed(nodes):
        node_type: type[str] | type[bytes] | None = None
        length = 0
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            node_type = type(node.value)
            length = len(node.value)
        elif isinstance(node, ast.JoinedStr):
            value = _static_joined_text(node)
            if value is not None:
                node_type = str
                length = len(value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = info.get(id(node.left))
            right = info.get(id(node.right))
            if left is not None and right is not None and left[0] is right[0]:
                node_type = left[0]
                length = left[1] + right[1]
        if node_type is not None and length <= MAX_STATIC_LITERAL_BYTES:
            info[id(node)] = (node_type, length)
    return nodes, info


def _materialize_static_text(
    node: ast.AST, info: dict[int, tuple[type[str] | type[bytes], int]]
) -> str | bytes:
    parts: list[str] | list[bytes] = []
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.BinOp) and id(current) in info:
            pending.append(current.right)
            pending.append(current.left)
        elif isinstance(current, ast.JoinedStr):
            value = _static_joined_text(current)
            assert value is not None
            parts.append(value)
        else:
            assert isinstance(current, ast.Constant)
            assert isinstance(current.value, (str, bytes))
            parts.append(current.value)
    if info[id(node)][0] is bytes:
        return b"".join(parts)  # type: ignore[arg-type]
    return "".join(parts)  # type: ignore[arg-type]


def _iter_maximal_static_text_values(tree: ast.AST):
    """Yield disjoint static values after one postorder classification pass."""

    _, info = _static_text_node_info(tree)
    total = 0
    pending = [tree]
    while pending:
        node = pending.pop()
        node_info = info.get(id(node)) if isinstance(node, ast.expr) else None
        if node_info is not None:
            total += node_info[1]
            if total > MAX_RECONSTRUCTED_BYTES:
                raise ValueError("reconstructed static text exceeds scan budget")
            yield _materialize_static_text(node, info)
            continue
        pending.extend(ast.iter_child_nodes(node))


def _scan_text_value(value: str | bytes) -> str:
    if isinstance(value, str):
        return value
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1")


def _python_nonliteral_contains_secret(text: str) -> bool:
    """Scan comments and bare identifiers while string tokens use AST values."""

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return any(
            token.type in {tokenize.COMMENT, tokenize.NAME}
            and contains_secret(token.string)
            for token in tokens
        )
    except (IndentationError, tokenize.TokenError):
        return True


def _python_findings(
    text: str, normalized: str
) -> tuple[bool, bool, str | None]:
    """Return credential sink, reconstructed secret, and runtime-artifact findings."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return (
            _contains_credential_assignment(normalized, PYTHON_CREDENTIAL_LITERAL_RE),
            bool(SECRET_VALUE_RE.search(normalized)),
            None,
        )
    except (RecursionError, ValueError):
        # Parser exhaustion is a release failure, represented by both booleans.
        return True, True, None

    credential_literal = _python_tree_contains_credential_literal(tree)
    secret_literal = _python_nonliteral_contains_secret(text)
    artifact_kind: str | None = None
    try:
        container_secret, container_artifact = _python_static_container_findings(tree)
        secret_literal = secret_literal or container_secret
        artifact_kind = artifact_kind or container_artifact
        for value in _iter_maximal_static_text_values(tree):
            static_text = _scan_text_value(value)
            value_secret, value_artifact = _static_text_findings(static_text)
            secret_literal = secret_literal or value_secret
            artifact_kind = artifact_kind or value_artifact
    except ValueError:
        return credential_literal, True, artifact_kind
    return credential_literal, secret_literal, artifact_kind


def _python_contains_credential_literal(text: str, normalized: str) -> bool:
    """Compatibility wrapper for the credential-sink focused regressions."""

    return _python_findings(text, normalized)[0]


def _json_contains_secret(value: object) -> bool:
    """Use the shared bounded scan for every decoded JSON key and string leaf."""

    return contains_structured_secret(value)


PROFILE_TOP_LEVEL_FIELDS = frozenset(
    {
        "created_at",
        "dashboard",
        "git",
        "gpu",
        "local",
        "name",
        "network",
        "remote",
        "schema_version",
        "slug",
        "ssh",
        "trust",
    }
)
SANITIZED_PROFILE_EXAMPLE_SHA256 = (
    "f9712c3373e82f2690e3865e7a955b2bd4a43d775dfe8763acfc1b75f87701ac"
)
TICKET_CONFIG_FIELDS = frozenset(
    {
        "coordination_uid",
        "gpu_devices",
        "gpu_ids",
        "heartbeat_grace_minutes",
        "profile",
        "recent_terminal_limit",
        "reservation_ttl_minutes",
        "schema_version",
        "server",
        "tensorboard_port_end",
        "tensorboard_port_start",
    }
)
TICKET_STATE_FIELDS = frozenset({"schema_version", "tickets", "updated_at"})
TICKET_EVENT_FIELDS = frozenset(
    {"action", "assigned_gpus", "at", "detail", "status", "ticket_id"}
)
TICKET_RECORD_FIELDS = frozenset(
    {
        "assigned_gpus",
        "created_at",
        "expected_duration_minutes",
        "id",
        "owner",
        "project",
        "purpose",
        "requested_gpu_ids",
        "requested_gpus",
        "status",
        "updated_at",
    }
)
TICKET_STATES = frozenset(
    {"cancelled", "completed", "expired", "failed", "queued", "reserved", "running", "stale"}
)
TICKET_ACTIONS = frozenset(
    {
        "auto-expire",
        "auto-reserve",
        "auto-stale",
        "heartbeat",
        "queue",
        "release",
        "reserve",
        "start",
        "tensorboard",
    }
)
TICKET_ID_RE = re.compile(r"GPU-[\w-]{1,156}\Z", flags=re.UNICODE)
UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


def _has_fields(value: object, fields: frozenset[str]) -> bool:
    return isinstance(value, dict) and fields.issubset(value)


def _looks_like_profile_artifact(value: object) -> bool:
    if not _has_fields(value, PROFILE_TOP_LEVEL_FIELDS):
        return False
    assert isinstance(value, dict)
    return (
        _has_fields(
            value.get("trust"),
            frozenset(
                {
                    "coordination_uid",
                    "host_key_fingerprints",
                    "remote_machine_id_sha256",
                    "server_uid",
                }
            ),
        )
        and _has_fields(
            value.get("ssh"),
            frozenset(
                {"host", "identity_file", "known_hosts_file", "port", "user"}
            ),
        )
        and _has_fields(
            value.get("local"), frozenset({"projects_root", "ticket_root"})
        )
        and _has_fields(
            value.get("remote"),
            frozenset({"durable_root", "records_root", "temp_root"}),
        )
        and _has_fields(value.get("gpu"), frozenset({"devices", "environment", "ids"}))
    )


def _is_sanitized_profile_example(value: object) -> bool:
    if not _looks_like_profile_artifact(value):
        return False
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == SANITIZED_PROFILE_EXAMPLE_SHA256


def _looks_like_ticket_config(value: object) -> bool:
    return (
        _has_fields(value, TICKET_CONFIG_FIELDS)
        and isinstance(value, dict)
        and value.get("schema_version") == 1
        and isinstance(value.get("server"), str)
        and "@" in value["server"]
        and isinstance(value.get("profile"), str)
        and isinstance(value.get("coordination_uid"), str)
        and value["coordination_uid"].startswith("sha256:")
        and isinstance(value.get("gpu_ids"), list)
        and isinstance(value.get("gpu_devices"), list)
    )


def _looks_like_ticket_state(value: object) -> bool:
    if not (
        isinstance(value, dict)
        and set(value) == TICKET_STATE_FIELDS
        and value.get("schema_version") == 1
        and isinstance(value.get("updated_at"), str)
        and UTC_TIMESTAMP_RE.fullmatch(value["updated_at"])
        and isinstance(value.get("tickets"), dict)
    ):
        return False
    tickets = value["tickets"]
    return not tickets or all(
        isinstance(ticket_id, str)
        and _looks_like_ticket_record(ticket)
        and ticket.get("id") == ticket_id
        for ticket_id, ticket in tickets.items()
    )


def _looks_like_ticket_record(value: object) -> bool:
    return (
        _has_fields(value, TICKET_RECORD_FIELDS)
        and isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and TICKET_ID_RE.fullmatch(value["id"]) is not None
        and value.get("status") in TICKET_STATES
        and isinstance(value.get("project"), str)
        and isinstance(value.get("owner"), str)
        and isinstance(value.get("purpose"), str)
        and isinstance(value.get("assigned_gpus"), list)
        and isinstance(value.get("requested_gpus"), int)
        and isinstance(value.get("created_at"), str)
        and UTC_TIMESTAMP_RE.fullmatch(value["created_at"])
        and isinstance(value.get("updated_at"), str)
        and UTC_TIMESTAMP_RE.fullmatch(value["updated_at"])
    )


def _looks_like_ticket_event(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == TICKET_EVENT_FIELDS
        and isinstance(value.get("ticket_id"), str)
        and TICKET_ID_RE.fullmatch(value["ticket_id"]) is not None
        and isinstance(value.get("assigned_gpus"), list)
        and value.get("action") in TICKET_ACTIONS
        and value.get("status") in TICKET_STATES
        and isinstance(value.get("detail"), str)
        and isinstance(value.get("at"), str)
        and UTC_TIMESTAMP_RE.fullmatch(value["at"])
    )


def _looks_like_ticket_overview(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("snapshot_note") != (
        "read-only; run reconcile to apply pending transitions"
    ):
        return False
    fields = frozenset(
        {
            "board",
            "coordination_uid",
            "gpu_devices",
            "gpu_ids",
            "occupied_gpus",
            "profile",
            "server",
            "tickets",
        }
    )
    return (
        _has_fields(value, fields)
        and isinstance(value.get("tickets"), list)
        and all(_looks_like_ticket_record(ticket) for ticket in value["tickets"])
    )


def _private_json_artifact_kind(value: object) -> str | None:
    if _looks_like_profile_artifact(value) and not _is_sanitized_profile_example(value):
        return "remote-gpu-dev profile"
    if _looks_like_ticket_config(value):
        return "remote-gpu-dev ticket config"
    if _looks_like_ticket_state(value):
        return "remote-gpu-dev ticket state"
    if _looks_like_ticket_record(value):
        return "remote-gpu-dev ticket record"
    if _looks_like_ticket_overview(value):
        return "remote-gpu-dev ticket overview"
    if _looks_like_ticket_event(value):
        return "remote-gpu-dev ticket event"
    return None


def _private_text_artifact_kind(text: str) -> str | None:
    if (
        text.startswith("# Remote GPU Ticket Board\n")
        and "\nGenerated at `" in text[:512]
        and "Use `gpu_ticket.py`; do not edit this board." in text
        and "## Active allocations" in text
        and "## Queue" in text
        and "## Recent terminal tickets" in text
    ):
        return "remote-gpu-dev ticket board"
    if (
        text.startswith("---\n")
        and re.search(r'(?m)^id:\s*"?GPU-[^"\r\n]+"?\s*$', text[:1024])
        and "\n---\n\n# GPU-" in text[:2048]
        and "| Field | Value |" in text
        and "| Remote workdir |" in text
        and "| TensorBoard status |" in text
        and "This file is generated from `state.json`. "
        "Use `gpu_ticket.py` for updates." in text
    ):
        return "remote-gpu-dev ticket Markdown"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(line.startswith("{") and line.endswith("}") for line in lines):
        events: list[object] = []
        for line in lines:
            try:
                events.append(decode_json_for_secret_scan(line))
            except (json.JSONDecodeError, StructuredSecretScanError):
                break
        else:
            if all(_looks_like_ticket_event(value) for value in events):
                return "remote-gpu-dev ticket event log"
    return None


def _static_text_findings(text: str) -> tuple[bool, str | None]:
    """Scan one reconstructed static string with bounded shared classifiers."""

    secret = contains_secret(text)
    document: object | None = None
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            document = decode_json_for_secret_scan(text)
        except json.JSONDecodeError:
            document = None
        except StructuredSecretScanError:
            return True, None
        else:
            secret = secret or _json_contains_secret(document)
    artifact_kind = (
        _private_json_artifact_kind(document) if document is not None else None
    )
    if artifact_kind is None:
        artifact_kind = _private_text_artifact_kind(text)
    return secret, artifact_kind


def _contains_binary_controls(text: str) -> bool:
    """Reject control bytes that cannot occur in ordinary UTF-8 source text."""

    allowed = {"\t", "\n", "\r", "\f"}
    return any(
        (ord(character) < 0x20 and character not in allowed)
        or 0x7F <= ord(character) <= 0x9F
        for character in text
    )


def _validate_public_png(data: bytes) -> str | None:
    """Validate a metadata-free, bounded, non-animated README PNG."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        return "invalid PNG signature"

    offset = len(signature)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if len(data) - offset < 12:
            return "truncated PNG chunk"
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            return "PNG chunk length exceeds file"
        if not re.fullmatch(rb"[A-Za-z]{4}", chunk_type):
            return "invalid PNG chunk type"
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            return f"PNG CRC mismatch in {chunk_type.decode('ascii')}"
        chunks.append((chunk_type, chunk_data))
        offset = chunk_end
        if chunk_type == b"IEND":
            break

    if offset != len(data):
        return "trailing bytes after PNG IEND"
    if not chunks or chunks[0][0] != b"IHDR":
        return "PNG IHDR must be first"
    if len(chunks[0][1]) != 13:
        return "invalid PNG IHDR length"
    if sum(chunk_type == b"IHDR" for chunk_type, _ in chunks) != 1:
        return "PNG must contain exactly one IHDR"
    if chunks[-1] != (b"IEND", b""):
        return "PNG must end with one empty IEND"
    if sum(chunk_type == b"IEND" for chunk_type, _ in chunks) != 1:
        return "PNG must contain exactly one IEND"

    allowed_chunks = {b"IHDR", b"IDAT", b"IEND"}
    unsupported = [
        chunk_type.decode("ascii")
        for chunk_type, _ in chunks
        if chunk_type not in allowed_chunks
    ]
    if unsupported:
        return "PNG metadata or unsupported chunk is forbidden: " + ", ".join(
            sorted(set(unsupported))
        )

    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", chunks[0][1])
    )
    if width < 1 or height < 1:
        return "PNG dimensions must be positive"
    if width > MAX_PUBLIC_IMAGE_EDGE or height > MAX_PUBLIC_IMAGE_EDGE:
        return "PNG dimensions exceed public limit"
    if width * height > MAX_PUBLIC_IMAGE_PIXELS:
        return "PNG pixel count exceeds public limit"
    if bit_depth != 8 or color_type not in {2, 6}:
        return "PNG must use 8-bit RGB or RGBA pixels"
    if (compression, filter_method, interlace) != (0, 0, 0):
        return "PNG must use standard compression/filtering and no interlace"

    idat_indices = [
        index for index, (chunk_type, _) in enumerate(chunks) if chunk_type == b"IDAT"
    ]
    if not idat_indices:
        return "PNG has no IDAT data"
    if idat_indices != list(range(idat_indices[0], idat_indices[-1] + 1)):
        return "PNG IDAT chunks must be consecutive"

    channels = 3 if color_type == 2 else 4
    row_bytes = width * channels
    expected_size = height * (row_bytes + 1)
    compressed = b"".join(chunks[index][1] for index in idat_indices)
    try:
        decompressor = zlib.decompressobj()
        pixels = decompressor.decompress(compressed, expected_size + 1)
    except zlib.error:
        return "PNG IDAT stream is invalid"
    if len(pixels) > expected_size or decompressor.unconsumed_tail:
        return "PNG decompressed pixel size is inconsistent with IHDR"
    try:
        pixels += decompressor.flush(expected_size + 1 - len(pixels))
    except zlib.error:
        return "PNG IDAT stream is invalid"
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        return "PNG IDAT stream is incomplete or has trailing data"
    if len(pixels) != expected_size:
        return "PNG decompressed pixel size is inconsistent with IHDR"
    if any(pixels[row * (row_bytes + 1)] > 4 for row in range(height)):
        return "PNG contains an invalid row filter"
    return None


def check(root: Path, *, local_deny: tuple[str, ...] = ()) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if ".git" in path.parts or "__pycache__" in path.parts:
            if "__pycache__" in path.parts:
                errors.append(f"generated cache is forbidden: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"symlink is forbidden in public tree: {relative}")
            continue
        if path.is_dir():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"private/runtime file is forbidden: {relative}")
            continue
        if path.stat().st_size > MAX_PUBLIC_SOURCE_BYTES:
            errors.append(f"file exceeds public source size limit: {relative}")
            continue
        if path.suffix.lower() == ".png":
            png_error = _validate_public_png(path.read_bytes())
            if png_error is not None:
                errors.append(f"invalid public PNG {relative}: {png_error}")
            continue
        text_like = path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if text_like:
                errors.append(f"text-like file is not UTF-8: {relative}")
            else:
                errors.append(f"binary or non-UTF-8 file is forbidden: {relative}")
            continue
        if _contains_binary_controls(text):
            errors.append(f"binary control content is forbidden: {relative}")
            continue
        for value in local_deny:
            if value in text:
                errors.append(f"workstation-specific value in {relative}")
        normalized = normalize_for_secret_scan(text)
        credential_assignment = False
        reconstructed_secret = False
        reconstructed_artifact_kind: str | None = None
        assignment_pattern = ASSIGNMENT_CANDIDATE_RE
        if path.suffix.lower() == ".py":
            assignment_pattern = PYTHON_CREDENTIAL_LITERAL_RE
            (
                credential_assignment,
                reconstructed_secret,
                reconstructed_artifact_kind,
            ) = _python_findings(
                text,
                normalized,
            )
        else:
            credential_assignment = _contains_credential_assignment(
                normalized,
                assignment_pattern,
                allow_browser_credentials_mode=path.suffix.lower() == ".js",
            )
        document: object | None = None
        stripped_text = text.lstrip()
        if path.suffix.lower() == ".json" or stripped_text.startswith(("{", "[")):
            try:
                document = decode_json_for_secret_scan(text)
            except json.JSONDecodeError:
                if path.suffix.lower() == ".json":
                    errors.append(
                        f"JSON file could not be structurally scanned: {relative}"
                    )
            except StructuredSecretScanError:
                if path.suffix.lower() == ".json":
                    errors.append(
                        f"JSON file could not be structurally scanned: {relative}"
                    )
                elif stripped_text.startswith(("{", "[")):
                    reconstructed_secret = True
            else:
                reconstructed_secret = reconstructed_secret or (
                    _json_contains_secret(document)
                )
        artifact_kind = (
            _private_json_artifact_kind(document)
            if document is not None
            else None
        )
        artifact_kind = artifact_kind or reconstructed_artifact_kind
        artifact_kind = artifact_kind or _private_text_artifact_kind(text)
        if artifact_kind:
            errors.append(f"private/runtime {artifact_kind} is forbidden: {relative}")
        if credential_assignment:
            errors.append(
                f"secret-like value in {relative}: "
                f"{assignment_pattern.pattern[:36]}"
            )
        raw_secret = path.suffix.lower() != ".py" and any(
            pattern.search(normalized) for pattern in SECRET_PATTERNS
        )
        if reconstructed_secret or raw_secret:
            errors.append(f"secret-like value in {relative}: reconstructed/static")
        if not text_like:
            errors.append(f"unrecognized public file type is forbidden: {relative}")
    return errors


def load_local_deny(path: Path, release_root: Path) -> tuple[str, ...]:
    resolved = path.expanduser().resolve(strict=True)
    root = release_root.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("the local deny file must be stored outside the release tree")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("the local deny file must be a regular file")
    if stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError("the local deny file must have mode 0600 or stricter")
    try:
        values = tuple(
            line.strip()
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except UnicodeDecodeError as exc:
        raise ValueError("the local deny file must be UTF-8") from exc
    if not values:
        raise ValueError("the local deny file has no deny values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--local-deny-file",
        type=Path,
        help="private newline-delimited deny values stored outside the release tree",
    )
    args = parser.parse_args()
    try:
        local_deny = (
            load_local_deny(args.local_deny_file, args.root)
            if args.local_deny_file
            else ()
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    errors = check(args.root, local_deny=local_deny)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"public tree check passed: {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
