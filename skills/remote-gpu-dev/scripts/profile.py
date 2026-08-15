#!/usr/bin/env python3
"""Profile loading and validation for remote-gpu-dev.

Profiles contain topology and policy, never passwords, tokens, or private-key
contents.  A profile can be selected with REMOTE_GPU_DEV_PROFILE or by the
active-profile file under the configuration root.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from credential_guard import (
    contains_private_jwk,
    contains_secret,
    is_credential_field_name,
)
from remote_path_guard import (
    PROTECTED_RUNTIME_ENV,
    RemotePathError,
    managed_runtime_environment,
    validate_remote_layout,
)


SCHEMA_VERSION = 1
PROFILE_ENV = "REMOTE_GPU_DEV_PROFILE"
PROFILE_PATH_ENV = "REMOTE_GPU_DEV_PROFILE_PATH"
HOME_ENV = "REMOTE_GPU_DEV_HOME"
SLUG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?\Z")
ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,63}\Z")
class ProfileError(RuntimeError):
    """A safe, user-facing profile error."""


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def config_root() -> Path:
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "remote-gpu-dev").resolve()


def profiles_dir() -> Path:
    return config_root() / "profiles"


def active_profile_path() -> Path:
    return config_root() / "active-profile"


def known_hosts_path(slug: str) -> Path:
    validate_slug(slug)
    return config_root() / "known_hosts" / slug


def state_root(slug: str) -> Path:
    validate_slug(slug)
    override = os.environ.get("REMOTE_GPU_DEV_STATE_HOME")
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "state"
        base = base / "remote-gpu-dev"
    return base.resolve() / slug


def validate_slug(value: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise ProfileError(
            "profile slug must be 1-48 lowercase ASCII letters, digits, or hyphens"
        )
    return value


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower(), flags=re.ASCII).strip("-")
    value = value[:48].strip("-")
    return validate_slug(value or "gpu-server")


def compute_server_uid(
    host_key_fingerprints: list[str], remote_machine_id_sha256: str | None
) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            {
                "host_keys": sorted(host_key_fingerprints),
                "machine": remote_machine_id_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def compute_coordination_uid(gpu_uuids: list[str]) -> str:
    """Return a stable allocation namespace independent of an SSH endpoint."""

    normalized = sorted(gpu_uuids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ProfileError("coordination identity requires unique GPU UUIDs")
    return "sha256:" + hashlib.sha256(
        json.dumps(
            {"gpu_uuids": normalized},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{field} must be a JSON object")
    return value


def _require_string(
    value: Any,
    field: str,
    *,
    maximum: int = 512,
    empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"{field} must be a string")
    if value != value.strip():
        raise ProfileError(f"{field} must not have surrounding whitespace")
    if not empty and not value:
        raise ProfileError(f"{field} must not be empty")
    if len(value) > maximum:
        raise ProfileError(f"{field} must not exceed {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ProfileError(f"{field} contains control characters")
    if contains_secret(value):
        raise ProfileError(f"{field} appears to contain a secret")
    return value


def _require_int(
    value: Any, field: str, *, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ProfileError(f"{field} must be between {minimum} and {maximum}")
    return value


def _absolute_local_path(value: Any, field: str) -> str:
    rendered = _require_string(value, field)
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        raise ProfileError(f"{field} must be an absolute local path")
    normalized = str(path.resolve(strict=False))
    if normalized == "/":
        raise ProfileError(f"{field} must be more specific than /")
    return normalized


def _absolute_remote_path(value: Any, field: str) -> str:
    rendered = _require_string(value, field)
    if not rendered.startswith("/") or rendered.startswith("//"):
        raise ProfileError(f"{field} must be an absolute remote POSIX path")
    parts = rendered.split("/")
    if rendered == "/" or any(part in {".", ".."} for part in parts):
        raise ProfileError(f"{field} must be normalized and more specific than /")
    normalized = "/" + "/".join(part for part in parts if part)
    if normalized != rendered.rstrip("/"):
        raise ProfileError(f"{field} must be normalized without a trailing slash")
    return normalized


def _reject_secret_fields(value: Any, path: str = "profile") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProfileError(f"{path} contains a non-string key")
            if is_credential_field_name(key):
                raise ProfileError(
                    f"{path}.{key} is forbidden; profiles never store secrets"
                )
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")
    elif isinstance(value, str) and contains_secret(value):
        raise ProfileError(f"{path} appears to contain a secret")


def validate_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized profile or raise ProfileError."""

    value = json.loads(json.dumps(raw))
    if contains_private_jwk(value):
        raise ProfileError("profile contains private JWK material")
    _reject_secret_fields(value)
    if set(value) != {
        "schema_version",
        "name",
        "slug",
        "created_at",
        "trust",
        "ssh",
        "local",
        "remote",
        "gpu",
        "network",
        "git",
        "dashboard",
    }:
        raise ProfileError("profile has missing or unsupported top-level fields")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ProfileError("unsupported profile schema version")
    value["name"] = _require_string(value["name"], "name", maximum=80)
    value["slug"] = validate_slug(value["slug"])
    value["created_at"] = _require_string(
        value["created_at"], "created_at", maximum=40
    )

    trust = _require_object(value["trust"], "trust")
    if set(trust) != {
        "server_uid",
        "coordination_uid",
        "host_key_fingerprints",
        "remote_machine_id_sha256",
    }:
        raise ProfileError("trust has missing or unsupported fields")
    if not isinstance(trust["server_uid"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trust["server_uid"]
    ):
        raise ProfileError("trust.server_uid must be sha256:<64 lowercase hex>")
    if not isinstance(trust["coordination_uid"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trust["coordination_uid"]
    ):
        raise ProfileError("trust.coordination_uid must be sha256:<64 lowercase hex>")
    fingerprints = trust["host_key_fingerprints"]
    if (
        not isinstance(fingerprints, list)
        or not fingerprints
        or len(set(fingerprints)) != len(fingerprints)
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{20,60}", item)
            for item in fingerprints
        )
    ):
        raise ProfileError("trust.host_key_fingerprints is malformed")
    trust["host_key_fingerprints"] = sorted(fingerprints)
    machine_hash = trust["remote_machine_id_sha256"]
    if machine_hash is not None and (
        not isinstance(machine_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", machine_hash)
    ):
        raise ProfileError("trust.remote_machine_id_sha256 is malformed")
    if trust["server_uid"] != compute_server_uid(
        trust["host_key_fingerprints"], machine_hash
    ):
        raise ProfileError("trust.server_uid is inconsistent with trusted server identity")

    ssh = _require_object(value["ssh"], "ssh")
    if set(ssh) != {
        "host",
        "user",
        "port",
        "identity_file",
        "known_hosts_file",
        "connect_timeout_seconds",
        "proxy_jump",
    }:
        raise ProfileError("ssh has missing or unsupported fields")
    ssh["host"] = _require_string(ssh["host"], "ssh.host", maximum=253)
    if not re.fullmatch(r"[A-Za-z0-9._:%-]{1,253}", ssh["host"]):
        raise ProfileError("ssh.host must be a hostname, address, or safe SSH alias")
    ssh["user"] = _require_string(ssh["user"], "ssh.user", maximum=64)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", ssh["user"]):
        raise ProfileError("ssh.user has an invalid format")
    ssh["port"] = _require_int(ssh["port"], "ssh.port", minimum=1, maximum=65535)
    ssh["identity_file"] = _absolute_local_path(
        ssh["identity_file"], "ssh.identity_file"
    )
    ssh["known_hosts_file"] = _absolute_local_path(
        ssh["known_hosts_file"], "ssh.known_hosts_file"
    )
    if Path(ssh["known_hosts_file"]) != known_hosts_path(value["slug"]):
        raise ProfileError("ssh.known_hosts_file must be the profile-managed known-hosts path")
    ssh["connect_timeout_seconds"] = _require_int(
        ssh["connect_timeout_seconds"],
        "ssh.connect_timeout_seconds",
        minimum=2,
        maximum=120,
    )
    if ssh["proxy_jump"] is not None:
        ssh["proxy_jump"] = _require_string(
            ssh["proxy_jump"], "ssh.proxy_jump", maximum=255
        )
        if not re.fullmatch(r"[A-Za-z0-9._@:%\[\],-]{1,255}", ssh["proxy_jump"]):
            raise ProfileError("ssh.proxy_jump has an invalid format")

    local = _require_object(value["local"], "local")
    if set(local) != {
        "projects_root",
        "ticket_root",
        "coordination_scope",
        "proxy_host",
        "proxy_port",
    }:
        raise ProfileError("local has missing or unsupported fields")
    local["projects_root"] = _absolute_local_path(
        local["projects_root"], "local.projects_root"
    )
    local["ticket_root"] = _absolute_local_path(
        local["ticket_root"], "local.ticket_root"
    )
    if local["coordination_scope"] not in {"single-controller", "shared-filesystem"}:
        raise ProfileError(
            "local.coordination_scope must be single-controller or shared-filesystem"
        )
    local["proxy_host"] = _require_string(
        local["proxy_host"], "local.proxy_host", maximum=255
    )
    local["proxy_port"] = _require_int(
        local["proxy_port"], "local.proxy_port", minimum=1, maximum=65535
    )

    remote = _require_object(value["remote"], "remote")
    if set(remote) != {
        "temp_root",
        "durable_root",
        "git_bare_root",
        "projects_root",
        "records_root",
        "conda_executable",
        "monitor_python",
        "proxy_host",
        "proxy_port",
    }:
        raise ProfileError("remote has missing or unsupported fields")
    for field in (
        "temp_root",
        "durable_root",
        "git_bare_root",
        "projects_root",
        "records_root",
        "conda_executable",
        "monitor_python",
    ):
        remote[field] = _absolute_remote_path(remote[field], f"remote.{field}")
    remote["proxy_host"] = _require_string(
        remote["proxy_host"], "remote.proxy_host", maximum=255
    )
    remote["proxy_port"] = _require_int(
        remote["proxy_port"], "remote.proxy_port", minimum=1024, maximum=65535
    )
    try:
        validate_remote_layout(value)
    except RemotePathError as exc:
        raise ProfileError(str(exc)) from exc

    gpu = _require_object(value["gpu"], "gpu")
    if set(gpu) != {
        "ids",
        "devices",
        "reservation_ttl_minutes",
        "heartbeat_grace_minutes",
        "environment",
        "mig_mode",
    }:
        raise ProfileError("gpu has missing or unsupported fields")
    ids = gpu["ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ProfileError("gpu.ids must be a non-empty unique integer list")
    gpu["ids"] = sorted(ids)
    devices = gpu["devices"]
    if not isinstance(devices, list) or len(devices) != len(ids):
        raise ProfileError("gpu.devices must contain one entry for every managed GPU")
    normalized_devices: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    for position, device in enumerate(devices):
        device = _require_object(device, f"gpu.devices[{position}]")
        if set(device) != {"index", "uuid", "name", "memory_mib"}:
            raise ProfileError("GPU device entries have missing or unsupported fields")
        index = _require_int(
            device["index"], f"gpu.devices[{position}].index", minimum=0, maximum=65535
        )
        uuid = _require_string(
            device["uuid"], f"gpu.devices[{position}].uuid", maximum=128
        )
        if not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9-]{8,120}", uuid):
            raise ProfileError(f"gpu.devices[{position}].uuid is invalid")
        if uuid in seen_uuids:
            raise ProfileError("gpu.devices UUIDs must be unique")
        seen_uuids.add(uuid)
        normalized_devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": _require_string(
                    device["name"], f"gpu.devices[{position}].name", maximum=160
                ),
                "memory_mib": _require_int(
                    device["memory_mib"],
                    f"gpu.devices[{position}].memory_mib",
                    minimum=1,
                    maximum=10_000_000,
                ),
            }
        )
    if sorted(device["index"] for device in normalized_devices) != gpu["ids"]:
        raise ProfileError("gpu.ids and gpu.devices indices differ")
    gpu["devices"] = sorted(normalized_devices, key=lambda item: item["index"])
    if trust["coordination_uid"] != compute_coordination_uid(
        [device["uuid"] for device in gpu["devices"]]
    ):
        raise ProfileError(
            "trust.coordination_uid is inconsistent with the managed GPU UUID set"
        )
    gpu["reservation_ttl_minutes"] = _require_int(
        gpu["reservation_ttl_minutes"],
        "gpu.reservation_ttl_minutes",
        minimum=1,
        maximum=1440,
    )
    gpu["heartbeat_grace_minutes"] = _require_int(
        gpu["heartbeat_grace_minutes"],
        "gpu.heartbeat_grace_minutes",
        minimum=1,
        maximum=1440,
    )
    if gpu["mig_mode"] not in {"disabled", "unsupported"}:
        raise ProfileError(
            "this release coordinates physical GPUs only; MIG must be disabled or marked unsupported"
        )
    environment = _require_object(gpu["environment"], "gpu.environment")
    # Materialize the complete reserved set before validating user additions.
    managed_runtime_environment(value)
    normalized_environment: dict[str, str] = {}
    for key, item in environment.items():
        if not ENV_NAME_RE.fullmatch(key):
            raise ProfileError(f"invalid GPU environment variable name: {key}")
        if key in PROTECTED_RUNTIME_ENV:
            raise ProfileError(
                f"gpu.environment.{key} is managed by the filesystem policy"
            )
        normalized_item = _require_string(
            item, f"gpu.environment.{key}", maximum=512, empty=True
        )
        if "`" in normalized_item or "$(" in normalized_item:
            raise ProfileError(
                f"gpu.environment.{key} contains forbidden shell-substitution syntax"
            )
        normalized_environment[key] = normalized_item
    gpu["environment"] = normalized_environment

    network = _require_object(value["network"], "network")
    if set(network) != {
        "hf_endpoint",
        "pip_index_url",
        "pip_extra_index_urls",
        "conda_policy",
        "proxy_policy",
    }:
        raise ProfileError("network has missing or unsupported fields")
    for field in ("hf_endpoint", "pip_index_url"):
        if network[field] is not None:
            network[field] = _require_string(
                network[field], f"network.{field}", maximum=500
            ).rstrip("/")
    extras = network["pip_extra_index_urls"]
    if not isinstance(extras, list) or len(extras) > 8:
        raise ProfileError("network.pip_extra_index_urls must be a list of at most 8 URLs")
    network["pip_extra_index_urls"] = [
        _require_string(item, "network.pip_extra_index_urls[]", maximum=500).rstrip("/")
        for item in extras
    ]
    if network["conda_policy"] not in {"direct", "direct-then-proxy", "proxy"}:
        raise ProfileError("network.conda_policy is invalid")
    if network["proxy_policy"] not in {"disabled", "on-demand"}:
        raise ProfileError("network.proxy_policy is invalid")

    git = _require_object(value["git"], "git")
    if set(git) != {"allow_lfs", "allow_submodules", "max_tracked_file_mib"}:
        raise ProfileError("git has missing or unsupported fields")
    if not isinstance(git["allow_lfs"], bool) or not isinstance(
        git["allow_submodules"], bool
    ):
        raise ProfileError("git allow_lfs and allow_submodules must be booleans")
    git["max_tracked_file_mib"] = _require_int(
        git["max_tracked_file_mib"],
        "git.max_tracked_file_mib",
        minimum=1,
        maximum=1024,
    )

    dashboard = _require_object(value["dashboard"], "dashboard")
    if set(dashboard) != {
        "local_port",
        "sample_interval_seconds",
        "tensorboard_remote_port_start",
        "tensorboard_remote_port_end",
    }:
        raise ProfileError("dashboard has missing or unsupported fields")
    dashboard["local_port"] = _require_int(
        dashboard["local_port"], "dashboard.local_port", minimum=1024, maximum=65535
    )
    interval = dashboard["sample_interval_seconds"]
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise ProfileError("dashboard.sample_interval_seconds must be numeric")
    if not 0.5 <= float(interval) <= 60:
        raise ProfileError("dashboard.sample_interval_seconds must be between 0.5 and 60")
    dashboard["sample_interval_seconds"] = float(interval)
    for field in ("tensorboard_remote_port_start", "tensorboard_remote_port_end"):
        dashboard[field] = _require_int(
            dashboard[field], f"dashboard.{field}", minimum=1024, maximum=65535
        )
    if dashboard["tensorboard_remote_port_start"] > dashboard["tensorboard_remote_port_end"]:
        raise ProfileError("TensorBoard port range is reversed")
    if remote["proxy_port"] in range(
        dashboard["tensorboard_remote_port_start"],
        dashboard["tensorboard_remote_port_end"] + 1,
    ):
        raise ProfileError("remote proxy port overlaps the TensorBoard port pool")
    return value


def profile_path(slug: str) -> Path:
    return profiles_dir() / f"{validate_slug(slug)}.json"


def selected_profile_name(explicit: str | None = None) -> str:
    if explicit:
        return validate_slug(explicit)
    env_path = os.environ.get(PROFILE_PATH_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.parent != profiles_dir().resolve() or path.suffix != ".json":
            raise ProfileError(f"{PROFILE_PATH_ENV} must select a file inside {profiles_dir()}")
        return validate_slug(path.stem)
    env_name = os.environ.get(PROFILE_ENV)
    if env_name:
        return validate_slug(env_name)
    try:
        name = active_profile_path().read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ProfileError(
            "no profile selected; run remote-gpu-dev setup or pass --profile"
        ) from exc
    return validate_slug(name)


def load_profile(explicit: str | None = None) -> dict[str, Any]:
    slug = selected_profile_name(explicit)
    path = profile_path(slug)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise ProfileError(f"profile does not exist: {slug}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid JSON in {path}: {exc}") from exc
    profile = validate_profile(raw)
    if profile["slug"] != slug:
        raise ProfileError("profile filename and profile slug differ")
    profile["_path"] = str(path)
    return profile


def public_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if not key.startswith("_")}


def _atomic_write(path: Path, text: str, mode: int) -> None:
    root = config_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_parent = path.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ProfileError("profile path escaped the private configuration root")
    cursor = resolved_parent
    while cursor != root:
        os.chmod(cursor, 0o700)
        cursor = cursor.parent
    os.chmod(root, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _profile_store_lock():
    root = config_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root / ".profiles.lock", flags, 0o600)
    except OSError as exc:
        raise ProfileError("could not open the private profile-store lock") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise ProfileError("the private profile-store lock is not an owned regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _assert_coordination_compatible(normalized: Mapping[str, Any]) -> None:
    for other_slug in list_profiles():
        if other_slug == normalized["slug"]:
            continue
        try:
            with profile_path(other_slug).open("r", encoding="utf-8") as handle:
                other = validate_profile(json.load(handle))
        except (OSError, json.JSONDecodeError, ProfileError) as exc:
            raise ProfileError(
                f"cannot validate existing profile {other_slug}; refusing to "
                "create a potentially conflicting coordination namespace"
            ) from exc
        other_root = other["local"]["ticket_root"]
        selected_root = normalized["local"]["ticket_root"]
        other_coordination = other["trust"]["coordination_uid"]
        selected_coordination = normalized["trust"]["coordination_uid"]
        other_uuids = {device["uuid"] for device in other["gpu"]["devices"]}
        selected_uuids = {
            device["uuid"] for device in normalized["gpu"]["devices"]
        }
        overlap = sorted(other_uuids & selected_uuids)
        if other_coordination == selected_coordination and any(
            other[field] != normalized[field]
            for field in (
                "local",
                "remote",
                "gpu",
                "network",
                "git",
                "dashboard",
            )
        ):
            raise ProfileError(
                f"profile {other_slug} uses the same GPU coordination identity "
                "but has a different operational contract"
            )
        if other_root == selected_root and (
            other_coordination != selected_coordination
            or other["gpu"]["devices"] != normalized["gpu"]["devices"]
        ):
            raise ProfileError(
                f"ticket root is already bound to profile {other_slug} with a "
                "different coordination identity or GPU mapping"
            )
        if overlap and other_root != selected_root:
            raise ProfileError(
                f"profile {other_slug} overlaps managed GPU UUIDs but uses a "
                f"different ticket root: {', '.join(overlap)}"
            )
        if overlap and (
            other_coordination != selected_coordination
            or other["gpu"]["devices"] != normalized["gpu"]["devices"]
        ):
            raise ProfileError(
                f"profile {other_slug} overlaps managed GPU UUIDs without an "
                "identical managed GPU mapping"
            )


def save_profile(profile: Mapping[str, Any], *, replace: bool = False) -> Path:
    normalized = validate_profile(profile)
    path = profile_path(normalized["slug"])
    with _profile_store_lock():
        if path.exists() and not replace:
            raise ProfileError(f"profile already exists: {normalized['slug']}")
        _assert_coordination_compatible(normalized)
        text = (
            json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        _atomic_write(path, text, 0o600)
    return path


def set_active_profile(slug: str) -> Path:
    slug = validate_slug(slug)
    if not profile_path(slug).is_file():
        raise ProfileError(f"profile does not exist: {slug}")
    _atomic_write(active_profile_path(), slug + "\n", 0o600)
    return active_profile_path()


def list_profiles() -> list[str]:
    directory = profiles_dir()
    if not directory.is_dir():
        return []
    return sorted(
        entry.stem
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix == ".json" and SLUG_RE.fullmatch(entry.stem)
    )


def subprocess_environment(slug: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment[PROFILE_ENV] = validate_slug(slug)
    environment[PROFILE_PATH_ENV] = str(profile_path(slug).resolve())
    return environment


def dashboard_runtime_dir(profile: Mapping[str, Any]) -> Path:
    validate_slug(str(profile["slug"]))
    coordination_uid = str(profile["trust"]["coordination_uid"])
    digest = hashlib.sha256(coordination_uid.encode("ascii")).hexdigest()[:16]
    override = os.environ.get("REMOTE_GPU_DEV_STATE_HOME")
    if override:
        base = Path(override).expanduser().resolve()
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        base = (
            Path(xdg).expanduser()
            if xdg
            else Path.home() / ".local" / "state"
        )
        base = base.resolve() / "remote-gpu-dev"
    return base / "servers" / digest / "dashboard"


def ticket_config(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server": f"{profile['ssh']['user']}@{profile['ssh']['host']}:{profile['ssh']['port']}",
        "profile": profile["slug"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "gpu_ids": profile["gpu"]["ids"],
        "gpu_devices": profile["gpu"]["devices"],
        "reservation_ttl_minutes": profile["gpu"]["reservation_ttl_minutes"],
        "heartbeat_grace_minutes": profile["gpu"]["heartbeat_grace_minutes"],
        "recent_terminal_limit": 12,
        "tensorboard_port_start": profile["dashboard"]["tensorboard_remote_port_start"],
        "tensorboard_port_end": profile["dashboard"]["tensorboard_remote_port_end"],
    }


def default_profile(
    *,
    name: str,
    slug: str,
    host: str,
    user: str,
    port: int,
    identity_file: str,
    local_projects_root: str,
    ticket_root: str,
    remote_temp_root: str,
    remote_durable_root: str,
    gpu_ids: list[int],
    conda_executable: str,
    monitor_python: str,
    host_key_fingerprints: list[str],
    remote_machine_id_sha256: str | None,
    gpu_devices: list[dict[str, Any]],
) -> dict[str, Any]:
    durable = remote_durable_root.rstrip("/")
    temporary = remote_temp_root.rstrip("/")
    value = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "slug": slug,
        "created_at": utc_now(),
        "trust": {
            "server_uid": compute_server_uid(
                host_key_fingerprints, remote_machine_id_sha256
            ),
            "coordination_uid": compute_coordination_uid(
                [device["uuid"] for device in gpu_devices]
            ),
            "host_key_fingerprints": sorted(host_key_fingerprints),
            "remote_machine_id_sha256": remote_machine_id_sha256,
        },
        "ssh": {
            "host": host,
            "user": user,
            "port": port,
            "identity_file": str(Path(identity_file).expanduser().resolve()),
            "known_hosts_file": str(known_hosts_path(slug)),
            "connect_timeout_seconds": 12,
            "proxy_jump": None,
        },
        "local": {
            "projects_root": str(Path(local_projects_root).expanduser().resolve()),
            "ticket_root": str(Path(ticket_root).expanduser().resolve()),
            "coordination_scope": "single-controller",
            "proxy_host": "127.0.0.1",
            "proxy_port": 7890,
        },
        "remote": {
            "temp_root": temporary,
            "durable_root": durable,
            "git_bare_root": durable + "/git",
            "projects_root": temporary + "/projects",
            "records_root": durable + "/records",
            "conda_executable": conda_executable,
            "monitor_python": monitor_python,
            "proxy_host": "127.0.0.1",
            "proxy_port": 17890,
        },
        "gpu": {
            "ids": gpu_ids,
            "devices": gpu_devices,
            "reservation_ttl_minutes": 30,
            "heartbeat_grace_minutes": 30,
            "environment": {},
            "mig_mode": "disabled",
        },
        "network": {
            "hf_endpoint": "https://hf-mirror.com",
            "pip_index_url": None,
            "pip_extra_index_urls": [
                "https://pypi.tuna.tsinghua.edu.cn/simple"
            ],
            "conda_policy": "direct-then-proxy",
            "proxy_policy": "on-demand",
        },
        "git": {
            "allow_lfs": False,
            "allow_submodules": False,
            "max_tracked_file_mib": 10,
        },
        "dashboard": {
            "local_port": 8765,
            "sample_interval_seconds": 2.0,
            "tensorboard_remote_port_start": 16006,
            "tensorboard_remote_port_end": 16105,
        },
    }
    return validate_profile(value)
