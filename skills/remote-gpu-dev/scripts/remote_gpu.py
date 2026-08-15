#!/usr/bin/env python3
"""Interactive setup and command router for remote-gpu-dev."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from profile import (
    ProfileError,
    active_profile_path,
    default_profile,
    known_hosts_path,
    list_profiles,
    load_profile,
    profile_path,
    public_profile,
    save_profile,
    selected_profile_name,
    set_active_profile,
    slugify,
    subprocess_environment,
    ticket_config,
    utc_now,
    validate_profile,
)
from ssh_remote import SSHError, ssh_argv
from managed_run import ManagedRunError, build_landlock_command
from remote_path_guard import RemotePathError


SCRIPT_DIR = Path(__file__).resolve().parent
SSH_HELPER = SCRIPT_DIR / "ssh_remote.py"
TICKET_TOOL = SCRIPT_DIR / "gpu_ticket.py"
DASHBOARD_TOOL = SCRIPT_DIR / "remote_gpu_dashboard.py"
SIDECAR_TOOL = SCRIPT_DIR / "tensorboard_sidecar.py"
PROJECT_TOOL = SCRIPT_DIR / "git_project.py"
INFRA_TOOL = SCRIPT_DIR / "infra_tools.py"
GPU_TOOL = SCRIPT_DIR / "gpu_status.py"
RUN_TOOL = SCRIPT_DIR / "managed_run.py"
DELEGATED_TOOLS = {
    "ssh": SSH_HELPER,
    "ticket": TICKET_TOOL,
    "dashboard": DASHBOARD_TOOL,
    "tensorboard": SIDECAR_TOOL,
    "project": PROJECT_TOOL,
    "infra": INFRA_TOOL,
    "gpu": GPU_TOOL,
    "run": RUN_TOOL,
}


class SetupError(RuntimeError):
    pass


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except EOFError as exc:
            raise SetupError("interactive input ended before setup completed") from exc
        if value:
            return value
        if default is not None:
            return default
        print("该项不能为空。", file=sys.stderr)


def confirm(prompt: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        value = ask(f"{prompt} ({marker})", "").lower()
        if not value:
            return default
        if value in {"y", "yes", "是"}:
            return True
        if value in {"n", "no", "否"}:
            return False
        print("请输入 y 或 n。", file=sys.stderr)


def ask_int(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        value = ask(prompt, str(default))
        try:
            result = int(value)
        except ValueError:
            print("请输入整数。", file=sys.stderr)
            continue
        if minimum <= result <= maximum:
            return result
        print(f"请输入 {minimum} 到 {maximum} 之间的整数。", file=sys.stderr)


def _atomic_write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _ssh_base(
    *,
    host: str,
    user: str,
    port: int,
    identity: Path,
    known_hosts: Path,
    proxy_jump: str | None,
) -> list[str]:
    if not identity.is_file():
        raise SetupError(f"私钥不存在：{identity}")
    mode = stat.S_IMODE(identity.stat().st_mode)
    if mode & 0o077:
        raise SetupError(f"私钥权限过宽 ({mode:o})；请执行 chmod 600 {identity}")
    argv = [
        "ssh",
        "-p",
        str(port),
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "ForwardX11Trusted=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "VerifyHostKeyDNS=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "RemoteCommand=none",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=12",
    ]
    if proxy_jump:
        argv.extend(["-J", proxy_jump])
    argv.append(f"{user}@{host}")
    return argv


def _classify_ssh_failure(stderr: str) -> str:
    lowered = stderr.lower()
    if "permission denied" in lowered or "publickey" in lowered:
        return "authentication"
    if "host key verification failed" in lowered or "remote host identification has changed" in lowered:
        return "host-key"
    if "could not resolve hostname" in lowered or "name or service not known" in lowered:
        return "dns"
    if "connection timed out" in lowered or "operation timed out" in lowered:
        return "timeout"
    if "connection refused" in lowered:
        return "refused"
    if "no route to host" in lowered or "network is unreachable" in lowered:
        return "network"
    return "other"


def _run_bootstrap_ssh(
    *,
    host: str,
    user: str,
    port: int,
    identity: Path,
    known_hosts: Path,
    proxy_jump: str | None,
    command: str,
    timeout: float = 25,
) -> subprocess.CompletedProcess[str]:
    argv = _ssh_base(
        host=host,
        user=user,
        port=port,
        identity=identity,
        known_hosts=known_hosts,
        proxy_jump=proxy_jump,
    )
    argv.append(command)
    return subprocess.run(
        argv,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _fingerprints_for_host_keys(key_text: str) -> list[str]:
    fingerprint = subprocess.run(
        ["ssh-keygen", "-l", "-E", "sha256", "-f", "-"],
        input=key_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if fingerprint.returncode != 0:
        raise SetupError("ssh-keygen 无法计算 host-key 指纹")
    values: list[str] = []
    for line in fingerprint.stdout.splitlines():
        match = re.search(r"\b(SHA256:[A-Za-z0-9+/]{20,60})\b", line)
        if match:
            values.append(match.group(1))
    if not values:
        raise SetupError("没有解析到 SSH host-key 指纹")
    return sorted(set(values))


def _route_requires_openssh(host: str, port: int, proxy_jump: str | None) -> bool:
    """Return whether candidate keys must be acquired through the OpenSSH route."""

    argv = ["ssh", "-G", "-p", str(port)]
    if proxy_jump:
        argv.extend(["-J", proxy_jump])
    argv.append(host)
    try:
        rendered = subprocess.run(
            argv,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"无法解析 SSH 路由：{exc}") from exc
    if rendered.returncode != 0:
        detail = " ".join(rendered.stderr.split())[:300]
        raise SetupError(f"无法解析 SSH 路由：{detail or 'ssh -G failed'}")
    effective: dict[str, str] = {}
    for line in rendered.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            effective.setdefault(key.lower(), value.strip())
    configured_host = host.strip("[]").lower()
    effective_host = effective.get("hostname", configured_host).strip("[]").lower()
    effective_proxy_jump = effective.get("proxyjump", "none").lower()
    effective_proxy_command = effective.get("proxycommand", "none").lower()
    return bool(
        proxy_jump
        or effective_host != configured_host
        or effective_proxy_jump not in {"", "none"}
        or effective_proxy_command not in {"", "none"}
    )


def _scan_host_keys_direct(host: str, port: int) -> str:
    try:
        scan = subprocess.run(
            ["ssh-keyscan", "-T", "8", "-p", str(port), host],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"无法运行 ssh-keyscan：{exc}") from exc
    lines = [
        line
        for line in scan.stdout.splitlines()
        if line and not line.startswith("#")
    ]
    if scan.returncode != 0 or not lines:
        detail = " ".join(scan.stderr.split())[:300]
        raise SetupError(f"无法取得 SSH host key：{detail or 'no key returned'}")
    return "\n".join(lines) + "\n"


def _scan_host_keys_via_openssh(
    host: str, user: str, port: int, proxy_jump: str | None
) -> str:
    """Acquire untrusted candidate keys through aliases/ProxyJump without logging in."""

    descriptor, temporary_name = tempfile.mkstemp(prefix="remote-gpu-candidate-hosts-")
    os.close(descriptor)
    candidate_path = Path(temporary_name)
    os.chmod(candidate_path, 0o600)
    try:
        argv = [
            "ssh",
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "PubkeyAuthentication=no",
            "-o",
            "PreferredAuthentications=none",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            "-o",
            "ForwardX11Trusted=no",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={candidate_path}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "VerifyHostKeyDNS=no",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ControlPersist=no",
            "-o",
            "RemoteCommand=none",
            "-o",
            "HashKnownHosts=no",
            "-o",
            "ConnectTimeout=12",
        ]
        if proxy_jump:
            argv.extend(["-J", proxy_jump])
        argv.extend([f"{user}@{host}", "exit 99"])
        try:
            completed = subprocess.run(
                argv,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SetupError(f"无法通过 SSH 路由取得 host key：{exc}") from exc
        lines = [
            line
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        if not lines:
            detail = " ".join(completed.stderr.split())[:300]
            raise SetupError(
                "无法通过 SSH alias/ProxyJump 取得目标 host key："
                + (detail or "no key returned")
            )
        return "\n".join(lines) + "\n"
    finally:
        candidate_path.unlink(missing_ok=True)


def _scan_host_keys(
    host: str, user: str, port: int, proxy_jump: str | None
) -> tuple[str, list[str]]:
    if _route_requires_openssh(host, port, proxy_jump):
        key_text = _scan_host_keys_via_openssh(host, user, port, proxy_jump)
    else:
        key_text = _scan_host_keys_direct(host, port)
    return key_text, _fingerprints_for_host_keys(key_text)


def _guide_public_key(
    identity: Path, user: str, host: str, port: int, proxy_jump: str | None
) -> None:
    public = Path(str(identity) + ".pub")
    print("\n公钥认证尚未成功。此工具不会读取或保存服务器密码。")
    if public.is_file():
        print(f"已有公钥：{public}")
    else:
        print(f"未找到 {public}。可在另一个终端生成专用密钥：")
        print(f"  ssh-keygen -t ed25519 -f {shlex.quote(str(identity))} -C remote-gpu-dev")
    print("把公钥交给管理员，或在你自己的终端运行交互式命令：")
    copy_argv = ["ssh-copy-id", "-i", str(public), "-p", str(port)]
    if proxy_jump:
        copy_argv.extend(["-o", f"ProxyJump={proxy_jump}"])
    copy_argv.append(f"{user}@{host}")
    print("  " + shlex.join(copy_argv))
    print("完成后回到这里重试；密码或私钥内容不要粘贴到 Codex。\n")


PROBE_COMMAND = r'''set -eu
printf 'IDENTITY\t%s\t%s\t%s\t%s\n' "$(id -un)" "$(hostname)" "$HOME" "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || printf unavailable)"
if test -r /etc/machine-id; then printf 'MACHINE\t%s\n' "$(sha256sum /etc/machine-id | awk '{print $1}')"; else printf 'MACHINE\tunavailable\n'; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'NVIDIA\tyes\n'
  nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits | sed 's/^/GPU\t/'
  if mig_output="$(nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader,nounits 2>/dev/null)"; then
    printf '%s\n' "$mig_output" | sed 's/^/MIG\t/'
  else
    printf 'MIG_ERROR\tquery-failed\n'
  fi
else printf 'NVIDIA\tno\n'; fi
for tool in git tmux flock scontrol qstat; do if command -v "$tool" >/dev/null 2>&1; then printf 'TOOL\t%s\t%s\n' "$tool" "$(command -v "$tool")"; else printf 'TOOL\t%s\tmissing\n' "$tool"; fi; done
for candidate in "$(command -v conda 2>/dev/null || true)" /opt/conda/bin/conda; do if test -n "$candidate" && test -x "$candidate"; then printf 'CONDA\t%s\n' "$candidate"; break; fi; done
'''


REMOTE_CREATE_MANAGED_ROOTS = r'''
import os, pathlib, sys

def normalized(raw):
    path = pathlib.PurePosixPath(raw)
    if not path.is_absolute() or path == pathlib.PurePosixPath("/") or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise RuntimeError("managed root is not a normalized absolute path")
    return path

def create_root(raw):
    path = normalized(raw)
    parent = pathlib.Path(str(path.parent))
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent or not parent.is_dir():
        raise RuntimeError("managed root parent must already exist and be canonical")
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=descriptor)
        except FileExistsError:
            pass
        child = os.open(
            path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        os.close(child)
    finally:
        os.close(descriptor)

def create_child(root_raw, child_raw):
    root = pathlib.Path(root_raw)
    child = pathlib.Path(child_raw)
    resolved_root = root.resolve(strict=True)
    if resolved_root != root or not root.is_dir():
        raise RuntimeError("managed root is not canonical")
    try:
        relative = child.relative_to(root)
    except ValueError:
        raise RuntimeError("managed child is outside its root")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts:
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    finally:
        os.close(descriptor)

temporary, durable, *children = sys.argv[1:]
create_root(temporary)
create_root(durable)
for child in children:
    root = temporary if pathlib.Path(child).is_relative_to(temporary) else durable
    create_child(root, child)
'''


def _parse_probe(stdout: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "identity": None,
        "machine_hash": None,
        "nvidia": False,
        "gpus": [],
        "mig": {},
        "mig_error": None,
        "tools": {},
        "conda": None,
    }
    for line in stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        if parts[0] == "IDENTITY" and len(parts) == 5:
            result["identity"] = {
                "user": parts[1],
                "hostname": parts[2],
                "home": parts[3],
                "boot_id": parts[4],
            }
        elif parts[0] == "MACHINE" and len(parts) == 2 and parts[1] != "unavailable":
            result["machine_hash"] = "sha256:" + parts[1]
        elif parts[0] == "NVIDIA" and len(parts) == 2:
            result["nvidia"] = parts[1] == "yes"
        elif parts[0] == "GPU" and len(parts) >= 2:
            fields = [field.strip() for field in parts[1].split(",", 3)]
            if len(fields) == 4 and fields[0].isdigit():
                memory = re.sub(r"\s*MiB\s*\Z", "", fields[3])
                if memory.isdigit():
                    result["gpus"].append(
                        {
                            "index": int(fields[0]),
                            "uuid": fields[1],
                            "name": fields[2],
                            "memory_mib": int(memory),
                        }
                    )
        elif parts[0] == "MIG" and len(parts) >= 2:
            fields = [field.strip().lower() for field in parts[1].split(",", 1)]
            if len(fields) == 2 and fields[0].isdigit():
                result["mig"][int(fields[0])] = fields[1]
        elif parts[0] == "MIG_ERROR" and len(parts) == 2:
            result["mig_error"] = parts[1]
        elif parts[0] == "TOOL" and len(parts) == 3:
            result["tools"][parts[1]] = parts[2]
        elif parts[0] == "CONDA" and len(parts) == 2:
            result["conda"] = parts[1]
    if not result["identity"]:
        raise SetupError("SSH 已连接，但无法解析远端身份探针")
    if not result["nvidia"] or not result["gpus"]:
        raise SetupError("远端没有可用的 nvidia-smi GPU inventory")
    return result


def _normalize_mig_mode(value: str) -> str | None:
    compact = re.sub(r"[\s_-]+", "", value.strip().lower().strip("[]"))
    if compact == "disabled":
        return "disabled"
    if compact in {"na", "n/a", "notsupported", "unsupported"}:
        return "unsupported"
    if compact == "enabled":
        return "enabled"
    return None


def _mig_readiness(
    probe: dict[str, Any], expected_policy: str | None = None
) -> tuple[bool, str, str | None]:
    if probe.get("mig_error"):
        return False, f"MIG mode query failed: {probe['mig_error']}", None
    gpu_indices = {device["index"] for device in probe["gpus"]}
    observed_indices = set(probe["mig"])
    if observed_indices != gpu_indices:
        missing = sorted(gpu_indices - observed_indices)
        unexpected = sorted(observed_indices - gpu_indices)
        return (
            False,
            f"MIG mode inventory mismatch missing={missing} unexpected={unexpected}",
            None,
        )
    normalized = {
        index: _normalize_mig_mode(mode) for index, mode in probe["mig"].items()
    }
    unknown = {
        index: mode
        for index, mode in normalized.items()
        if mode is None
    }
    if unknown:
        return False, f"unrecognized MIG modes: {probe['mig']}", None
    modes = set(normalized.values())
    if "enabled" in modes:
        return False, f"MIG is enabled on one or more GPUs: {normalized}", None
    if len(modes) != 1:
        return False, f"mixed MIG policy across physical GPUs: {normalized}", None
    observed_policy = next(iter(modes))
    if observed_policy not in {"disabled", "unsupported"}:
        return False, f"unsafe MIG policy: {observed_policy}", None
    if expected_policy is not None and observed_policy != expected_policy:
        return (
            False,
            f"MIG policy changed: expected={expected_policy} observed={observed_policy}",
            observed_policy,
        )
    return True, f"MIG policy={observed_policy} on every physical GPU", observed_policy


def _exact_managed_gpu_mapping(
    expected_devices: list[dict[str, Any]], actual_devices: list[dict[str, Any]]
) -> tuple[bool, str]:
    expected = {device["index"]: device["uuid"] for device in expected_devices}
    actual = {device["index"]: device["uuid"] for device in actual_devices}
    mismatches = {
        index: {"expected": uuid, "actual": actual.get(index)}
        for index, uuid in expected.items()
        if actual.get(index) != uuid
    }
    if mismatches:
        return False, f"managed index-to-UUID mismatch: {mismatches}"
    return True, f"exact managed index-to-UUID mapping present: {len(expected)}"


def _parse_env_assignments(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value.strip():
        return result
    for assignment in value.split(","):
        if "=" not in assignment:
            raise SetupError("环境变量必须是 NAME=value，用逗号分隔")
        name, item = assignment.split("=", 1)
        name = name.strip()
        item = item.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,63}", name):
            raise SetupError(f"无效的环境变量名：{name}")
        result[name] = item
    return result


def _require_no_external_scheduler(probe: dict[str, Any]) -> None:
    detected = [
        tool for tool in ("scontrol", "qstat") if probe["tools"].get(tool) != "missing"
    ]
    if detected:
        raise SetupError(
            "检测到现有集群调度器（"
            + ", ".join(detected)
            + "）。不要叠加直接 GPU 文件工单；请使用该调度器的队列。"
        )
    declared = ask(
        "10/20 现有 GPU 调度器（none / slurm / pbs / kubernetes / cloud / other）",
        "none",
    ).strip().lower()
    if declared not in {"none", "no", "无"}:
        raise SetupError(
            f"用户声明存在外部调度器 ({declared})；停止创建并行文件工单，请使用现有调度器"
        )


def _validate_dashboard_port_plan(
    *,
    dashboard_port: int,
    proxy_enabled: bool,
    proxy_local_port: int,
    proxy_remote_port: int,
    tensorboard_port_start: int,
    tensorboard_port_end: int,
) -> None:
    if tensorboard_port_start > tensorboard_port_end:
        raise SetupError("TensorBoard 端口池起始端口不能大于结束端口")
    tensorboard_ports = range(tensorboard_port_start, tensorboard_port_end + 1)
    if proxy_remote_port in tensorboard_ports:
        raise SetupError("远端代理端口不能落入 TensorBoard 远端端口池")
    if proxy_enabled and dashboard_port == proxy_local_port:
        raise SetupError("本地 dashboard 端口不能与本地代理端口相同")


def _validate_profile_port_plan(profile: dict[str, Any]) -> None:
    _validate_dashboard_port_plan(
        dashboard_port=profile["dashboard"]["local_port"],
        proxy_enabled=profile["network"]["proxy_policy"] == "on-demand",
        proxy_local_port=profile["local"]["proxy_port"],
        proxy_remote_port=profile["remote"]["proxy_port"],
        tensorboard_port_start=profile["dashboard"]["tensorboard_remote_port_start"],
        tensorboard_port_end=profile["dashboard"]["tensorboard_remote_port_end"],
    )


def _ensure_not_in_git(path: Path) -> None:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    completed = subprocess.run(
        ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode == 0:
        worktree = Path(completed.stdout.strip()).resolve()
        resolved = path.resolve(strict=False)
        if resolved == worktree or worktree in resolved.parents:
            raise SetupError("工单目录不能位于 Git worktree 内")


def _write_ticket_config(profile: dict[str, Any]) -> None:
    root = Path(profile["local"]["ticket_root"])
    _ensure_not_in_git(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    lock_path = root / ".setup.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        config_path = root / "config.json"
        candidate = ticket_config(profile)
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SetupError(f"现有工单配置不可读：{exc}") from exc
            coordination_fields = (
                "schema_version",
                "coordination_uid",
                "gpu_ids",
                "gpu_devices",
                "reservation_ttl_minutes",
                "heartbeat_grace_minutes",
                "recent_terminal_limit",
                "tensorboard_port_start",
                "tensorboard_port_end",
            )
            mismatches = [
                field
                for field in coordination_fields
                if existing.get(field) != candidate.get(field)
            ]
            if mismatches:
                raise SetupError(
                    "现有工单目录与该 profile 的协调契约不一致："
                    + ", ".join(mismatches)
                )
            return
        content = json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write(config_path, content, 0o600)
    finally:
        os.close(descriptor)


def _check_duplicate_server(profile: dict[str, Any]) -> None:
    for other_name in list_profiles():
        if other_name == profile["slug"]:
            continue
        try:
            other = load_profile(other_name)
        except ProfileError:
            continue
        other_uuids = {device["uuid"] for device in other["gpu"]["devices"]}
        selected_uuids = {device["uuid"] for device in profile["gpu"]["devices"]}
        overlap = sorted(other_uuids & selected_uuids)
        same_root = other["local"]["ticket_root"] == profile["local"]["ticket_root"]
        exact_contract = (
            other["trust"]["coordination_uid"]
            == profile["trust"]["coordination_uid"]
            and other["gpu"]["devices"] == profile["gpu"]["devices"]
        )
        if same_root and not exact_contract:
            raise SetupError(
                f"工单目录已由 profile {other_name} 绑定到不同的 GPU 协调身份或映射"
            )
        if overlap and not same_root:
            raise SetupError(
                f"profile {other_name} 与当前 profile 重叠 GPU UUID 但使用不同工单目录："
                + ", ".join(overlap)
            )
        if overlap and not exact_contract:
            raise SetupError(
                f"profile {other_name} 与当前 profile 的 GPU UUID 有重叠但映射并不完全一致"
            )


def _setup_from_file(path: Path, *, replace: bool, offline: bool) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"无法读取 profile JSON：{exc}") from exc
    profile = validate_profile(raw)
    _validate_profile_port_plan(profile)
    save_profile(profile, replace=replace)
    _write_ticket_config(profile)
    set_active_profile(profile["slug"])
    if not offline:
        report = run_doctor(profile)
        if any(item["status"] == "FAIL" for item in report["checks"]):
            raise SetupError("profile 已保存，但 doctor 失败；请修复后重新运行 doctor")
    return profile


def setup_interactive(*, replace: bool) -> dict[str, Any]:
    print("Remote GPU Dev 交互式初始化。每一步只保存非秘密配置。\n")
    name = ask("1/20 服务器显示名称", "GPU Server")
    slug = ask("配置短名称", slugify(name))
    if profile_path(slug).exists() and not replace:
        raise SetupError(
            f"profile 已存在：{slug}；如需明确覆盖，请重新运行 setup --replace"
        )
    host = ask("2/20 SSH 主机名、IP 或 SSH config 可解析地址")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    user = ask("SSH 用户名", "root")
    port = ask_int("SSH 端口", 22, 1, 65535)
    proxy_jump_value = ask("可选 ProxyJump（无则留空）", "")
    proxy_jump = proxy_jump_value or None
    identity = Path(ask("SSH 私钥绝对路径", str(Path.home() / ".ssh" / "id_ed25519"))).expanduser().resolve()

    print("\n3/20 正在通过配置的 SSH 路由取得候选 host key；候选值不等于身份验证。")
    key_text, fingerprints = _scan_host_keys(host, user, port, proxy_jump)
    print("候选 host-key 指纹：")
    for item in fingerprints:
        print(f"  {item}")
    if not confirm("你是否已通过服务器控制台/管理员等可信渠道核对这些指纹", False):
        raise SetupError("未确认 host-key 指纹，停止配置")
    final_known_hosts = known_hosts_path(slug)
    temporary_known_hosts_handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="remote-gpu-known-hosts-", delete=False
    )
    temporary_known_hosts = Path(temporary_known_hosts_handle.name)
    temporary_known_hosts_handle.write(key_text)
    temporary_known_hosts_handle.flush()
    os.fsync(temporary_known_hosts_handle.fileno())
    temporary_known_hosts_handle.close()
    os.chmod(temporary_known_hosts, 0o600)
    atexit.register(temporary_known_hosts.unlink, missing_ok=True)
    known_hosts = temporary_known_hosts

    print("\n正在执行严格 key-only SSH 测试……")
    while not identity.is_file():
        _guide_public_key(identity, user, host, port, proxy_jump)
        if not confirm("密钥已生成或已有其他密钥，是否继续", True):
            raise SetupError("本地 SSH 私钥尚未准备好")
        identity = Path(ask("SSH 私钥绝对路径", str(identity))).expanduser().resolve()
    while True:
        try:
            connected = _run_bootstrap_ssh(
                host=host,
                user=user,
                port=port,
                identity=identity,
                known_hosts=known_hosts,
                proxy_jump=proxy_jump,
                command=PROBE_COMMAND,
                timeout=35,
            )
        except subprocess.TimeoutExpired:
            raise SetupError("SSH 探针超时；先检查网络、VPN、跳板机和安全组")
        if connected.returncode == 0:
            break
        kind = _classify_ssh_failure(connected.stderr)
        detail = " ".join(connected.stderr.split())[:500]
        if kind != "authentication":
            guidance = {
                "dns": "检查主机名或 DNS。",
                "timeout": "检查 VPN、路由、安全组和端口。",
                "refused": "远端端口没有 SSH 服务，或端口填写错误。",
                "network": "当前网络无法到达服务器。",
                "host-key": "host key 与已确认值不一致；不要自动删除记录，请重新核对。",
            }.get(kind, "检查 SSH 配置、跳板机和远端 shell。")
            raise SetupError(f"SSH 连接失败 ({kind})：{detail}\n{guidance}")
        print(f"4/20 SSH 公钥认证失败：{detail}")
        _guide_public_key(identity, user, host, port, proxy_jump)
        if not confirm("公钥已经安装，是否重试", True):
            raise SetupError("公钥尚未配置，setup 暂停")

    probe = _parse_probe(connected.stdout)
    identity_info = probe["identity"]
    print(
        f"连接成功：user={identity_info['user']} host={identity_info['hostname']} "
        f"home={identity_info['home']}，发现 {len(probe['gpus'])} 张 GPU。"
    )
    mig_ready, mig_detail, mig_policy = _mig_readiness(probe)
    if not mig_ready:
        raise SetupError(mig_detail)
    missing_tools = [tool for tool in ("git", "tmux", "flock") if probe["tools"].get(tool) == "missing"]
    if missing_tools:
        raise SetupError("远端缺少必需工具：" + ", ".join(missing_tools))
    if not probe["conda"]:
        raise SetupError(
            "未找到 Conda。请先安装 Miniforge/Miniconda，再重新运行 setup；本工具不会静默安装。"
        )

    default_local_projects = str(Path.home() / "gpu-projects" / slug)
    local_projects = Path(ask("5/20 本地多项目容器目录", default_local_projects)).expanduser().resolve()
    default_ticket = str(Path.home() / ".local" / "state" / "remote-gpu-dev" / slug / "tickets")
    ticket_root = Path(ask("6/20 本地共享工单目录", default_ticket)).expanduser().resolve()
    _ensure_not_in_git(ticket_root)
    remote_home = identity_info["home"]
    scratch_base = ask("7/20 远端临时/高速存储基目录", remote_home)
    durable_base = ask("8/20 远端持久化存储基目录", remote_home)
    if not confirm("你确认该持久化目录在预期的重启/实例生命周期后仍会保留数据", True):
        raise SetupError("请先确认真正的持久化存储位置，再继续配置")
    remote_temp = scratch_base.rstrip("/") + f"/remote-gpu-dev/{slug}"
    remote_durable = durable_base.rstrip("/") + f"/remote-gpu-dev/{slug}"
    print(f"将只管理临时子目录：{remote_temp}")
    print(f"将只管理持久化子目录：{remote_durable}")

    controllers = ask("9/20 协调范围：single（本机多会话）或 shared（多工作站共享文件系统）", "single")
    if controllers not in {"single", "shared"}:
        raise SetupError("协调范围只能是 single 或 shared")
    if controllers == "shared":
        print("警告：ticket_root 必须位于所有控制工作站共享、且支持 flock/原子 rename 的文件系统。")
        if not confirm("确认该工单目录满足这一条件", False):
            raise SetupError("多工作站模式需要真正共享的锁文件系统")

    _require_no_external_scheduler(probe)

    print("11/20 远端 GPU：")
    for device in probe["gpus"]:
        print(
            f"  {device['index']}: {device['name']} {device['memory_mib']} MiB {device['uuid']}"
        )
    default_ids = ",".join(str(device["index"]) for device in probe["gpus"])
    managed_text = ask("由工单系统管理的 GPU 索引（逗号分隔）", default_ids)
    try:
        managed_ids = sorted({int(item.strip()) for item in managed_text.split(",") if item.strip()})
    except ValueError as exc:
        raise SetupError("GPU 索引必须是逗号分隔的整数") from exc
    available = {device["index"] for device in probe["gpus"]}
    if not managed_ids or not set(managed_ids).issubset(available):
        raise SetupError("选择了不存在的 GPU 索引")
    managed_devices = [device for device in probe["gpus"] if device["index"] in managed_ids]

    conda = probe["conda"]
    monitor_prefix = remote_durable + "/infra/monitor-env"
    monitor_python = monitor_prefix + "/bin/python"
    print(f"12/20 使用 Conda：{conda}")
    preset = ask("13/20 网络预设：china / global / custom", "china")
    if preset not in {"china", "global", "custom"}:
        raise SetupError("网络预设必须是 china、global 或 custom")
    proxy_enabled = confirm("14/20 是否允许按命令临时建立本机代理反向转发", preset == "china")
    proxy_local_port = ask_int("本机代理端口", 7890, 1, 65535) if proxy_enabled else 7890
    proxy_remote_port = ask_int("远端 loopback 代理端口", 17890, 1024, 65535) if proxy_enabled else 17890
    env_text = ask("15/20 服务器专属多卡环境变量（NAME=value，逗号分隔；无则留空）", "")
    gpu_environment = _parse_env_assignments(env_text)
    print(
        f"16/20 Git 将使用 {remote_durable}/git/<project>.git + "
        f"{remote_temp}/projects/<project>；只同步 tracked source。"
    )
    if not confirm("使用此 Git 布局", True):
        raise SetupError("当前版本只支持每项目独立 bare + execution clone 布局")
    dashboard_port = ask_int("17/20 本地看板端口", 8765, 1024, 65535)
    tensorboard_port_start = ask_int(
        "18/20 TensorBoard 远端端口池起始端口", 16006, 1024, 65535
    )
    tensorboard_port_end = ask_int(
        "TensorBoard 远端端口池结束端口", 16105, 1024, 65535
    )
    _validate_dashboard_port_plan(
        dashboard_port=dashboard_port,
        proxy_enabled=proxy_enabled,
        proxy_local_port=proxy_local_port,
        proxy_remote_port=proxy_remote_port,
        tensorboard_port_start=tensorboard_port_start,
        tensorboard_port_end=tensorboard_port_end,
    )
    install_monitor = confirm(
        f"是否在专用 Conda 环境 {monitor_prefix} 安装 nvitop（不会修改 base）",
        True,
    )
    ttl = ask_int("19/20 工单预留 TTL（分钟）", 30, 1, 1440)
    heartbeat = ask_int("运行工单 heartbeat 宽限（分钟）", 30, 1, 1440)

    machine_hash = probe["machine_hash"]
    profile = default_profile(
        name=name,
        slug=slug,
        host=host,
        user=user,
        port=port,
        identity_file=str(identity),
        local_projects_root=str(local_projects),
        ticket_root=str(ticket_root),
        remote_temp_root=remote_temp,
        remote_durable_root=remote_durable,
        gpu_ids=managed_ids,
        conda_executable=conda,
        monitor_python=monitor_python,
        host_key_fingerprints=fingerprints,
        remote_machine_id_sha256=machine_hash,
        gpu_devices=managed_devices,
    )
    profile["ssh"]["proxy_jump"] = proxy_jump
    profile["local"]["coordination_scope"] = (
        "single-controller" if controllers == "single" else "shared-filesystem"
    )
    profile["local"]["proxy_port"] = proxy_local_port
    profile["remote"]["proxy_port"] = proxy_remote_port
    profile["gpu"]["environment"] = gpu_environment
    profile["gpu"]["mig_mode"] = mig_policy
    profile["gpu"]["reservation_ttl_minutes"] = ttl
    profile["gpu"]["heartbeat_grace_minutes"] = heartbeat
    profile["dashboard"]["local_port"] = dashboard_port
    profile["dashboard"]["tensorboard_remote_port_start"] = tensorboard_port_start
    profile["dashboard"]["tensorboard_remote_port_end"] = tensorboard_port_end
    if preset == "global":
        profile["network"]["hf_endpoint"] = None
        profile["network"]["pip_extra_index_urls"] = []
        profile["network"]["conda_policy"] = "direct"
    elif preset == "custom":
        hf_value = ask("自定义 HF endpoint（无则留空）", "")
        pip_primary = ask("自定义 PyPI primary index（无则使用 pip 默认）", "")
        pip_extra = ask("另一个 PyPI extra index（无则留空）", "")
        profile["network"]["hf_endpoint"] = hf_value or None
        profile["network"]["pip_index_url"] = pip_primary or None
        tuna = "https://pypi.tuna.tsinghua.edu.cn/simple"
        profile["network"]["pip_extra_index_urls"] = list(dict.fromkeys(
            value
            for value in (tuna, pip_extra)
            if value and value != pip_primary
        ))
    profile["network"]["proxy_policy"] = "on-demand" if proxy_enabled else "disabled"
    profile = validate_profile(profile)
    _validate_profile_port_plan(profile)
    _check_duplicate_server(profile)

    print("\n20/20 脱敏配置摘要：")
    summary = {
        "profile": profile["slug"],
        "server": f"{user}@{host}:{port}",
        "identity_path": str(identity),
        "server_uid": profile["trust"]["server_uid"],
        "ssh_trust_uid": profile["trust"]["server_uid"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "gpu_indices": managed_ids,
        "gpu_uuids": [item["uuid"] for item in managed_devices],
        "local_projects_root": str(local_projects),
        "ticket_root": str(ticket_root),
        "remote_temp_root": remote_temp,
        "remote_durable_root": remote_durable,
        "conda": conda,
        "network": profile["network"],
        "gpu_environment": gpu_environment,
        "dashboard": profile["dashboard"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not confirm("确认写入 profile，并创建这些专用目录", True):
        raise SetupError("用户取消 setup")
    mkdir_command = shlex.join(
        [
            "/usr/bin/python3",
            "-c",
            REMOTE_CREATE_MANAGED_ROOTS,
            profile["remote"]["temp_root"],
            profile["remote"]["durable_root"],
            profile["remote"]["git_bare_root"],
            profile["remote"]["projects_root"],
            profile["remote"]["records_root"],
        ]
    )
    created = _run_bootstrap_ssh(
        host=host,
        user=user,
        port=port,
        identity=identity,
        known_hosts=known_hosts,
        proxy_jump=proxy_jump,
        command=mkdir_command,
        timeout=30,
    )
    if created.returncode != 0:
        raise SetupError("远端专用目录创建失败：" + " ".join(created.stderr.split())[:400])
    local_projects.mkdir(parents=True, exist_ok=True)
    _atomic_write(final_known_hosts, key_text, 0o600)
    save_profile(profile, replace=replace)
    _write_ticket_config(profile)
    set_active_profile(slug)
    temporary_known_hosts.unlink(missing_ok=True)
    profile["_install_monitor_requested"] = install_monitor
    return profile


def _run_profile_command(profile: dict[str, Any], command: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    argv = ssh_argv(profile, batch=True)
    argv.extend([f"{profile['ssh']['user']}@{profile['ssh']['host']}", command])
    return subprocess.run(
        argv,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_doctor(profile: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail[:500]})

    identity = Path(profile["ssh"]["identity_file"])
    if identity.is_file() and stat.S_IMODE(identity.stat().st_mode) & 0o077 == 0:
        record("ssh_identity", "PASS", f"readable mode={stat.S_IMODE(identity.stat().st_mode):o}")
    else:
        record("ssh_identity", "FAIL", "missing or permissions are broader than 0600")
    known = Path(profile["ssh"]["known_hosts_file"])
    if known.is_file():
        fingerprints = subprocess.run(
            ["ssh-keygen", "-l", "-E", "sha256", "-f", str(known)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        actual_fingerprints = sorted(
            set(
                match.group(1)
                for match in re.finditer(
                    r"\b(SHA256:[A-Za-z0-9+/]{20,60})\b",
                    fingerprints.stdout,
                )
            )
        )
        expected_fingerprints = profile["trust"]["host_key_fingerprints"]
        record(
            "known_hosts",
            "PASS"
            if fingerprints.returncode == 0 and actual_fingerprints == expected_fingerprints
            else "FAIL",
            f"fingerprints={actual_fingerprints}",
        )
    else:
        record("known_hosts", "FAIL", str(known))
    try:
        probe_run = _run_profile_command(profile, PROBE_COMMAND, timeout=40)
    except subprocess.TimeoutExpired:
        record("ssh", "FAIL", "connection timed out")
        probe_run = None
    if probe_run is not None and probe_run.returncode == 0:
        record("ssh", "PASS", "strict key-only connection succeeded")
        try:
            probe = _parse_probe(probe_run.stdout)
        except SetupError as exc:
            record("inventory", "FAIL", str(exc))
        else:
            expected_machine = profile["trust"]["remote_machine_id_sha256"]
            actual_machine = probe["machine_hash"]
            if expected_machine and actual_machine != expected_machine:
                record("server_identity", "FAIL", "remote machine-id hash changed")
            else:
                record("server_identity", "PASS", profile["trust"]["server_uid"])
            mapping_ready, mapping_detail = _exact_managed_gpu_mapping(
                profile["gpu"]["devices"], probe["gpus"]
            )
            record("gpu_inventory", "PASS" if mapping_ready else "FAIL", mapping_detail)
            mig_ready, mig_detail, _mig_policy = _mig_readiness(
                probe, profile["gpu"]["mig_mode"]
            )
            record("mig_mode", "PASS" if mig_ready else "FAIL", mig_detail)
            for tool in ("git", "tmux", "flock"):
                record(
                    f"remote_{tool}",
                    "PASS" if probe["tools"].get(tool) != "missing" else "FAIL",
                    str(probe["tools"].get(tool)),
                )
            schedulers = [
                tool
                for tool in ("scontrol", "qstat")
                if probe["tools"].get(tool) != "missing"
            ]
            record(
                "external_scheduler",
                "FAIL" if schedulers else "PASS",
                ",".join(schedulers) if schedulers else "none detected",
            )
            record(
                "conda",
                "PASS" if probe["conda"] == profile["remote"]["conda_executable"] else "FAIL",
                str(probe["conda"]),
            )
    elif probe_run is not None:
        record("ssh", "FAIL", _classify_ssh_failure(probe_run.stderr))

    paths_command = " && ".join(
        f"test -d {shlex.quote(profile['remote'][field])} && test -w {shlex.quote(profile['remote'][field])}"
        for field in ("temp_root", "durable_root", "git_bare_root", "projects_root", "records_root")
    )
    try:
        paths = _run_profile_command(profile, paths_command, timeout=25)
        record("remote_paths", "PASS" if paths.returncode == 0 else "FAIL", "managed roots writable")
    except subprocess.TimeoutExpired:
        record("remote_paths", "FAIL", "path probe timed out")
    monitor_command = shlex.join(
        [
            profile["remote"]["monitor_python"],
            "-c",
            "import nvitop; print(nvitop.__version__)",
        ]
    )
    try:
        monitor_command = build_landlock_command(
            profile,
            [
                profile["remote"]["monitor_python"],
                "-c",
                "import nvitop; print(nvitop.__version__)",
            ],
            workdir=profile["remote"]["temp_root"],
            environment={"CUDA_VISIBLE_DEVICES": ""},
            device_ids=profile["gpu"]["ids"],
        )
    except (RemotePathError, ManagedRunError) as exc:
        record("nvitop", "FAIL", str(exc))
        monitor_command = "false"
    try:
        monitor = _run_profile_command(profile, monitor_command, timeout=25)
        record(
            "nvitop",
            "PASS" if monitor.returncode == 0 else "WARN",
            monitor.stdout.strip() or "not installed in monitor Python",
        )
    except subprocess.TimeoutExpired:
        record("nvitop", "WARN", "probe timed out")
    report = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "profile": profile["slug"],
        "server_uid": profile["trust"]["server_uid"],
        "ssh_trust_uid": profile["trust"]["server_uid"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "checks": checks,
        "ready": not any(item["status"] == "FAIL" for item in checks),
    }
    return report


def _delegate(tool: Path, profile_name: str, arguments: list[str]) -> int:
    environment = subprocess_environment(profile_name)
    os.execve(sys.executable, [sys.executable, str(tool), *arguments], environment)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="server profile slug")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="configure a server interactively or import a profile")
    setup.add_argument("--from-json", type=Path)
    setup.add_argument("--replace", action="store_true")
    setup.add_argument("--offline", action="store_true", help="import without a live doctor check")
    subparsers.add_parser("profiles", help="list configured profiles")
    use = subparsers.add_parser("use", help="select the active profile")
    use.add_argument("name")
    subparsers.add_parser("show", help="show the selected non-secret profile")
    doctor = subparsers.add_parser("doctor", help="run read-only readiness checks")
    doctor.add_argument("--json", action="store_true")
    for name, help_text in (
        ("ssh", "test strict SSH or open a forwarding-only connection"),
        ("ticket", "operate the shared GPU ticket ledger"),
        ("dashboard", "manually control the local dashboard"),
        ("tensorboard", "configure or diagnose a TensorBoard source"),
        ("project", "push committed source and verify a remote execution clone"),
        ("infra", "install or inspect server-level monitoring tools"),
        ("gpu", "show a read-only nvidia-smi snapshot"),
        ("run", "run a ticket-bound Python target in PyTorch compatibility mode"),
    ):
        delegated = subparsers.add_parser(name, help=help_text, add_help=False)
        delegated.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def parse_command_line(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse router options while leaving every delegated option untouched."""

    raw = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "--profile":
            index += 2
            continue
        if token.startswith("--profile="):
            index += 1
            continue
        break
    if index < len(raw) and raw[index] in DELEGATED_TOOLS:
        arguments = raw[index + 1 :]
        parsed = build_parser().parse_args(raw[: index + 1])
        parsed.arguments = arguments
        return parsed
    return build_parser().parse_args(raw)


def main() -> int:
    args = parse_command_line()
    try:
        if args.command == "setup":
            if args.from_json:
                profile = _setup_from_file(
                    args.from_json.resolve(), replace=args.replace, offline=args.offline
                )
            else:
                if args.offline:
                    raise SetupError("interactive setup cannot be offline")
                profile = setup_interactive(replace=args.replace)
            initialized = subprocess.run(
                [sys.executable, str(TICKET_TOOL), "init"],
                env=subprocess_environment(profile["slug"]),
                text=True,
                check=False,
            )
            if initialized.returncode != 0:
                raise SetupError("profile 已保存，但工单账本初始化失败")
            if profile.pop("_install_monitor_requested", False):
                completed = subprocess.run(
                    [sys.executable, str(INFRA_TOOL), "install-monitor"],
                    env=subprocess_environment(profile["slug"]),
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    print(
                        "profile 已保存，但 nvitop 专用环境安装失败；可稍后运行 "
                        "remote-gpu-dev infra install-monitor",
                        file=sys.stderr,
                    )
            print(f"profile ready: {profile['slug']} ({profile_path(profile['slug'])})")
            print("next: remote-gpu-dev doctor")
            return 0
        if args.command == "profiles":
            try:
                active = selected_profile_name()
            except ProfileError:
                active = None
            for name in list_profiles():
                print(("* " if name == active else "  ") + name)
            return 0
        if args.command == "use":
            path = set_active_profile(args.name)
            print(f"active profile={args.name} file={path}")
            return 0
        profile = load_profile(args.profile)
        if args.command == "show":
            print(json.dumps(public_profile(profile), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "doctor":
            report = run_doctor(profile)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                for item in report["checks"]:
                    print(f"{item['status']:4} {item['name']}: {item['detail']}")
                print(f"ready={str(report['ready']).lower()}")
            return 0 if report["ready"] else 1
        return _delegate(DELEGATED_TOOLS[args.command], profile["slug"], args.arguments)
    except (ProfileError, SetupError, SSHError, ManagedRunError, OSError, subprocess.SubprocessError) as exc:
        print(f"remote-gpu-dev: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
