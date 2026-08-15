#!/usr/bin/env python3
"""Lexical remote-file policy shared by remote-gpu-dev helpers.

This module deliberately does not pretend to be an operating-system sandbox.
It validates every remote user-file path accepted or derived by the tools.  A
trusted workload executed inside an allowed root can still make its own system
calls; containment of untrusted code requires a server-side sandbox.
"""

from __future__ import annotations

import posixpath
import shlex
from pathlib import PurePosixPath
from typing import Any, Mapping


class RemotePathError(ValueError):
    """A remote path is outside the two profile-managed roots."""


PROTECTED_RUNTIME_ENV = {
    "PATH",
    "CUDA_VISIBLE_DEVICES",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "REMOTE_GPU_DEV_TICKET",
}


def normalize_remote_path(value: Any, field: str = "remote path") -> str:
    """Return one normalized absolute POSIX path, rejecting ambiguous forms."""

    if not isinstance(value, str):
        raise RemotePathError(f"{field} must be a string")
    if value != value.strip() or not value or len(value) > 1024:
        raise RemotePathError(f"{field} must be a non-empty trimmed path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RemotePathError(f"{field} contains control characters")
    if not value.startswith("/") or value.startswith("//") or value == "/":
        raise RemotePathError(
            f"{field} must be an absolute remote path more specific than /"
        )
    if posixpath.normpath(value) != value or any(
        part in {".", ".."} for part in value.split("/")
    ):
        raise RemotePathError(f"{field} must be normalized without . or ..")
    return value


def _inside(path: str, root: str, *, allow_root: bool) -> bool:
    candidate = PurePosixPath(path)
    boundary = PurePosixPath(root)
    return (allow_root and candidate == boundary) or boundary in candidate.parents


def managed_roots(profile: Mapping[str, Any]) -> tuple[str, str]:
    """Return the only two remote roots in which user files may be handled."""

    try:
        remote = profile["remote"]
        temporary = normalize_remote_path(remote["temp_root"], "remote.temp_root")
        durable = normalize_remote_path(remote["durable_root"], "remote.durable_root")
    except (KeyError, TypeError) as exc:
        raise RemotePathError("profile has no valid remote managed roots") from exc
    if temporary == durable:
        raise RemotePathError("temporary and durable roots must be distinct")
    temporary_path = PurePosixPath(temporary)
    durable_path = PurePosixPath(durable)
    if temporary_path in durable_path.parents or durable_path in temporary_path.parents:
        raise RemotePathError("temporary and durable roots must not contain each other")
    return temporary, durable


def require_managed_remote_path(
    profile: Mapping[str, Any],
    value: Any,
    field: str = "remote path",
    *,
    allow_root: bool = True,
) -> str:
    """Require a path to equal or descend from temp_root or durable_root."""

    path = normalize_remote_path(value, field)
    roots = managed_roots(profile)
    if not any(_inside(path, root, allow_root=allow_root) for root in roots):
        raise RemotePathError(
            f"{field} is outside remote.temp_root and remote.durable_root"
        )
    return path


def validate_remote_layout(profile: Mapping[str, Any]) -> None:
    """Validate all profile paths except the read/execute-only Conda binary."""

    remote = profile["remote"]
    managed_roots(profile)
    for field in ("git_bare_root", "projects_root", "records_root", "monitor_python"):
        require_managed_remote_path(profile, remote[field], f"remote.{field}")
    # Conda itself is an explicit read/execute-only exception.  Environments,
    # package caches, temp files, and monitor Python are not exceptions.
    normalize_remote_path(remote["conda_executable"], "remote.conda_executable")
    conda_read_exec_root(profile)


def conda_read_exec_root(profile: Mapping[str, Any]) -> str:
    """Return a narrow Conda prefix permitted read/execute but never write."""

    executable = PurePosixPath(
        normalize_remote_path(
            profile["remote"]["conda_executable"], "remote.conda_executable"
        )
    )
    if executable.name != "conda" or executable.parent.name != "bin":
        raise RemotePathError("remote.conda_executable must end in /bin/conda")
    prefix = executable.parent.parent
    if str(prefix) in {"/", "/root", "/home", "/opt", "/usr", "/usr/local"}:
        raise RemotePathError("Conda read/execute prefix is too broad")
    return str(prefix)


def managed_runtime_paths(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return deterministic cache/temp/home locations under the temp root."""

    temporary, _durable = managed_roots(profile)
    base = temporary + "/runtime"
    return {
        "base": base,
        "home": base + "/home",
        "tmp": base + "/tmp",
        "xdg_cache": base + "/xdg/cache",
        "xdg_config": base + "/xdg/config",
        "xdg_data": base + "/xdg/data",
        "xdg_state": base + "/xdg/state",
        "conda_pkgs": base + "/conda-pkgs",
        "conda_envs": base + "/conda-envs",
        "condarc": base + "/conda/condarc",
        "pip_cache": base + "/pip-cache",
        "hf_home": base + "/huggingface",
        "hf_hub": base + "/huggingface/hub",
        "hf_datasets": base + "/huggingface/datasets",
        "hf_assets": base + "/huggingface/assets",
        "transformers": base + "/huggingface/transformers",
        "torch_home": base + "/torch",
        "torch_extensions": base + "/torch/extensions",
        "torch_inductor": base + "/torch/inductor",
        "torch_compile_debug": base + "/torch/compile-debug",
        "triton": base + "/triton",
        "cuda_cache": base + "/cuda-cache",
        "cupy": base + "/cupy",
        "numba": base + "/numba",
        "wandb": base + "/wandb",
        "wandb_cache": base + "/wandb/cache",
        "wandb_config": base + "/wandb/config",
        "wandb_data": base + "/wandb/data",
        "wandb_artifact": base + "/wandb/artifacts",
        "uv_cache": base + "/uv/cache",
        "uv_python": base + "/uv/python",
        "uv_tool": base + "/uv/tool",
        "python_pycache": base + "/python/pycache",
        "python_userbase": base + "/python/userbase",
        "matplotlib": base + "/matplotlib",
        "keras": base + "/keras",
        "ccache": base + "/ccache",
        "sccache": base + "/sccache",
        "cargo": base + "/cargo",
        "rustup": base + "/rustup",
        "go_cache": base + "/go/cache",
        "go_path": base + "/go/path",
        "vllm": base + "/vllm",
        "jax": base + "/jax",
        "tmux": base + "/tmux",
    }


def managed_runtime_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return common tool/cache variables, all rooted below remote.temp_root."""

    path = managed_runtime_paths(profile)
    environment = {
        "HOME": path["home"],
        "TMPDIR": path["tmp"],
        "TMP": path["tmp"],
        "TEMP": path["tmp"],
        "XDG_CACHE_HOME": path["xdg_cache"],
        "XDG_CONFIG_HOME": path["xdg_config"],
        "XDG_DATA_HOME": path["xdg_data"],
        "XDG_STATE_HOME": path["xdg_state"],
        "CONDA_PKGS_DIRS": path["conda_pkgs"],
        "CONDA_ENVS_PATH": path["conda_envs"],
        "CONDARC": path["condarc"],
        "CONDA_NO_PLUGINS": "true",
        "PIP_CACHE_DIR": path["pip_cache"],
        "PIP_CONFIG_FILE": "/dev/null",
        "HF_HOME": path["hf_home"],
        "HF_HUB_CACHE": path["hf_hub"],
        "HUGGINGFACE_HUB_CACHE": path["hf_hub"],
        "HF_DATASETS_CACHE": path["hf_datasets"],
        "HF_ASSETS_CACHE": path["hf_assets"],
        "TRANSFORMERS_CACHE": path["transformers"],
        "TORCH_HOME": path["torch_home"],
        "TORCH_EXTENSIONS_DIR": path["torch_extensions"],
        "TORCHINDUCTOR_CACHE_DIR": path["torch_inductor"],
        "TORCH_COMPILE_DEBUG_DIR": path["torch_compile_debug"],
        "TRITON_CACHE_DIR": path["triton"],
        "CUDA_CACHE_PATH": path["cuda_cache"],
        "CUPY_CACHE_DIR": path["cupy"],
        "NUMBA_CACHE_DIR": path["numba"],
        "WANDB_DIR": path["wandb"],
        "WANDB_CACHE_DIR": path["wandb_cache"],
        "WANDB_CONFIG_DIR": path["wandb_config"],
        "WANDB_DATA_DIR": path["wandb_data"],
        "WANDB_ARTIFACT_DIR": path["wandb_artifact"],
        "UV_CACHE_DIR": path["uv_cache"],
        "UV_PYTHON_INSTALL_DIR": path["uv_python"],
        "UV_TOOL_DIR": path["uv_tool"],
        "PYTHONPYCACHEPREFIX": path["python_pycache"],
        "PYTHONUSERBASE": path["python_userbase"],
        "PYTHONNOUSERSITE": "1",
        "MPLCONFIGDIR": path["matplotlib"],
        "KERAS_HOME": path["keras"],
        "CCACHE_DIR": path["ccache"],
        "SCCACHE_DIR": path["sccache"],
        "CARGO_HOME": path["cargo"],
        "RUSTUP_HOME": path["rustup"],
        "GOCACHE": path["go_cache"],
        "GOPATH": path["go_path"],
        "VLLM_CACHE_ROOT": path["vllm"],
        "JAX_COMPILATION_CACHE_DIR": path["jax"],
        "TMUX_TMPDIR": path["tmux"],
    }
    PROTECTED_RUNTIME_ENV.update(environment)
    return environment


_REMOTE_CANONICAL_GUARD = r'''
import pathlib, sys

mode = sys.argv[1]
temporary = pathlib.Path(sys.argv[2])
durable = pathlib.Path(sys.argv[3])
targets = [pathlib.Path(item) for item in sys.argv[4:]]

def inside(path, root):
    return path == root or root in path.parents

roots = [temporary, durable]
for root in roots:
    resolved = root.resolve(strict=True)
    if resolved != root or not root.is_dir():
        raise SystemExit("managed root is missing, not a directory, or uses a symlink")
for target in targets:
    cursor = target
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    ancestor = cursor.resolve(strict=True)
    if not any(inside(ancestor, root) for root in roots):
        raise SystemExit("remote target escapes managed roots")
    if mode == "create":
        target.mkdir(parents=True, exist_ok=True)
    resolved = target.resolve(strict=True)
    if not any(inside(resolved, root) for root in roots):
        raise SystemExit("resolved remote target escapes managed roots")
'''


def canonical_guard_command(
    profile: Mapping[str, Any],
    paths: list[str],
    *,
    create_directories: bool = False,
) -> str:
    """Build a fixed remote Python guard for trusted structured operations.

    It detects symlink escapes before the following operation, but—as an
    ordinary user-space preflight—it cannot eliminate a malicious concurrent
    rename race.  Server-side containment is required for hostile workloads.
    """

    temporary, durable = managed_roots(profile)
    normalized = [
        require_managed_remote_path(profile, path, "remote operation path")
        for path in paths
    ]
    argv = [
        "/usr/bin/python3",
        "-c",
        _REMOTE_CANONICAL_GUARD,
        "create" if create_directories else "check",
        temporary,
        durable,
        *normalized,
    ]
    return shlex.join(argv)
