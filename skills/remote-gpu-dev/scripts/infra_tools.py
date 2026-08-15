#!/usr/bin/env python3
"""Manage root-bound Conda environments and monitoring infrastructure."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Any

from profile import ProfileError, load_profile
from remote_path_guard import (
    RemotePathError,
    canonical_guard_command,
    conda_read_exec_root,
    managed_runtime_environment,
    managed_runtime_paths,
    require_managed_remote_path,
)
from ssh_remote import SSHError, add_profile_proxy_forward, ssh_argv
from managed_run import ManagedRunError, build_landlock_command


class InfraError(RuntimeError):
    pass


PACKAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+*<>=!-]{0,127}\Z")
PYTHON_RE = re.compile(r"3\.(?:10|11|12|13|14)(?:\.\d+)?\Z")


def _package(value: str) -> str:
    if not PACKAGE_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("package specs may not contain paths or URLs")
    return value


def _python(value: str) -> str:
    if not PYTHON_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("Python must be a supported 3.x version")
    return value


def _monitor_prefix(profile: dict[str, Any]) -> str:
    python = PurePosixPath(profile["remote"]["monitor_python"])
    if python.name != "python" or python.parent.name != "bin":
        raise InfraError("remote.monitor_python must end in /bin/python")
    prefix = str(python.parent.parent)
    durable = PurePosixPath(profile["remote"]["durable_root"])
    path = PurePosixPath(prefix)
    if path != durable and durable not in path.parents:
        raise InfraError("monitor environment must be inside the managed durable root")
    return prefix


def _remote(
    profile: dict[str, Any], command: str, *, proxy: bool = False, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    command = build_landlock_command(
        profile,
        ["/bin/sh", "-c", command],
        workdir=profile["remote"]["temp_root"],
        device_ids=profile["gpu"]["ids"],
        read_exec_roots=[conda_read_exec_root(profile)],
    )
    argv = ssh_argv(profile, batch=True)
    if proxy:
        add_profile_proxy_forward(profile, argv)
    argv.extend([f"{profile['ssh']['user']}@{profile['ssh']['host']}", command])
    try:
        return subprocess.run(
            argv,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, SSHError) as exc:
        raise InfraError(f"remote command failed to run: {exc}") from exc


def _install_command(profile: dict[str, Any], *, proxy: bool) -> str:
    conda = profile["remote"]["conda_executable"]
    prefix = _monitor_prefix(profile)
    remote_proxy = f"http://{profile['remote']['proxy_host']}:{profile['remote']['proxy_port']}"
    runtime = managed_runtime_paths(profile)
    runtime_environment = managed_runtime_environment(profile)
    runtime_directories = [
        value for key, value in runtime.items() if key != "condarc"
    ]
    for path in [prefix, runtime["condarc"], *runtime_directories]:
        require_managed_remote_path(profile, path, "Conda runtime path")
    environment: list[str] = [
        "env", "CUDA_VISIBLE_DEVICES=", *(
            f"{key}={value}" for key, value in runtime_environment.items()
        )
    ]
    if proxy:
        environment.extend(
            [
                f"HTTP_PROXY={remote_proxy}",
                f"HTTPS_PROXY={remote_proxy}",
                "NO_PROXY=127.0.0.1,localhost",
            ]
        )
    create = [
        *environment,
        conda,
        "create",
        "--yes",
        "--prefix",
        prefix,
        "--override-channels",
        "--strict-channel-priority",
        "--channel",
        "conda-forge",
        "python=3.12",
        "nvitop>=1.4,<2",
    ]
    install = [
        *environment,
        conda,
        "install",
        "--yes",
        "--prefix",
        prefix,
        "--freeze-installed",
        "--override-channels",
        "--strict-channel-priority",
        "--channel",
        "conda-forge",
        "nvitop>=1.4,<2",
    ]
    verify_code = (
        "import json,nvitop; from nvitop import Device; "
        "print(json.dumps({'nvitop':nvitop.__version__,'gpus':len(Device.all())}))"
    )
    guarded_paths = [str(PurePosixPath(prefix).parent), *runtime_directories]
    return (
        "set -eu; "
        + canonical_guard_command(
            profile, guarded_paths, create_directories=True
        )
        + "; "
        + f": > {shlex.quote(runtime['condarc'])}; "
        f"if test -d {shlex.quote(prefix + '/conda-meta')}; then {shlex.join(install)}; "
        f"elif test -e {shlex.quote(prefix)}; then echo 'monitor prefix exists but is not a Conda env' >&2; exit 42; "
        f"else if {shlex.join(create)} && test -d {shlex.quote(prefix + '/conda-meta')}; then :; "
        f"else status=$?; /bin/rm -rf -- {shlex.quote(prefix)}; exit \"$status\"; fi; fi; "
        f"{shlex.join([profile['remote']['monitor_python'], '-c', verify_code])}"
    )


def install_monitor(profile: dict[str, Any]) -> dict[str, Any]:
    policy = profile["network"]["conda_policy"]
    attempts: list[tuple[str, bool]] = []
    if policy in {"direct", "direct-then-proxy"}:
        attempts.append(("direct", False))
    if policy in {"proxy", "direct-then-proxy"}:
        if profile["network"]["proxy_policy"] != "on-demand":
            if policy == "proxy":
                raise InfraError("Conda requires proxy, but on-demand proxy is disabled")
        else:
            attempts.append(("proxy", True))
    errors: list[str] = []
    for label, proxy in attempts:
        result = _remote(profile, _install_command(profile, proxy=proxy), proxy=proxy)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout.splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as exc:
                raise InfraError("monitor install succeeded but verification output was invalid") from exc
            return {
                "status": "installed",
                "route": label,
                "prefix": _monitor_prefix(profile),
                "python": profile["remote"]["monitor_python"],
                "verification": payload,
            }
        detail = " ".join((result.stderr or result.stdout).split())[:300]
        errors.append(f"{label}: exit {result.returncode} {detail}")
    raise InfraError("; ".join(errors) or "no permitted Conda network route")


def monitor_status(profile: dict[str, Any]) -> dict[str, Any]:
    code = (
        "import json,nvitop; from nvitop import Device; "
        "print(json.dumps({'nvitop':nvitop.__version__,'gpus':len(Device.all())}))"
    )
    runtime = managed_runtime_paths(profile)
    runtime_environment = managed_runtime_environment(profile)
    environment = ["env", "CUDA_VISIBLE_DEVICES=", *(
        f"{key}={value}" for key, value in runtime_environment.items()
    )]
    command = (
        canonical_guard_command(
            profile,
            [_monitor_prefix(profile), *(
                value for key, value in runtime.items() if key != "condarc"
            )],
        )
        + "; "
        + shlex.join([*environment, profile["remote"]["monitor_python"], "-c", code])
    )
    result = _remote(profile, command, timeout=30)
    if result.returncode != 0:
        return {
            "status": "missing",
            "prefix": _monitor_prefix(profile),
            "detail": "monitor Python cannot import nvitop",
        }
    try:
        verification = json.loads(result.stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise InfraError("monitor status returned invalid JSON") from exc
    return {
        "status": "ready",
        "prefix": _monitor_prefix(profile),
        "verification": verification,
    }


def _project_env_command(
    profile: dict[str, Any], *, action: str, prefix: str, python: str | None,
    packages: list[str], proxy: bool,
) -> str:
    if action not in {"create", "install"}:
        raise InfraError("unsupported Conda environment action")
    if any(not PACKAGE_RE.fullmatch(package) for package in packages):
        raise InfraError("package specs may not contain paths or URLs")
    if python is not None and not PYTHON_RE.fullmatch(python):
        raise InfraError("Python must be a supported 3.x version")
    prefix = require_managed_remote_path(profile, prefix, "environment prefix")
    if prefix in {
        profile["remote"]["temp_root"], profile["remote"]["durable_root"]
    }:
        raise InfraError("environment prefix must be below, not equal to, a managed root")
    runtime = managed_runtime_paths(profile)
    managed_environment = managed_runtime_environment(profile)
    conda = profile["remote"]["conda_executable"]
    remote_proxy = (
        f"http://{profile['remote']['proxy_host']}:{profile['remote']['proxy_port']}"
    )
    environment = [
        "env", "CUDA_VISIBLE_DEVICES=",
        *(f"{key}={value}" for key, value in managed_environment.items()),
    ]
    if proxy:
        environment.extend([
            f"HTTP_PROXY={remote_proxy}", f"HTTPS_PROXY={remote_proxy}",
            "NO_PROXY=127.0.0.1,localhost",
        ])
    base = [*environment, conda, action, "--yes", "--prefix", prefix]
    if action == "create":
        assert python is not None
        command = [*base, "python=" + python, *packages]
        predicate = (
            f"test ! -e {shlex.quote(prefix)} || "
            "{ echo 'environment prefix already exists' >&2; exit 42; }; "
        )
    else:
        command = [*base, "--freeze-installed", *packages]
        predicate = (
            f"test -d {shlex.quote(prefix + '/conda-meta')} || "
            "{ echo 'prefix is not a Conda environment' >&2; exit 42; }; "
        )
    directories = [
        str(PurePosixPath(prefix).parent),
        *(value for key, value in runtime.items() if key != "condarc"),
    ]
    return (
        "set -eu; "
        + canonical_guard_command(profile, directories, create_directories=True)
        + "; "
        + f": > {shlex.quote(runtime['condarc'])}; "
        + predicate
        + (
            "if " + shlex.join(command)
            + f" && test -d {shlex.quote(prefix + '/conda-meta')}; then :; "
            + "else status=$?; "
            + (f"/bin/rm -rf -- {shlex.quote(prefix)}; " if action == "create" else "")
            + "exit \"$status\"; fi"
        )
    )


def _pip_environment(
    profile: dict[str, Any], *, proxy: bool
) -> list[str]:
    environment = managed_runtime_environment(profile)
    primary = profile["network"].get("pip_index_url")
    extras = profile["network"].get("pip_extra_index_urls") or []
    if primary:
        environment["PIP_INDEX_URL"] = primary
    if extras:
        environment["PIP_EXTRA_INDEX_URL"] = " ".join(extras)
    if proxy:
        remote_proxy = (
            f"http://{profile['remote']['proxy_host']}:"
            f"{profile['remote']['proxy_port']}"
        )
        environment.update({
            "HTTP_PROXY": remote_proxy,
            "HTTPS_PROXY": remote_proxy,
            "NO_PROXY": "127.0.0.1,localhost",
        })
    return ["env", "CUDA_VISIBLE_DEVICES=", *(
        f"{key}={value}" for key, value in environment.items()
    )]


def _project_pip_command(
    profile: dict[str, Any], *, prefix: str, packages: list[str], proxy: bool
) -> str:
    prefix = require_managed_remote_path(profile, prefix, "environment prefix")
    if prefix in {profile["remote"]["temp_root"], profile["remote"]["durable_root"]}:
        raise InfraError("environment prefix must be below, not equal to, a managed root")
    if not packages or any(not PACKAGE_RE.fullmatch(package) for package in packages):
        raise InfraError("pip package specs may not contain paths or URLs")
    runtime = managed_runtime_paths(profile)
    directories = [
        str(PurePosixPath(prefix).parent),
        *(value for key, value in runtime.items() if key != "condarc"),
    ]
    python = str(PurePosixPath(prefix) / "bin" / "python")
    command = [
        *_pip_environment(profile, proxy=proxy),
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *packages,
    ]
    return (
        "set -eu; "
        + canonical_guard_command(profile, directories, create_directories=True)
        + "; "
        + f"test -d {shlex.quote(prefix + '/conda-meta')} || "
        + "{ echo 'prefix is not a Conda environment' >&2; exit 42; }; "
        + f"test -x {shlex.quote(python)} || "
        + "{ echo 'environment Python is not executable' >&2; exit 42; }; "
        + shlex.join(command)
    )


def project_environment(
    profile: dict[str, Any], *, action: str, prefix: str, python: str | None,
    packages: list[str], use_proxy: bool,
) -> dict[str, Any]:
    policy = profile["network"]["conda_policy"]
    if use_proxy:
        if profile["network"]["proxy_policy"] != "on-demand":
            raise InfraError("on-demand proxy is disabled for this profile")
        routes = [("proxy", True)]
    elif policy == "proxy":
        if profile["network"]["proxy_policy"] != "on-demand":
            raise InfraError("Conda requires proxy, but on-demand proxy is disabled")
        routes = [("proxy", True)]
    elif policy == "direct-then-proxy" and profile["network"]["proxy_policy"] == "on-demand":
        routes = [("direct", False), ("proxy", True)]
    else:
        routes = [("direct", False)]
    errors: list[str] = []
    for route, proxy in routes:
        command = _project_env_command(
            profile, action=action, prefix=prefix, python=python,
            packages=packages, proxy=proxy,
        )
        result = _remote(profile, command, proxy=proxy)
        if result.returncode == 0:
            return {
                "status": "created" if action == "create" else "installed",
                "prefix": prefix, "packages": packages, "route": route,
            }
        errors.append(
            f"{route}: exit {result.returncode} "
            + " ".join((result.stderr or result.stdout).split())[:300]
        )
    raise InfraError("; ".join(errors) or "no permitted Conda route")


def pip_install_environment(
    profile: dict[str, Any], *, prefix: str, packages: list[str], use_proxy: bool
) -> dict[str, Any]:
    if use_proxy and profile["network"]["proxy_policy"] != "on-demand":
        raise InfraError("on-demand proxy is disabled for this profile")
    command = _project_pip_command(
        profile, prefix=prefix, packages=packages, proxy=use_proxy
    )
    result = _remote(profile, command, proxy=use_proxy)
    if result.returncode != 0:
        detail = " ".join((result.stderr or result.stdout).split())[:300]
        raise InfraError(
            f"{'proxy' if use_proxy else 'direct'}: exit {result.returncode} {detail}"
        )
    return {
        "status": "installed",
        "installer": "pip",
        "prefix": prefix,
        "packages": packages,
        "route": "proxy" if use_proxy else "direct",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("install-monitor", help="install nvitop in the dedicated Conda prefix")
    subparsers.add_parser("status", help="inspect the dedicated monitor environment")
    create = subparsers.add_parser("create-env", help="create one managed project Conda prefix")
    create.add_argument("--prefix", required=True)
    create.add_argument("--python", required=True, type=_python)
    create.add_argument("--package", action="append", type=_package, default=[])
    create.add_argument("--proxy", action="store_true")
    install = subparsers.add_parser("install-env", help="install packages into a managed project prefix")
    install.add_argument("--prefix", required=True)
    install.add_argument("--package", action="append", type=_package, required=True)
    install.add_argument("--proxy", action="store_true")
    pip_install = subparsers.add_parser(
        "pip-install-env", help="install closed-grammar PyPI specs into a managed Conda prefix"
    )
    pip_install.add_argument("--prefix", required=True)
    pip_install.add_argument("--package", action="append", type=_package, required=True)
    pip_install.add_argument("--proxy", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        profile = load_profile()
        if args.command == "install-monitor":
            payload = install_monitor(profile)
        elif args.command == "status":
            payload = monitor_status(profile)
        elif args.command == "pip-install-env":
            payload = pip_install_environment(
                profile,
                prefix=args.prefix,
                packages=args.package,
                use_proxy=args.proxy,
            )
        else:
            payload = project_environment(
                profile,
                action="create" if args.command == "create-env" else "install",
                prefix=args.prefix,
                python=getattr(args, "python", None),
                packages=args.package,
                use_proxy=args.proxy,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["status"] in {"created", "installed", "ready"} else 1
    except (ProfileError, RemotePathError, SSHError, ManagedRunError, InfraError) as exc:
        print(f"remote-gpu-infra: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
