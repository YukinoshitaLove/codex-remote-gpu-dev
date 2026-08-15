#!/usr/bin/env python3
"""Deploy local-authoritative Git source to one remote execution clone."""

from __future__ import annotations

import argparse
import hashlib
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
    canonical_guard_command,
    require_managed_remote_path,
)
from ssh_remote import SSHError, ssh_argv
from managed_run import ManagedRunError, build_landlock_command


PROJECT_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\Z")
GIT_TRANSPORT = Path(__file__).resolve().with_name("git_transport.py")
ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".npz",
    ".npy",
    ".tar",
    ".zip",
    ".7z",
}
ARTIFACT_PARTS = {
    "checkpoints",
    "weights",
    "datasets",
    "data",
    "records",
    "runs",
    "logs",
    "tensorboard",
    "wandb",
    ".conda",
    ".venv",
}


class ProjectError(RuntimeError):
    pass


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=env,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProjectError(f"could not run {argv[0]}: {exc}") from exc
    return completed


def _checked(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> str:
    completed = _run(argv, cwd=cwd, env=env, timeout=timeout)
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[:500]
        raise ProjectError(f"{Path(argv[0]).name} exited {completed.returncode}: {detail}")
    return completed.stdout.strip()


def _project_name(value: str) -> str:
    if not PROJECT_RE.fullmatch(value):
        raise ProjectError("project name must be 1-64 lowercase ASCII letters, digits, or hyphens")
    return value


def _local_repository(profile: dict[str, Any], project: str, explicit: Path | None) -> Path:
    root = Path(profile["local"]["projects_root"]).resolve()
    path = explicit.expanduser().resolve() if explicit else root / project
    if path != root / project:
        raise ProjectError(f"local repository must exactly equal {root / project}")
    if not path.is_dir():
        raise ProjectError(f"local repository does not exist: {path}")
    top = _checked(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if Path(top).resolve() != path:
        raise ProjectError("project path is not the Git worktree root")
    return path


def _tracked_files(repository: Path) -> list[str]:
    output = _checked(["git", "ls-files", "-z"], cwd=repository)
    return [item for item in output.split("\0") if item]


def _validate_repository(profile: dict[str, Any], repository: Path) -> str:
    if _checked(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository):
        raise ProjectError("local repository must be clean, including untracked files")
    ignored = _checked(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        cwd=repository,
    )
    if ignored:
        raise ProjectError(
            "local repository contains ignored files; move runtime data outside "
            "the source worktree before deployment"
        )
    commit = _checked(["git", "rev-parse", "HEAD"], cwd=repository)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProjectError("HEAD is not a full SHA-1 commit ID")
    submodules = _run(["git", "submodule", "status", "--recursive"], cwd=repository)
    if submodules.returncode != 0:
        raise ProjectError("git submodule status failed")
    if submodules.stdout.strip() and not profile["git"]["allow_submodules"]:
        raise ProjectError("submodules are disabled by the server profile")
    max_bytes = profile["git"]["max_tracked_file_mib"] * 1024 * 1024
    for relative in _tracked_files(repository):
        path = repository / relative
        if path.is_symlink():
            raise ProjectError(
                f"tracked symlink is forbidden because its target is outside "
                f"the verified commit content: {relative}"
            )
        parts = set(Path(relative).parts)
        suffix = Path(relative).suffix.lower()
        if parts & ARTIFACT_PARTS or suffix in ARTIFACT_SUFFIXES:
            raise ProjectError(f"tracked non-source artifact is forbidden: {relative}")
        if path.is_file() and path.stat().st_size > max_bytes:
            raise ProjectError(f"tracked file exceeds {profile['git']['max_tracked_file_mib']} MiB: {relative}")
        if path.is_file() and path.stat().st_size < 1024:
            head = path.read_bytes()[:200]
            if head.startswith(b"version https://git-lfs.github.com/spec/v1") and not profile["git"]["allow_lfs"]:
                raise ProjectError(f"Git LFS pointer is disabled: {relative}")
    return commit


def _remote_paths(profile: dict[str, Any], project: str) -> tuple[str, str]:
    paths = (
        profile["remote"]["git_bare_root"].rstrip("/") + f"/{project}.git",
        profile["remote"]["projects_root"].rstrip("/") + f"/{project}",
    )
    return tuple(
        require_managed_remote_path(profile, path, "remote Git path")
        for path in paths
    )


def _remote(profile: dict[str, Any], command: str, timeout: float = 120) -> str:
    command = build_landlock_command(
        profile,
        ["/bin/sh", "-c", command],
        workdir=profile["remote"]["temp_root"],
    )
    argv = ssh_argv(profile, batch=True)
    argv.extend([f"{profile['ssh']['user']}@{profile['ssh']['host']}", command])
    return _checked(argv, timeout=timeout)


def _git_ssh_command(profile: dict[str, Any], bare: str) -> str:
    # Git requires a command string here.  Every token is separately shell-quoted
    # from already validated profile fields; no user-supplied command fragments
    # are accepted.
    return shlex.join(
        [
            sys.executable,
            str(GIT_TRANSPORT),
            "--profile",
            profile["slug"],
            "--bare",
            bare,
        ]
    )


def _remote_url(profile: dict[str, Any], bare: str) -> str:
    host = profile["ssh"]["host"]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"ssh://{profile['ssh']['user']}@{host}:{profile['ssh']['port']}{bare}"


def deploy(profile: dict[str, Any], project: str, local: Path | None) -> dict[str, Any]:
    project = _project_name(project)
    repository = _local_repository(profile, project, local)
    commit = _validate_repository(profile, repository)
    bare, checkout = _remote_paths(profile, project)
    initialize = (
        "set -eu; "
        + canonical_guard_command(
            profile,
            [
                profile["remote"]["git_bare_root"],
                profile["remote"]["projects_root"],
            ],
            create_directories=True,
        )
        + "; "
        f"if test ! -e {shlex.quote(bare)}; then git init --bare {shlex.quote(bare)} >/dev/null; "
        f"elif test ! -d {shlex.quote(bare)} || test \"$(git --git-dir={shlex.quote(bare)} rev-parse --is-bare-repository)\" != true; "
        "then echo 'remote bare path is not a bare repository' >&2; exit 41; fi"
    )
    _remote(profile, initialize)
    environment = os.environ.copy()
    environment["GIT_SSH_COMMAND"] = _git_ssh_command(profile, bare)
    _checked(
        ["git", "push", _remote_url(profile, bare), f"{commit}:refs/heads/main"],
        cwd=repository,
        env=environment,
        timeout=300,
    )
    checkout_command = (
        "set -eu; "
        + canonical_guard_command(
            profile,
            [profile["remote"]["git_bare_root"], profile["remote"]["projects_root"]],
        )
        + "; "
        f"if test ! -e {shlex.quote(checkout)}; then git clone {shlex.quote(bare)} {shlex.quote(checkout)} >/dev/null; fi; "
        f"test -d {shlex.quote(checkout + '/.git')}; "
        f"test -z \"$(git -C {shlex.quote(checkout)} status --porcelain=v1 --untracked-files=all)\"; "
        f"test \"$(git -C {shlex.quote(checkout)} ls-files --others --ignored --exclude-standard -z | wc -c)\" -eq 0; "
        f"test \"$(git -C {shlex.quote(checkout)} remote get-url origin)\" = {shlex.quote(bare)}; "
        f"git -C {shlex.quote(checkout)} fetch --prune origin >/dev/null; "
        f"git -C {shlex.quote(checkout)} cat-file -e {shlex.quote(commit + '^{commit}')}; "
        f"git -C {shlex.quote(checkout)} checkout --detach {shlex.quote(commit)} >/dev/null; "
        f"test \"$(git -C {shlex.quote(checkout)} rev-parse HEAD)\" = {shlex.quote(commit)}; "
        f"test -z \"$(git -C {shlex.quote(checkout)} status --porcelain=v1 --untracked-files=all)\"; "
        f"test \"$(git -C {shlex.quote(checkout)} ls-files --others --ignored --exclude-standard -z | wc -c)\" -eq 0; "
        f"printf '%s\\n' {shlex.quote(commit)}"
    )
    remote_commit = _remote(profile, checkout_command, timeout=180)
    if remote_commit.splitlines()[-1:] != [commit]:
        raise ProjectError("remote checkout did not report the expected commit")
    return {
        "status": "deployed",
        "profile": profile["slug"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "ssh_trust_uid": profile["trust"]["server_uid"],
        "project": project,
        "local_repository": str(repository),
        "commit": commit,
        "remote_bare": bare,
        "remote_checkout": checkout,
    }


def verify(profile: dict[str, Any], project: str, commit: str | None) -> dict[str, Any]:
    project = _project_name(project)
    bare, checkout = _remote_paths(profile, project)
    expected = commit
    if expected is None:
        repository = _local_repository(profile, project, None)
        expected = _validate_repository(profile, repository)
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ProjectError("--commit must be a full lowercase 40-character SHA")
    command = (
        "set -eu; "
        + canonical_guard_command(
            profile,
            [profile["remote"]["git_bare_root"], profile["remote"]["projects_root"]],
        )
        + "; "
        f"test \"$(git --git-dir={shlex.quote(bare)} rev-parse --is-bare-repository)\" = true; "
        f"test -d {shlex.quote(checkout + '/.git')}; "
        f"test \"$(git -C {shlex.quote(checkout)} rev-parse HEAD)\" = {shlex.quote(expected)}; "
        f"test -z \"$(git -C {shlex.quote(checkout)} status --porcelain=v1 --untracked-files=all)\"; "
        f"test \"$(git -C {shlex.quote(checkout)} ls-files --others --ignored --exclude-standard -z | wc -c)\" -eq 0; "
        f"test \"$(git -C {shlex.quote(checkout)} remote get-url origin)\" = {shlex.quote(bare)}; "
        f"printf '%s\\n' {shlex.quote(expected)}"
    )
    actual = _remote(profile, command)
    if actual.splitlines()[-1:] != [expected]:
        raise ProjectError("remote verification returned an unexpected commit")
    return {
        "status": "verified",
        "profile": profile["slug"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "ssh_trust_uid": profile["trust"]["server_uid"],
        "project": project,
        "commit": expected,
        "remote_bare": bare,
        "remote_checkout": checkout,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    deploy_parser = subparsers.add_parser("deploy", help="push clean source and checkout exact HEAD")
    deploy_parser.add_argument("project")
    deploy_parser.add_argument("--local", type=Path)
    verify_parser = subparsers.add_parser("verify", help="verify a clean exact remote checkout")
    verify_parser.add_argument("project")
    verify_parser.add_argument("--commit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        profile = load_profile()
        result = (
            deploy(profile, args.project, args.local)
            if args.command == "deploy"
            else verify(profile, args.project, args.commit)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ProfileError, RemotePathError, SSHError, ManagedRunError, ProjectError) as exc:
        print(f"remote-gpu-project: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
