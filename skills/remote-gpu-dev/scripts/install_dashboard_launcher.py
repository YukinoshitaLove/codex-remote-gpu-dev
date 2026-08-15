#!/usr/bin/env python3
"""Install the active-profile remote GPU dashboard launcher."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve()
LAUNCHER = SCRIPT.with_name("remote_gpu_dashboard.py")
COMMAND_NAME = "remote-gpu-dashboard"
DESKTOP_NAME = "remote-gpu-dashboard.desktop"
OWNERSHIP_MARKER = "X-Remote-GPU-Dev-Dashboard=true"


class InstallError(RuntimeError):
    pass


def install_paths(home: Path) -> tuple[Path, Path]:
    return (
        home / ".local" / "bin" / COMMAND_NAME,
        home / ".local" / "share" / "applications" / DESKTOP_NAME,
    )


def desktop_text(command: Path) -> str:
    if any(char in str(command) for char in "\n\r\0"):
        raise InstallError("launcher command path contains an invalid character")
    return (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=远程 GPU 工单看板\n"
        "Name[en]=Remote GPU Dashboard\n"
        "Comment=Open the active server GPU ticket and TensorBoard dashboard\n"
        f"Exec={command}\n"
        f"TryExec={command}\n"
        "Icon=utilities-system-monitor\n"
        "Terminal=false\n"
        "Categories=Development;\n"
        "StartupNotify=true\n"
        f"{OWNERSHIP_MARKER}\n"
    )


def atomic_write(path: Path, content: str, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def install(home: Path) -> tuple[Path, Path]:
    command, desktop = install_paths(home)
    command.parent.mkdir(parents=True, exist_ok=True)
    desktop.parent.mkdir(parents=True, exist_ok=True)

    if command.exists() or command.is_symlink():
        if not command.is_symlink() or command.resolve() != LAUNCHER:
            raise InstallError(f"refusing to replace unrelated launcher: {command}")
    else:
        temporary = command.with_name(f".{command.name}.{os.getpid()}.tmp")
        try:
            temporary.symlink_to(LAUNCHER)
            os.replace(temporary, command)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    expected = desktop_text(command)
    if desktop.exists():
        existing = desktop.read_text(encoding="utf-8")
        if existing != expected and OWNERSHIP_MARKER not in existing:
            raise InstallError(f"refusing to replace unrelated desktop entry: {desktop}")
    atomic_write(desktop, expected, 0o644)
    return command, desktop


def check(home: Path) -> tuple[Path, Path]:
    command, desktop = install_paths(home)
    if not command.is_symlink() or command.resolve() != LAUNCHER:
        raise InstallError(f"launcher is missing or points elsewhere: {command}")
    if not desktop.is_file() or desktop.read_text(encoding="utf-8") != desktop_text(
        command
    ):
        raise InstallError(f"desktop entry is missing or stale: {desktop}")
    return command, desktop


def refresh_desktop_database(desktop: Path) -> None:
    executable = Path("/usr/bin/update-desktop-database")
    if not executable.is_file():
        return
    subprocess.run(
        [str(executable), str(desktop.parent)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the existing user installation"
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=Path.home(),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        command, desktop = check(args.home) if args.check else install(args.home)
    except (InstallError, OSError) as exc:
        print(f"dashboard-launcher: {exc}", file=sys.stderr)
        return 2
    if not args.check:
        refresh_desktop_database(desktop)
    state = "verified" if args.check else "installed"
    print(f"dashboard launcher {state}: command={command} desktop={desktop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
