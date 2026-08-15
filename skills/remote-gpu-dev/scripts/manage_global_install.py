#!/usr/bin/env python3
"""Install, update, check, or recoverably uninstall remote-gpu-dev globally."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import py_compile
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SKILL_NAME = "remote-gpu-dev"
MARKER = ".remote-gpu-dev-install.json"
COMMANDS = {
    "remote-gpu-dev": "scripts/remote_gpu.py",
    "remote-gpu-dashboard": "scripts/remote_gpu_dashboard.py",
}
DESKTOP_NAME = "remote-gpu-dashboard.desktop"
DESKTOP_MARKER = "X-Remote-GPU-Dev-Dashboard=true"
FRONTMATTER_KEYS = {"name", "description"}
MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


class InstallError(RuntimeError):
    pass


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser().resolve()


def default_source() -> Path:
    return Path(__file__).resolve().parent.parent


def target_path() -> Path:
    # Resolve an existing symlinked `skills` parent once and then use this same
    # canonical identity for the installed tree and every launcher comparison.
    return (codex_home() / "skills" / SKILL_NAME).resolve(strict=False)


def backup_root() -> Path:
    return (codex_home() / "skill-backups").resolve(strict=False)


def _expected_mode(relative: Path, *, directory: bool) -> int:
    if directory:
        return 0o755
    if relative.parts and relative.parts[0] == "scripts" and relative.suffix in {".py", ".sh"}:
        return 0o755
    return 0o644


def _hash_tree(root: Path, *, normalized_modes: bool = False) -> str:
    digest = hashlib.sha256()
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
    for path in paths:
        relative = Path(".") if path == root else path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name == MARKER:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise InstallError(f"skill source must not contain symlinks: {path}")
        directory = stat.S_ISDIR(metadata.st_mode)
        if not directory and not stat.S_ISREG(metadata.st_mode):
            raise InstallError(f"skill source contains an unsupported file type: {path}")
        mode = (
            _expected_mode(relative, directory=directory)
            if normalized_modes
            else stat.S_IMODE(metadata.st_mode)
        )
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"d" if directory else b"f")
        digest.update(b"\0")
        digest.update(f"{mode:04o}".encode("ascii"))
        digest.update(b"\0")
        if not directory:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _frontmatter_scalar(raw: str, *, key: str) -> str:
    value = raw.strip()
    if not value:
        raise InstallError(f"SKILL.md frontmatter field {key!r} must not be empty")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InstallError(f"SKILL.md frontmatter field {key!r} has invalid quoting") from exc
        if not isinstance(parsed, str):
            raise InstallError(f"SKILL.md frontmatter field {key!r} must be a string")
        return parsed.strip()
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise InstallError(f"SKILL.md frontmatter field {key!r} has invalid quoting")
        return value[1:-1].replace("''", "'").strip()
    if value[0] in "[{&*!|>" or value.lower() in {"null", "true", "false", "~"}:
        raise InstallError(f"SKILL.md frontmatter field {key!r} must be a string")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        raise InstallError(f"SKILL.md frontmatter field {key!r} must be a string")
    return value


def _validate_skill_frontmatter(skill_md: Path) -> None:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
    if match is None:
        raise InstallError("SKILL.md has invalid YAML frontmatter boundaries")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(match.group(1).splitlines(), start=2):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in raw_line:
            raise InstallError(f"SKILL.md frontmatter line {line_number} is not a scalar field")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if key not in FRONTMATTER_KEYS:
            allowed = ", ".join(sorted(FRONTMATTER_KEYS))
            raise InstallError(
                f"unexpected SKILL.md frontmatter field {key!r}; allowed fields: {allowed}"
            )
        if key in values:
            raise InstallError(f"duplicate SKILL.md frontmatter field: {key}")
        values[key] = _frontmatter_scalar(raw_value, key=key)
    missing = FRONTMATTER_KEYS - values.keys()
    if missing:
        raise InstallError(
            "SKILL.md frontmatter is missing required field(s): " + ", ".join(sorted(missing))
        )
    name = values["name"]
    if name != SKILL_NAME:
        raise InstallError(f"SKILL.md name must be {SKILL_NAME!r}, got {name!r}")
    if len(name) > MAX_SKILL_NAME_LENGTH or not re.fullmatch(r"[a-z0-9-]+", name):
        raise InstallError("SKILL.md name is not a valid hyphen-case skill name")
    description = values["description"]
    if not description:
        raise InstallError("SKILL.md description must not be empty")
    if "<" in description or ">" in description:
        raise InstallError("SKILL.md description cannot contain angle brackets")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise InstallError(
            f"SKILL.md description exceeds {MAX_DESCRIPTION_LENGTH} characters"
        )


def validate_source(source: Path) -> str:
    source = source.resolve()
    skill_md = source / "SKILL.md"
    if not skill_md.is_file():
        raise InstallError(f"SKILL.md is missing: {source}")
    _validate_skill_frontmatter(skill_md)
    for command in COMMANDS.values():
        if not (source / command).is_file():
            raise InstallError(f"required command is missing: {command}")
    with tempfile.TemporaryDirectory(prefix="remote-gpu-dev-compile-") as temporary:
        for script in (source / "scripts").glob("*.py"):
            try:
                py_compile.compile(
                    str(script),
                    cfile=str(Path(temporary) / f"{script.stem}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                raise InstallError(f"Python compilation failed: {script.name}") from exc
    return _hash_tree(source, normalized_modes=True)


def _read_marker(target: Path) -> dict[str, Any]:
    try:
        value = json.loads((target / MARKER).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"target is not a managed {SKILL_NAME} install: {target}") from exc
    if not isinstance(value, dict) or value.get("skill") != SKILL_NAME:
        raise InstallError(f"invalid ownership marker in {target}")
    return value


def _write_marker(stage: Path, digest: str, source: Path) -> None:
    value = {
        "schema_version": 1,
        "skill": SKILL_NAME,
        "tree_sha256": digest,
        "source_hint": source.name,
        "installed_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = stage / MARKER
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _normalize_tree(root: Path) -> None:
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
    for path in paths:
        relative = Path(".") if path == root else path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise InstallError(f"staged skill must not contain symlinks: {path}")
        directory = stat.S_ISDIR(metadata.st_mode)
        if not directory and not stat.S_ISREG(metadata.st_mode):
            raise InstallError(f"staged skill contains an unsupported file type: {path}")
        os.chmod(path, _expected_mode(relative, directory=directory))


def _copy_stage(source: Path, parent: Path, digest: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{SKILL_NAME}.stage-", dir=parent))
    try:
        shutil.copytree(
            source,
            stage,
            dirs_exist_ok=True,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        _normalize_tree(stage)
        _write_marker(stage, digest, source)
        installed_digest = _hash_tree(stage)
        if installed_digest != digest:
            raise InstallError("staged skill digest differs from source")
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _launcher_paths() -> tuple[Path, Path]:
    return (
        Path.home() / ".local" / "bin",
        Path.home() / ".local" / "share" / "applications" / DESKTOP_NAME,
    )


def _desktop_text(command: Path) -> str:
    return (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=远程 GPU 工单看板\n"
        "Name[en]=Remote GPU Dashboard\n"
        "Comment=Open the active remote GPU ticket and TensorBoard dashboard\n"
        f"Exec={command}\n"
        f"TryExec={command}\n"
        "Icon=utilities-system-monitor\n"
        "Terminal=false\n"
        "Categories=Development;\n"
        "StartupNotify=true\n"
        f"{DESKTOP_MARKER}\n"
    )


def _lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _symlink_points_to(path: Path, expected: Path) -> bool:
    if not path.is_symlink():
        return False
    raw_target = Path(os.readlink(path))
    actual = raw_target if raw_target.is_absolute() else path.parent / raw_target
    return actual.resolve(strict=False) == expected.resolve(strict=False)


def _preflight_launchers(target: Path) -> None:
    bin_dir, desktop = _launcher_paths()
    for name, relative in COMMANDS.items():
        launcher = bin_dir / name
        expected = (target / relative).resolve(strict=False)
        if _lexists(launcher) and not _symlink_points_to(launcher, expected):
            raise InstallError(f"refusing to replace unrelated launcher: {launcher}")
    if _lexists(desktop):
        try:
            current = desktop.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InstallError(f"refusing to replace unrelated desktop entry: {desktop}") from exc
        if DESKTOP_MARKER not in current:
            raise InstallError(f"refusing to replace unrelated desktop entry: {desktop}")


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _commit_replacements(entries: list[tuple[Path, Path]]) -> None:
    committed: list[tuple[Path, Path | None]] = []
    try:
        for temporary, destination in entries:
            backup: Path | None = None
            if _lexists(destination):
                backup = _temporary_path(destination.parent, f".{destination.name}.backup-")
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            committed.append((destination, backup))
    except BaseException:
        for destination, backup in reversed(committed):
            if _lexists(destination):
                destination.unlink()
            if backup is not None and _lexists(backup):
                os.replace(backup, destination)
        raise
    else:
        for _, backup in committed:
            if backup is not None:
                backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in entries:
            temporary.unlink(missing_ok=True)


def install_launchers(target: Path) -> None:
    target = target.resolve(strict=False)
    _preflight_launchers(target)
    bin_dir, desktop = _launcher_paths()
    bin_dir.mkdir(parents=True, exist_ok=True)
    desktop.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, Path]] = []
    try:
        for name, relative in COMMANDS.items():
            launcher = bin_dir / name
            temporary = _temporary_path(bin_dir, f".{launcher.name}.new-")
            temporary.symlink_to((target / relative).resolve(strict=False))
            entries.append((temporary, launcher))
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{desktop.name}.new-", dir=desktop.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8") as handle:
            handle.write(_desktop_text(bin_dir / "remote-gpu-dashboard"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        entries.append((temporary, desktop))
        _commit_replacements(entries)
    finally:
        for temporary, _ in entries:
            temporary.unlink(missing_ok=True)


def remove_launchers(target: Path) -> None:
    target = target.resolve(strict=False)
    bin_dir, desktop = _launcher_paths()
    owned: list[Path] = []
    for name, relative in COMMANDS.items():
        launcher = bin_dir / name
        if _symlink_points_to(launcher, (target / relative).resolve(strict=False)):
            owned.append(launcher)
    if desktop.is_file():
        try:
            desktop_text = desktop.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            desktop_text = ""
        if DESKTOP_MARKER in desktop_text:
            owned.append(desktop)
    moved: list[tuple[Path, Path]] = []
    try:
        for path in owned:
            backup = _temporary_path(path.parent, f".{path.name}.remove-")
            os.replace(path, backup)
            moved.append((path, backup))
    except BaseException:
        for path, backup in reversed(moved):
            if _lexists(backup):
                os.replace(backup, path)
        raise
    else:
        for _, backup in moved:
            backup.unlink(missing_ok=True)


def _available_backup_path(label: str) -> Path:
    root = backup_root()
    root.mkdir(parents=True, exist_ok=True)
    base = f"{SKILL_NAME}-{label}-{_timestamp()}"
    candidate = root / base
    suffix = 1
    while _lexists(candidate):
        candidate = root / f"{base}-{suffix}"
        suffix += 1
    return candidate


def install(source: Path, *, update: bool) -> tuple[Path, Path | None]:
    source = source.resolve()
    digest = validate_source(source)
    target = target_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    old_backup: Path | None = None
    if target.exists():
        if not update:
            raise InstallError(f"target already exists; use update: {target}")
        _read_marker(target)
    elif update:
        raise InstallError(f"nothing to update: {target}")
    stage = _copy_stage(source, target.parent, digest)
    try:
        _preflight_launchers(target)
        if target.exists():
            old_backup = _available_backup_path("updated")
            os.replace(target, old_backup)
        try:
            os.replace(stage, target)
        except BaseException:
            # Committing the staged tree and moving the previous install out of
            # the way are one transaction.  If the stage cannot become the
            # target, restore the still-valid previous install before
            # propagating the failure; existing launchers then remain valid.
            if old_backup is not None and _lexists(old_backup) and not _lexists(target):
                os.replace(old_backup, target)
                old_backup = None
            raise
        try:
            install_launchers(target)
        except Exception:
            failed = _available_backup_path("failed")
            os.replace(target, failed)
            if old_backup is not None:
                os.replace(old_backup, target)
                install_launchers(target)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return target, old_backup


def check() -> dict[str, Any]:
    target = target_path()
    marker = _read_marker(target)
    normalized_digest = validate_source(target)
    if normalized_digest != marker.get("tree_sha256"):
        raise InstallError("installed source differs from its ownership marker")
    digest = _hash_tree(target)
    if digest != marker.get("tree_sha256"):
        raise InstallError("installed tree digest differs from its ownership marker")
    marker_mode = stat.S_IMODE((target / MARKER).stat().st_mode)
    if marker_mode != 0o600:
        raise InstallError(f"ownership marker mode is {marker_mode:04o}, expected 0600")
    bin_dir, desktop = _launcher_paths()
    for name, relative in COMMANDS.items():
        launcher = bin_dir / name
        if not _symlink_points_to(launcher, (target / relative).resolve(strict=False)):
            raise InstallError(f"launcher is missing or stale: {launcher}")
    if not desktop.is_file() or desktop.read_text(encoding="utf-8") != _desktop_text(
        bin_dir / "remote-gpu-dashboard"
    ):
        raise InstallError("desktop entry is missing or stale")
    return {"status": "verified", "target": str(target), "tree_sha256": digest}


def uninstall() -> Path:
    target = target_path()
    _read_marker(target)
    destination = _available_backup_path("uninstalled")
    os.replace(target, destination)
    try:
        remove_launchers(target)
    except BaseException:
        os.replace(destination, target)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("install", "update", "validate-source"):
        item = subparsers.add_parser(command)
        item.add_argument("--source", type=Path, default=default_source())
    subparsers.add_parser("check")
    subparsers.add_parser("uninstall")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command in {"install", "update"}:
            target, backup = install(args.source, update=args.command == "update")
            payload = {
                "status": "installed" if args.command == "install" else "updated",
                "target": str(target),
                "backup": str(backup) if backup else None,
            }
        elif args.command == "validate-source":
            payload = {
                "status": "valid",
                "source": str(args.source.resolve()),
                "tree_sha256": validate_source(args.source),
            }
        elif args.command == "check":
            payload = check()
        else:
            payload = {"status": "uninstalled", "backup": str(uninstall())}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (InstallError, OSError) as exc:
        print(f"remote-gpu-dev-install: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
