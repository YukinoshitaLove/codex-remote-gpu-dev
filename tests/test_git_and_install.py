#!/usr/bin/env python3
"""Local Git source-contract and managed-install tests."""

from __future__ import annotations

import os
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skills" / "remote-gpu-dev"
SCRIPT_ROOT = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import git_project  # noqa: E402


INSTALL_SPEC = importlib.util.spec_from_file_location(
    "manage_global_install_under_test", SCRIPT_ROOT / "manage_global_install.py"
)
assert INSTALL_SPEC is not None and INSTALL_SPEC.loader is not None
manage_global_install = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(manage_global_install)


class GitContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "projects" / "demo"
        self.repository.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.email", "test@example.test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.name", "Test"],
            check=True,
        )
        (self.repository / "train.py").write_text("print('ok')\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repository), "add", "train.py"], check=True)
        subprocess.run(["git", "-C", str(self.repository), "commit", "-qm", "initial"], check=True)
        self.profile = {
            "local": {"projects_root": str(self.root / "projects")},
            "git": {
                "allow_submodules": False,
                "allow_lfs": False,
                "max_tracked_file_mib": 10,
            },
        }

    def test_clean_source_commit_is_accepted(self) -> None:
        commit = git_project._validate_repository(self.profile, self.repository)
        self.assertRegex(commit, r"^[0-9a-f]{40}$")

    def test_weights_and_dirty_files_are_rejected(self) -> None:
        (self.repository / "weights.ckpt").write_bytes(b"fixture")
        subprocess.run(
            ["git", "-C", str(self.repository), "add", "weights.ckpt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-qm", "bad artifact"],
            check=True,
        )
        with self.assertRaises(git_project.ProjectError):
            git_project._validate_repository(self.profile, self.repository)
        subprocess.run(
            ["git", "-C", str(self.repository), "reset", "--hard", "HEAD^"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        (self.repository / "untracked.log").write_text("runtime\n", encoding="utf-8")
        with self.assertRaises(git_project.ProjectError):
            git_project._validate_repository(self.profile, self.repository)


class ManagedInstallTests(unittest.TestCase):
    def _environment(self, home: Path, *, codex_home: Path | None = None) -> dict[str, str]:
        environment = os.environ.copy()
        environment["HOME"] = str(home)
        environment["CODEX_HOME"] = str(codex_home or home / ".codex")
        return environment

    def _run(
        self,
        arguments: list[str],
        environment: dict[str, str],
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_ROOT / "manage_global_install.py"), *arguments],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

    def test_install_check_and_recoverable_uninstall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = self._environment(home)
            install = self._run(["install", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(install.returncode, 0, install.stderr)
            target = home / ".codex" / "skills" / "remote-gpu-dev"
            self.assertTrue((target / "SKILL.md").is_file())
            check = self._run(["check"], environment, timeout=30)
            self.assertEqual(check.returncode, 0, check.stderr)
            update = self._run(["update", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(update.returncode, 0, update.stderr)
            uninstall = self._run(["uninstall"], environment, timeout=30)
            self.assertEqual(uninstall.returncode, 0, uninstall.stderr)
            self.assertFalse(target.exists())
            self.assertTrue(any((home / ".codex" / "skill-backups").iterdir()))

    def test_distributable_skill_scripts_are_executable_and_not_group_writable(self) -> None:
        scripts = sorted(
            path
            for path in (SKILL_ROOT / "scripts").iterdir()
            if path.is_file() and path.suffix in {".py", ".sh"}
        )
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                self.assertEqual(stat.S_IMODE(script.stat().st_mode), 0o755)

    def test_source_validation_rejects_a_non_discoverable_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "remote-gpu-dev"
            shutil.copytree(SKILL_ROOT, source, ignore=shutil.ignore_patterns("__pycache__"))
            skill_md = source / "SKILL.md"
            lines = [
                line for line in skill_md.read_text(encoding="utf-8").splitlines()
                if not line.startswith("description:")
            ]
            skill_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            environment = self._environment(root / "home")
            invalid = self._run(["validate-source", "--source", str(source)], environment)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("description", invalid.stderr)
            install = self._run(["install", "--source", str(source)], environment)
            self.assertEqual(install.returncode, 2)
            self.assertFalse((root / "home" / ".codex" / "skills" / "remote-gpu-dev").exists())

    def test_install_normalizes_modes_and_check_detects_mode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "remote-gpu-dev"
            shutil.copytree(SKILL_ROOT, source, ignore=shutil.ignore_patterns("__pycache__"))
            os.chmod(source, 0o777)
            os.chmod(source / "SKILL.md", 0o666)
            os.chmod(source / "scripts" / "remote_gpu.py", 0o666)
            home = root / "home"
            home.mkdir()
            environment = self._environment(home)
            install = self._run(["install", "--source", str(source)], environment)
            self.assertEqual(install.returncode, 0, install.stderr)
            target = home / ".codex" / "skills" / "remote-gpu-dev"
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE((target / "SKILL.md").stat().st_mode), 0o644)
            command = target / "scripts" / "remote_gpu.py"
            self.assertEqual(stat.S_IMODE(command.stat().st_mode), 0o755)
            self.assertEqual(
                stat.S_IMODE((target / manage_global_install.MARKER).stat().st_mode), 0o600
            )
            self.assertEqual(self._run(["check"], environment).returncode, 0)
            os.chmod(command, 0o644)
            drift = self._run(["check"], environment)
            self.assertEqual(drift.returncode, 2)
            self.assertIn("digest", drift.stderr)

    def test_launcher_conflict_leaves_no_partial_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = self._environment(home)
            bin_dir = home / ".local" / "bin"
            bin_dir.mkdir(parents=True)
            conflict = bin_dir / "remote-gpu-dashboard"
            conflict.write_text("unrelated\n", encoding="utf-8")
            install = self._run(["install", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(install.returncode, 2)
            self.assertIn("unrelated launcher", install.stderr)
            self.assertFalse((bin_dir / "remote-gpu-dev").is_symlink())
            self.assertEqual(conflict.read_text(encoding="utf-8"), "unrelated\n")
            self.assertFalse((home / ".codex" / "skills" / "remote-gpu-dev").exists())

    def test_install_rolls_back_a_partially_committed_launcher_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = self._environment(home)
            real_replace = manage_global_install.os.replace

            def fail_second_launcher(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
                if Path(destination).name == "remote-gpu-dashboard":
                    raise OSError("injected launcher failure")
                real_replace(source, destination)

            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                manage_global_install.os, "replace", side_effect=fail_second_launcher
            ):
                with self.assertRaises(OSError):
                    manage_global_install.install(SKILL_ROOT, update=False)
            bin_dir = home / ".local" / "bin"
            self.assertFalse((bin_dir / "remote-gpu-dev").exists())
            self.assertFalse((bin_dir / "remote-gpu-dev").is_symlink())
            self.assertFalse((home / ".codex" / "skills" / "remote-gpu-dev").exists())

    def test_update_restores_old_install_when_stage_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = self._environment(home)
            install = self._run(["install", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(install.returncode, 0, install.stderr)
            target = home / ".codex" / "skills" / "remote-gpu-dev"
            original_marker = (target / manage_global_install.MARKER).read_bytes()
            real_replace = manage_global_install.os.replace

            def fail_stage_commit(
                source: os.PathLike[str], destination: os.PathLike[str]
            ) -> None:
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    destination_path == target
                    and source_path.name.startswith(".remote-gpu-dev.stage-")
                ):
                    raise OSError("injected stage commit failure")
                real_replace(source, destination)

            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                manage_global_install.os, "replace", side_effect=fail_stage_commit
            ):
                with self.assertRaises(OSError):
                    manage_global_install.install(SKILL_ROOT, update=True)

            self.assertTrue(target.is_dir())
            self.assertEqual(
                (target / manage_global_install.MARKER).read_bytes(), original_marker
            )
            self.assertTrue((home / ".local" / "bin" / "remote-gpu-dev").is_symlink())
            self.assertEqual(self._run(["check"], environment).returncode, 0)

    def test_uninstall_failure_restores_target_and_launchers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = self._environment(home)
            install = self._run(["install", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(install.returncode, 0, install.stderr)
            target = home / ".codex" / "skills" / "remote-gpu-dev"
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                manage_global_install, "remove_launchers", side_effect=OSError("injected")
            ):
                with self.assertRaises(OSError):
                    manage_global_install.uninstall()
            self.assertTrue(target.is_dir())
            self.assertTrue((home / ".local" / "bin" / "remote-gpu-dev").is_symlink())
            self.assertEqual(self._run(["check"], environment).returncode, 0)

    def test_symlinked_skills_parent_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root / "home"
            codex_home = home / ".codex"
            codex_home.mkdir(parents=True)
            real_skills = root / "real-skills"
            real_skills.mkdir()
            (codex_home / "skills").symlink_to(real_skills, target_is_directory=True)
            environment = self._environment(home, codex_home=codex_home)
            install = self._run(["install", "--source", str(SKILL_ROOT)], environment)
            self.assertEqual(install.returncode, 0, install.stderr)
            self.assertTrue((real_skills / "remote-gpu-dev" / "SKILL.md").is_file())
            check = self._run(["check"], environment)
            self.assertEqual(check.returncode, 0, check.stderr)
            uninstall = self._run(["uninstall"], environment)
            self.assertEqual(uninstall.returncode, 0, uninstall.stderr)
            for name in manage_global_install.COMMANDS:
                launcher = home / ".local" / "bin" / name
                self.assertFalse(launcher.exists())
                self.assertFalse(launcher.is_symlink())


if __name__ == "__main__":
    unittest.main()
