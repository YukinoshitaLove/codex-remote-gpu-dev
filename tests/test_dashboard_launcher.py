#!/usr/bin/env python3
"""Mechanical tests for the user-facing dashboard launcher install."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "remote-gpu-dev"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load(
    "remote_gpu_dashboard_launcher",
    SKILL_ROOT / "scripts" / "remote_gpu_dashboard.py",
)
installer = load(
    "remote_gpu_dashboard_installer",
    SKILL_ROOT / "scripts" / "install_dashboard_launcher.py",
)


class DashboardLauncherTests(unittest.TestCase):
    def test_bare_command_opens_verified_singleton(self) -> None:
        self.assertEqual(launcher.dashboard_arguments([]), ["ensure", "--open"])
        self.assertEqual(launcher.dashboard_arguments(["open"]), ["ensure", "--open"])
        self.assertEqual(launcher.dashboard_arguments(["status"]), ["status"])
        self.assertEqual(launcher.dashboard_arguments(["stop"]), ["stop"])
        self.assertEqual(
            launcher.dashboard_arguments(["ensure", "--open"]),
            ["ensure", "--open"],
        )
        with self.assertRaises(ValueError):
            launcher.dashboard_arguments(["unknown"])

    def test_install_is_user_local_idempotent_and_conflict_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            command, desktop = installer.install(home)
            self.assertTrue(command.is_symlink())
            self.assertEqual(command.resolve(), installer.LAUNCHER)
            self.assertIn(
                installer.OWNERSHIP_MARKER,
                desktop.read_text(encoding="utf-8"),
            )
            self.assertEqual(installer.install(home), (command, desktop))
            self.assertEqual(installer.check(home), (command, desktop))

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            command, _ = installer.install_paths(home)
            command.parent.mkdir(parents=True)
            command.write_text("unrelated\n", encoding="utf-8")
            with self.assertRaises(installer.InstallError):
                installer.install(home)


if __name__ == "__main__":
    unittest.main()
