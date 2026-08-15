#!/usr/bin/env python3
"""Security and cross-profile coordination contract regressions."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "skills" / "remote-gpu-dev" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import git_project  # noqa: E402
import profile as profiles  # noqa: E402
import ssh_remote  # noqa: E402


PUBLIC_SPEC = importlib.util.spec_from_file_location(
    "security_public_tree", REPO_ROOT / "tools" / "check_public_tree.py"
)
assert PUBLIC_SPEC and PUBLIC_SPEC.loader
public_tree = importlib.util.module_from_spec(PUBLIC_SPEC)
PUBLIC_SPEC.loader.exec_module(public_tree)


class SecurityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(
            os.environ,
            {
                "REMOTE_GPU_DEV_HOME": str(self.root / "config"),
                "REMOTE_GPU_DEV_STATE_HOME": str(self.root / "state"),
            },
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def profile(self, slug: str = "primary") -> dict:
        identity = self.root / "keys" / "shared-ed25519"
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text("fixture key path only\n", encoding="utf-8")
        identity.chmod(0o600)
        return profiles.default_profile(
            name="Fixture GPU",
            slug=slug,
            host="gpu.example.test",
            user="researcher",
            port=22,
            identity_file=str(identity),
            local_projects_root=str(self.root / "projects"),
            ticket_root=str(self.root / "tickets"),
            remote_temp_root="/scratch/remote-gpu-dev/shared",
            remote_durable_root="/data/remote-gpu-dev/shared",
            gpu_ids=[0, 1],
            conda_executable="/opt/conda/bin/conda",
            monitor_python="/data/remote-gpu-dev/shared/infra/monitor/bin/python",
            host_key_fingerprints=["SHA256:abcdefghijklmnopqrstuvwx"],
            remote_machine_id_sha256="sha256:" + "1" * 64,
            gpu_devices=[
                {
                    "index": 0,
                    "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                    "name": "Fixture GPU",
                    "memory_mib": 24576,
                },
                {
                    "index": 1,
                    "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "name": "Fixture GPU",
                    "memory_mib": 24576,
                },
            ],
        )

    def rotated_alias(self, profile: dict) -> dict:
        alias = copy.deepcopy(profile)
        alias["slug"] = "rotated-alias"
        alias["name"] = "Rotated SSH Alias"
        alias["ssh"]["host"] = "gpu-rotated.example.test"
        alias["ssh"]["known_hosts_file"] = str(
            profiles.known_hosts_path(alias["slug"])
        )
        alias["trust"]["host_key_fingerprints"] = [
            "SHA256:zyxwvutsrqponmlkjihgfedc"
        ]
        alias["trust"]["remote_machine_id_sha256"] = "sha256:" + "2" * 64
        alias["trust"]["server_uid"] = profiles.compute_server_uid(
            alias["trust"]["host_key_fingerprints"],
            alias["trust"]["remote_machine_id_sha256"],
        )
        return profiles.validate_profile(alias)

    def test_host_key_rotation_alias_shares_dashboard_and_live_ledger(self) -> None:
        primary = self.profile()
        alias = self.rotated_alias(primary)
        self.assertNotEqual(primary["trust"]["server_uid"], alias["trust"]["server_uid"])
        self.assertEqual(
            primary["trust"]["coordination_uid"],
            alias["trust"]["coordination_uid"],
        )
        self.assertEqual(
            profiles.dashboard_runtime_dir(primary), profiles.dashboard_runtime_dir(alias)
        )
        profiles.save_profile(primary)
        profiles.save_profile(alias)

        ticket_root = Path(primary["local"]["ticket_root"])
        ticket_root.mkdir(parents=True)
        (ticket_root / "config.json").write_text(
            json.dumps(profiles.ticket_config(primary)), encoding="utf-8"
        )
        environment = os.environ.copy()
        environment[profiles.PROFILE_ENV] = primary["slug"]
        reserve = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "gpu_ticket.py"),
                "reserve",
                "--project",
                "fixture",
                "--owner",
                "tester",
                "--purpose",
                "alias ledger regression",
                "--gpus",
                "1",
                "--expected",
                "10m",
                "--json",
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(reserve.returncode, 0, reserve.stderr)

        environment[profiles.PROFILE_ENV] = alias["slug"]
        status = subprocess.run(
            [sys.executable, str(SCRIPT_ROOT / "gpu_ticket.py"), "status", "--json"],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["profile"], alias["slug"])
        self.assertEqual(payload["ledger_profile"], primary["slug"])
        self.assertEqual(payload["coordination_uid"], primary["trust"]["coordination_uid"])
        self.assertEqual(len(payload["tickets"]), 1)

    def test_overlapping_gpu_or_alias_policy_drift_is_rejected(self) -> None:
        primary = self.profile()
        profiles.save_profile(primary)
        overlap = copy.deepcopy(primary)
        overlap["slug"] = "overlap"
        overlap["ssh"]["known_hosts_file"] = str(profiles.known_hosts_path("overlap"))
        overlap["local"]["ticket_root"] = str(self.root / "other-tickets")
        overlap["gpu"]["devices"][1]["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        overlap["trust"]["coordination_uid"] = profiles.compute_coordination_uid(
            [device["uuid"] for device in overlap["gpu"]["devices"]]
        )
        with self.assertRaises(profiles.ProfileError):
            profiles.save_profile(overlap)

        drifted_alias = self.rotated_alias(primary)
        drifted_alias["gpu"]["heartbeat_grace_minutes"] += 1
        with self.assertRaises(profiles.ProfileError):
            profiles.save_profile(drifted_alias)

    def test_ssh_argv_disables_ambient_agent_x11_and_local_commands(self) -> None:
        profile = self.profile()
        known_hosts = Path(profile["ssh"]["known_hosts_file"])
        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        known_hosts.write_text("fixture ssh-ed25519 AAAAFIXTURE\n", encoding="utf-8")
        known_hosts.chmod(0o600)
        rendered = " ".join(ssh_remote.ssh_argv(profile, batch=True))
        for option in (
            "ForwardAgent=no",
            "ForwardX11=no",
            "ForwardX11Trusted=no",
            "PermitLocalCommand=no",
            "RemoteCommand=none",
            "ControlMaster=no",
            "ControlPath=none",
            "ControlPersist=no",
            "GlobalKnownHostsFile=/dev/null",
            "UpdateHostKeys=no",
            "VerifyHostKeyDNS=no",
        ):
            self.assertIn(option, rendered)

    def test_git_deployment_rejects_ignored_runtime_files(self) -> None:
        repository = self.root / "projects" / "fixture"
        repository.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.email", "test@example.test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
        )
        (repository / ".gitignore").write_text("runtime/\n", encoding="utf-8")
        (repository / "train.py").write_text("print('fixture')\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repository), "add", ".gitignore", "train.py"], check=True
        )
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True
        )
        (repository / "runtime").mkdir()
        (repository / "runtime" / "events.out").write_text("runtime\n", encoding="utf-8")
        contract = {
            "git": {
                "allow_lfs": False,
                "allow_submodules": False,
                "max_tracked_file_mib": 10,
            }
        }
        with self.assertRaisesRegex(git_project.ProjectError, "ignored files"):
            git_project._validate_repository(contract, repository)

    def test_git_deployment_rejects_tracked_symlink_escape(self) -> None:
        repository = self.root / "symlink-project"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.email", "test@example.test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
        )
        (repository / "external.py").symlink_to(self.root / "outside.py")
        subprocess.run(["git", "-C", str(repository), "add", "external.py"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qm", "symlink"], check=True
        )
        contract = {
            "git": {
                "allow_lfs": False,
                "allow_submodules": False,
                "max_tracked_file_mib": 10,
            }
        }
        with self.assertRaisesRegex(git_project.ProjectError, "tracked symlink"):
            git_project._validate_repository(contract, repository)

    def test_private_public_tree_denylist_stays_outside_release(self) -> None:
        release = self.root / "release"
        release.mkdir()
        marker = "private-fixture-" + secrets.token_hex(12)
        (release / "README.md").write_text(marker + "\n", encoding="utf-8")
        deny_file = self.root / "private-deny.txt"
        deny_file.write_text(marker + "\n", encoding="utf-8")
        deny_file.chmod(0o600)
        values = public_tree.load_local_deny(deny_file, release)
        self.assertTrue(public_tree.check(release, local_deny=values))
        self.assertEqual(stat.S_IMODE(deny_file.stat().st_mode), 0o600)
        with self.assertRaises(ValueError):
            public_tree.load_local_deny(release / "README.md", release)

    def test_compatibility_first_evolution_contract_is_published(self) -> None:
        expected_fragments = {
            "skills/remote-gpu-dev/SKILL.md": (
                "evolvable development tool, not a frozen policy",
                "training, testing,\ninference, CUDA/NCCL/DDP, DataLoader workers, compilation",
                "fix the narrowest responsible\nSkill instruction, helper, or guard",
                "not universal paths hard-coded by the Skill",
                "the AI is authorized to relax that\nrestriction directly",
                "tests, documentation, and release metadata",
            ),
            "README.md": (
                "Compatibility-first development contract",
                "short, out-of-the-box path for normal training, testing, inference",
                "trusted-code\nworkflow, not hostile-code containment",
                "the AI may directly relax it in the\nminimum necessary scope",
            ),
            "README.zh-CN.md": (
                "兼容性优先的开发契约",
                "开箱可用且操作路径短",
                "不是写死的策略",
                "AI 可以直接在最小必要范围内放宽该限制",
                "旧限制不是不可突破的规则",
            ),
            "SECURITY.md": (
                "Compatibility is the highest-priority runtime requirement",
                "not hostile-code\n  containment",
                "Profile roots are configurable, not hard-coded",
                "the AI may directly\n  relax it in the minimum necessary scope",
            ),
            "skills/remote-gpu-dev/references/execution.md": (
                "Compatibility and evolution rule",
                "Keep the normal path short",
                "update it locally, and add a focused\nregression test",
                "the AI may directly relax it in the\nminimum necessary scope",
                "Never treat an older restriction as immutable",
            ),
        }
        for relative, fragments in expected_fragments.items():
            content = (REPO_ROOT / relative).read_text(encoding="utf-8")
            for fragment in fragments:
                with self.subTest(document=relative, fragment=fragment):
                    self.assertIn(fragment, content)

        metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["version"], "0.2.4")

    def test_public_visual_assets_are_bilingual_relative_and_documented(self) -> None:
        expected_assets = [
            "docs/assets/diagrams/system-workflow.png",
            "docs/assets/diagrams/ticket-system.png",
            "docs/assets/screenshots/setup-wizard-simulation.png",
            "docs/assets/screenshots/dashboard-overview.png",
            "docs/assets/screenshots/dashboard-scratch20-tensorboard.png",
        ]

        def markdown_images(content: str) -> tuple[list[str], list[str]]:
            lines = content.splitlines()
            paths: list[str] = []
            explanations: list[str] = []
            for index, line in enumerate(lines):
                if not line.startswith("![") or "](" not in line or not line.endswith(")"):
                    continue
                path = line.rsplit("](", 1)[1][:-1]
                paths.append(path)
                following = next(
                    (candidate for candidate in lines[index + 1 :] if candidate.strip()), ""
                )
                explanations.append(following)
            return paths, explanations

        english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (REPO_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        english_assets, english_explanations = markdown_images(english)
        chinese_assets, chinese_explanations = markdown_images(chinese)
        self.assertEqual(english_assets, expected_assets)
        self.assertEqual(chinese_assets, expected_assets)
        self.assertEqual(english_assets, chinese_assets)
        self.assertTrue(
            all(line.startswith("*What this ") for line in english_explanations)
        )
        self.assertTrue(
            all(line.startswith("*这张") for line in chinese_explanations)
        )

        for relative in expected_assets:
            with self.subTest(asset=relative):
                self.assertFalse(relative.startswith(("/", "http:", "https:")))
                self.assertFalse(Path(relative).is_absolute())
                self.assertTrue((REPO_ROOT / relative).is_file())

        provenance = (REPO_ROOT / "docs" / "assets" / "README.md").read_text(
            encoding="utf-8"
        )
        for relative in expected_assets:
            with self.subTest(documented=relative):
                self.assertIn(relative, provenance)

    def test_control_channel_retry_contract_is_published(self) -> None:
        expected_fragments = {
            "skills/remote-gpu-dev/SKILL.md": (
                "only after five consecutive\n   structured control checks fail",
                "failures one through four only emit a warning\n   and retry",
                "any successful structured check resets the count",
                "never authorizes stop or\n   ticket release",
                "exact tracked-process identity,\n   assigned-GPU state",
                "sidecar preflight, status, and exact-absence\nchecks are idempotent",
                "Launch and stop are single-attempt mutations",
                "generation-fenced as `cleanup_pending`",
            ),
            "skills/remote-gpu-dev/references/execution.md": (
                "mark the control channel\n`unavailable` after five consecutive failures",
                "Failures one through four emit a\nwarning and retry",
                "successful structured check resets the consecutive-failure\ncount",
                "must never\ntrigger stop or ticket release",
                "exact\ntracked-process identity, assigned-GPU state",
            ),
            "README.md": (
                "only after five consecutive structured control checks\nfail",
                "Failures one through four only warn and retry",
                "successful check resets\nthe count",
                "never permits\nstop or ticket release",
                "sidecar retries only idempotent read-only preflight, status",
                "Launch and stop are never blindly replayed",
                "generation remains fenced as `cleanup_pending`",
            ),
            "README.zh-CN.md": (
                "只有连续 5 次结构化控制检查失败",
                "第 1 至 4 次只告警并重试",
                "连续失败计数清零",
                "不能在缺少精确进程、GPU 和最终状态证据时触发\nstop 或释放工单",
                "幂等只读的\npreflight、status 和精确 absence 检查重试",
                "结果未知的 launch 和 stop 绝不盲目重放",
                "同一 generation 会保持\n`cleanup_pending`",
            ),
            "skills/remote-gpu-dev/references/dashboard-and-tensorboard.md": (
                "preflight, status, and exact-absence checks retry only SSH exit\n255",
                "at most five consecutive attempts",
                "each attempt is capped at six seconds",
                "Launch and stop change remote process state and therefore run exactly once",
                "preserves the same generation and records `cleanup_pending`",
            ),
        }
        for relative, fragments in expected_fragments.items():
            content = (REPO_ROOT / relative).read_text(encoding="utf-8")
            for fragment in fragments:
                with self.subTest(document=relative, fragment=fragment):
                    self.assertIn(fragment, content)


if __name__ == "__main__":
    unittest.main()
