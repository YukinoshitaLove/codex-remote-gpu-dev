#!/usr/bin/env python3
"""Profile, routing, SSH, and offline-onboarding tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skills" / "remote-gpu-dev"
SCRIPT_ROOT = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import profile as profiles  # noqa: E402
import remote_gpu  # noqa: E402
import ssh_remote  # noqa: E402


def runtime_object(*pairs: tuple[str, object]) -> dict[str, object]:
    return dict(pairs)


class ProfileTests(unittest.TestCase):
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

    def profile(self, slug: str = "lab-gpu") -> dict:
        identity = self.root / "keys" / slug
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text("test-key-placeholder\n", encoding="utf-8")
        identity.chmod(0o600)
        gpu_seed = hashlib.sha256(slug.encode("ascii")).hexdigest()
        return profiles.default_profile(
            name="Lab GPU",
            slug=slug,
            host="gpu.example.test",
            user="researcher",
            port=2222,
            identity_file=str(identity),
            local_projects_root=str(self.root / "projects" / slug),
            ticket_root=str(self.root / "tickets" / slug),
            remote_temp_root=f"/scratch/remote-gpu-dev/{slug}",
            remote_durable_root=f"/data/remote-gpu-dev/{slug}",
            gpu_ids=[0, 1],
            conda_executable="/opt/conda/bin/conda",
            monitor_python=f"/data/remote-gpu-dev/{slug}/infra/monitor-env/bin/python",
            host_key_fingerprints=["SHA256:abcdefghijklmnopqrstuvwx"],
            remote_machine_id_sha256="sha256:"
            + hashlib.sha256(slug.encode("ascii")).hexdigest(),
            gpu_devices=[
                {
                    "index": 0,
                    "uuid": f"GPU-{gpu_seed[:32]}",
                    "name": "Example GPU",
                    "memory_mib": 24576,
                },
                {
                    "index": 1,
                    "uuid": f"GPU-{gpu_seed[32:]}",
                    "name": "Example GPU",
                    "memory_mib": 24576,
                },
            ],
        )

    def test_profile_round_trip_is_private_and_isolated(self) -> None:
        first = self.profile("lab-a")
        second = self.profile("lab-b")
        first_path = profiles.save_profile(first)
        second_path = profiles.save_profile(second)
        self.assertEqual(stat.S_IMODE(first_path.stat().st_mode), 0o600)
        self.assertNotEqual(first["local"]["ticket_root"], second["local"]["ticket_root"])
        self.assertNotEqual(
            profiles.dashboard_runtime_dir(first), profiles.dashboard_runtime_dir(second)
        )
        profiles.set_active_profile("lab-a")
        self.assertEqual(profiles.load_profile()["slug"], "lab-a")
        self.assertEqual(profiles.load_profile("lab-b")["slug"], "lab-b")

    def test_aliases_for_same_physical_server_share_dashboard_runtime(self) -> None:
        first = self.profile("lab-a")
        second = copy.deepcopy(first)
        second["slug"] = "lab-alias"
        second["name"] = "Alias"
        self.assertEqual(
            profiles.dashboard_runtime_dir(first), profiles.dashboard_runtime_dir(second)
        )
        profiles.save_profile(first)
        second["local"]["ticket_root"] = str(self.root / "tickets" / "alias")
        with self.assertRaises(profiles.ProfileError):
            profiles.save_profile(second)

    def test_secret_fields_paths_and_port_overlap_fail_closed(self) -> None:
        value = self.profile()
        value["ssh"]["".join(("pass", "word"))] = "forbidden-value"
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(value)
        value = self.profile()
        value["remote"]["temp_root"] = "/"
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(value)
        value = self.profile()
        value["remote"]["proxy_port"] = value["dashboard"][
            "tensorboard_remote_port_start"
        ]
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(value)

    def test_secret_values_fail_before_profile_persistence(self) -> None:
        github_oauth = "gho" + "_" + "C" * 24
        github_classic = "ghp" + "_" + "D" * 32
        secret_values = (
            github_oauth,
            "xox" + "b-" + "E" * 24,
            "AK" + "IA" + "F" * 16,
            "说明" + github_classic + "结束",
            "".join(("pass", "phrase=fixture-correct-horse")),
            "".join(("-----BEGIN ", "PRIVATE KEY----- ", "G" * 64)),
            "sk" + "-ant-api03-" + "H" * 48,
            "gl" + "pat-" + "I" * 20,
            "pypi-" + "AgEIcHlwaS5vcmc" + "J" * 32,
            "npm" + "_" + "K" * 36,
            "sk" + "_live_" + "L" * 24,
            "AI" + "za" + "M" * 35,
            "ya" + "29." + "N" * 40,
            "".join(("OPENAI", "_API_KEY=fixture-value")),
            "".join(("MLFLOW_TRACKING_PASS", "WORD=fixture-value")),
            "".join(("postgresql://fixture-user", ":fixture-password@db.invalid/run")),
            "".join(("redis://fixture-user", ":fixture-password@cache.invalid/0")),
            "".join(("ssh://fixture-user", ":fixture-password@gpu.invalid/home")),
            "".join(("https://", ":fixture-password@example.invalid/simple")),
            "".join(("x://fixture-user", ":fixture-password@host")),
            "".join(("//fixture-user", ":fixture-password@host/path")),
            "".join(("//", ":fixture-password@host/path")),
        )
        for index, secret_value in enumerate(secret_values):
            with self.subTest(index=index):
                value = self.profile(f"secret-{index}")
                value["gpu"]["environment"] = {"MODEL_ACCESS": secret_value}
                destination = profiles.profile_path(value["slug"])
                with self.assertRaisesRegex(profiles.ProfileError, "secret"):
                    profiles.save_profile(value)
                self.assertFalse(destination.exists())

    def test_secret_token_boundaries_are_ascii_aware(self) -> None:
        github_classic = "ghp" + "_" + "A" * 32
        for value in (
            github_classic,
            "说明" + github_classic,
            github_classic + "结束",
            "说明" + github_classic + "结束",
            "GHP" + "_" + "B" * 32,
            github_classic + "\n",
        ):
            with self.subTest(secret=value[:12]):
                self.assertTrue(profiles.contains_secret(value))

        for value in (
            "x" + github_classic,
            "hf-transformers-benchmark",
            "sk-learn-classification",
            "postgres" + "ql://db.invalid:5432/experiments",
            "redis" + "://cache.invalid:6379/0",
            "ssh" + "://fixture-user@gpu.invalid/home",
            "x://fixture-user@host",
            "x://fixture-user:@host",
            "x://:@host",
            "x://host:22/path",
            "//fixture-user@host/path",
            "//fixture-user:@host/path",
            "//:@host/path",
            "//host:443/path",
        ):
            with self.subTest(normal=value[:12]):
                self.assertFalse(profiles.contains_secret(value))

        profile = self.profile("normal-secret-prefix-names")
        profile["gpu"]["environment"] = {
            "MODEL_FAMILY": "hf-transformers-benchmark",
            "ESTIMATOR": "sk-learn-classification",
        }
        normalized = profiles.validate_profile(profile)
        self.assertEqual(normalized["gpu"]["environment"], profile["gpu"]["environment"])

    def test_private_jwk_and_obfuscated_credential_fields_are_rejected(self) -> None:
        private_jwks = [
            runtime_object(("kty", "EC"), ("crv", "P-256"), ("d", "private-fixture")),
            runtime_object(("kty", "OKP"), ("crv", "Ed25519"), ("d", "private-fixture")),
            runtime_object(("kty", "oct"), ("k", "symmetric-fixture")),
            *(
                runtime_object(("kty", "RSA"), (parameter, "private-fixture"))
                for parameter in ("d", "p", "q", "dp", "dq", "qi")
            ),
            runtime_object(
                ("kty", "RSA"),
                ("oth", [runtime_object(("r", "r"), ("d", "d"), ("t", "t"))]),
            ),
        ]
        private_jwk_strings = [
            json.dumps(private_jwks[0]),
            "\ufeff" + json.dumps(runtime_object(("kty", "oct"), ("k", "private-fixture"))),
            "\u200b" + json.dumps(runtime_object(("kty", "OKP"), ("d", "private-fixture"))),
        ]
        for index, jwk in enumerate(private_jwks):
            with self.subTest(private_jwk=index):
                candidate = self.profile(f"private-jwk-{index}")
                candidate["gpu"]["environment"] = {
                    "MODEL_METADATA": json.dumps(jwk)
                }
                with self.assertRaisesRegex(profiles.ProfileError, "secret|JWK"):
                    profiles.validate_profile(candidate)
        for index, rendered in enumerate(private_jwk_strings):
            with self.subTest(private_jwk_string=index):
                candidate = self.profile(f"private-jwk-text-{index}")
                candidate["gpu"]["environment"] = {"MODEL_METADATA": rendered}
                with self.assertRaisesRegex(profiles.ProfileError, "secret|JWK"):
                    profiles.validate_profile(candidate)

        for index, field in enumerate(
            (
                "SERVICE_ACCOUNT_KEY",
                "serviceAccountKey",
                "pass\u200bword",
            )
        ):
            with self.subTest(field=repr(field)):
                candidate = self.profile(f"structured-secret-{index}")
                candidate["ssh"][field] = "fixture-value"
                with self.assertRaisesRegex(profiles.ProfileError, "forbidden"):
                    profiles.validate_profile(candidate)

        public_jwks = (
            {"kty": "EC", "crv": "P-256", "x": "x-fixture", "y": "y-fixture"},
            {"kty": "OKP", "crv": "Ed25519", "x": "x-fixture"},
            {"kty": "RSA", "n": "modulus-fixture", "e": "AQAB"},
            {"kty": "ec", "d": "descriptive-dimension"},
            {"kty": "RSA", "p": None},
            {"kty": "oct", "k": ""},
        )
        candidate = self.profile("public-jwks")
        candidate["gpu"]["environment"] = {
            f"PUBLIC_JWK_{index}": json.dumps(jwk)
            for index, jwk in enumerate(public_jwks)
        }
        normalized = profiles.validate_profile(candidate)
        self.assertEqual(
            normalized["gpu"]["environment"], candidate["gpu"]["environment"]
        )

    def test_standard_secret_like_environment_names_remain_accepted(self) -> None:
        profile = self.profile("standard-token-settings")
        profile["gpu"]["environment"] = {
            "MAX_TOKEN_COUNT": "2048",
            "MAX_NEW_TOKENS": "256",
            "TOKEN_IDS": "input_ids",
            "TOKEN_EMBEDDING_DIMENSION": "4096",
            "TOKEN_BUDGET": "4096",
            "LOSS_TOKEN_WEIGHT": "0.25",
            "PAD_TOKEN": "pad",
            "EOS_TOKEN": "eos",
            "BOS_TOKEN": "bos",
            "MASK_TOKEN": "mask",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "true",
            "N_TOKENS": "1024",
            "TOKENS_PER_SECOND": "42.5",
            "PROMPT_TOKENS": "128",
            "COMPLETION_TOKENS": "64",
            "ADDITIONAL_SPECIAL_TOKENS": "<image>,<audio>",
            "SPECIAL_TOKENS_MAP": "tokenizer.json",
            "IMAGE_TOKEN_INDEX": "32000",
            "NUM_IMAGE_TOKENS": "576",
            "CLS_TOKEN": "[CLS]",
            "SEP_TOKEN": "[SEP]",
            "UNK_TOKEN": "[UNK]",
        }
        normalized = profiles.validate_profile(profile)
        self.assertEqual(normalized["gpu"]["environment"], profile["gpu"]["environment"])

        sensitive_fixture = "sk" + "-ant-api03-" + "M" * 48
        profile["gpu"]["environment"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = (
            sensitive_fixture
        )
        with self.assertRaisesRegex(profiles.ProfileError, "secret"):
            profiles.validate_profile(profile)

        credential_names = (
            "OPENAI" + "_API_KEY",
            "MLFLOW_TRACKING_" + "PASSWORD",
            "CUSTOM_ACCESS_" + "TOKEN",
            "ACCESS_" + "TOKENS",
            "CREDENTIALS",
        )
        for index, field in enumerate(credential_names):
            with self.subTest(field=field):
                candidate = self.profile(f"credential-field-{index}")
                candidate["gpu"]["environment"] = {field: "fixture-value"}
                with self.assertRaisesRegex(profiles.ProfileError, "forbidden"):
                    profiles.validate_profile(candidate)

        for index, field in enumerate(
            ("client" + "Secret", "auth" + "Tokens", "client" + "Credentials")
        ):
            with self.subTest(structured_field=field):
                candidate = self.profile(f"structured-credential-{index}")
                candidate["ssh"][field] = "fixture-value"
                with self.assertRaisesRegex(profiles.ProfileError, "forbidden"):
                    profiles.validate_profile(candidate)

        for assignment in (
            "".join(("OPENAI", "_API_KEY=fixture-value")),
            "".join(("MLFLOW_TRACKING_PASS", "WORD=fixture-value")),
            "".join(("CUSTOM_ACCESS_TO", "KEN=fixture-value")),
            "".join(("clientSec", "ret=fixture-value")),
            "".join(("authTo", "kens=fixture-value")),
            "".join(("clientCred", "entials=fixture-value")),
        ):
            with self.subTest(assignment=assignment[:16]):
                self.assertTrue(profiles.contains_secret(assignment))

        for assignment in (
            "MAX_TOKEN_COUNT=2048",
            "MAX_NEW_TOKENS=256",
            "TOKEN_IDS=input_ids",
            "TOKEN_EMBEDDING_DIMENSION=4096",
            "TOKEN_BUDGET=4096",
            "LOSS_TOKEN_WEIGHT=0.25",
            "PAD_TOKEN=pad",
            "EOS_TOKEN=eos",
            "BOS_TOKEN=bos",
            "MASK_TOKEN=mask",
            "TOKENIZERS_PARALLELISM=false",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN=true",
            "N_TOKENS=1024",
            "TOKENS_PER_SECOND=42.5",
            "PROMPT_TOKENS=128",
            "COMPLETION_TOKENS=64",
            "ADDITIONAL_SPECIAL_TOKENS=<image>,<audio>",
            "SPECIAL_TOKENS_MAP=tokenizer.json",
            "IMAGE_TOKEN_INDEX=32000",
            "NUM_IMAGE_TOKENS=576",
            "CLS_TOKEN=[CLS]",
            "SEP_TOKEN=[SEP]",
            "UNK_TOKEN=[UNK]",
            "SERVICE_ACCOUNT_KEY_ID=fixture-id",
            "SERVICE_ACCOUNT_KEY_COUNT=2",
            "SERVICE_ACCOUNT_KEY_ROTATION_INTERVAL=86400",
        ):
            with self.subTest(normal_assignment=assignment):
                self.assertFalse(profiles.contains_secret(assignment))

    def test_secret_scan_normalization_does_not_mutate_safe_profile_values(self) -> None:
        credential_fragments = (
            "".join(("pass", "word\n=fixture-value")),
            "".join(("Bear", "er\nabc123fixture")),
            "".join(("pass", "word\N{NO-BREAK SPACE}=fixture-value")),
            "".join(("Bear", "er\N{NO-BREAK SPACE}abc123fixture")),
            "".join(("client\\\nSec", "ret=fixture-value")),
            "".join(("pass", "word\u200b=fixture-value")),
        )
        for value in credential_fragments:
            with self.subTest(value=repr(value[:18])):
                self.assertTrue(profiles.contains_secret(value))

        candidate = self.profile("scan-view-preserves-value")
        original = "token budget\N{NO-BREAK SPACE}planning note"
        candidate["gpu"]["environment"] = {"RUN_NOTE": original}
        normalized = profiles.validate_profile(candidate)
        self.assertEqual(normalized["gpu"]["environment"]["RUN_NOTE"], original)

        blocked = self.profile("normalized-secret")
        blocked["gpu"]["environment"] = {
            "RUN_NOTE": "".join(("pass", "word\N{NO-BREAK SPACE}=fixture-value"))
        }
        destination = profiles.profile_path(blocked["slug"])
        with self.assertRaisesRegex(profiles.ProfileError, "secret"):
            profiles.save_profile(blocked)
        self.assertFalse(destination.exists())

    def test_private_key_detection_has_linear_growth_on_million_character_input(self) -> None:
        chunk = "-----BEGIN A "
        elapsed: list[float] = []
        for size in (100_000, 1_000_000):
            value = (chunk * (size // len(chunk) + 1))[:size]
            started = time.perf_counter()
            self.assertFalse(profiles.contains_secret(value))
            elapsed.append(time.perf_counter() - started)
        self.assertLess(elapsed[-1], 5.0)
        self.assertLess(elapsed[-1], max(elapsed[0], 0.005) * 25 + 0.25)

    def test_key_only_ssh_argv_has_strict_known_hosts(self) -> None:
        value = self.profile()
        known = Path(value["ssh"]["known_hosts_file"])
        known.parent.mkdir(parents=True, exist_ok=True)
        known.write_text("example.test ssh-ed25519 AAAATEST\n", encoding="utf-8")
        known.chmod(0o600)
        argv = ssh_remote.ssh_argv(value, batch=True)
        rendered = " ".join(argv)
        self.assertIn("PasswordAuthentication=no", rendered)
        self.assertIn("KbdInteractiveAuthentication=no", rendered)
        self.assertIn("StrictHostKeyChecking=yes", rendered)
        self.assertIn("BatchMode=yes", rendered)
        self.assertIn("ForwardAgent=no", rendered)
        self.assertIn("ForwardX11=no", rendered)
        self.assertIn("ForwardX11Trusted=no", rendered)
        self.assertIn("PermitLocalCommand=no", rendered)
        self.assertNotIn("sshpass", rendered)

    def test_delegated_options_are_preserved_verbatim(self) -> None:
        parsed = remote_gpu.parse_command_line(["--profile", "lab-a", "gpu", "--json"])
        self.assertEqual(parsed.profile, "lab-a")
        self.assertEqual(parsed.command, "gpu")
        self.assertEqual(parsed.arguments, ["--json"])
        parsed = remote_gpu.parse_command_line(
            ["ssh", "--proxy", "env HTTP_PROXY=http://127.0.0.1:17890 true"]
        )
        self.assertEqual(
            parsed.arguments,
            ["--proxy", "env HTTP_PROXY=http://127.0.0.1:17890 true"],
        )

    def test_host_key_route_uses_openssh_for_alias_or_proxyjump(self) -> None:
        alias_config = subprocess.CompletedProcess(
            ["ssh", "-G"],
            0,
            stdout="hostname 10.0.0.8\nproxyjump bastion\nproxycommand none\n",
            stderr="",
        )
        with mock.patch.object(remote_gpu.subprocess, "run", return_value=alias_config):
            self.assertTrue(remote_gpu._route_requires_openssh("gpu-alias", 22, None))
        direct_config = subprocess.CompletedProcess(
            ["ssh", "-G"],
            0,
            stdout="hostname gpu.example.test\nproxyjump none\nproxycommand none\n",
            stderr="",
        )
        with mock.patch.object(remote_gpu.subprocess, "run", return_value=direct_config):
            self.assertFalse(
                remote_gpu._route_requires_openssh("gpu.example.test", 22, None)
            )
        with (
            mock.patch.object(remote_gpu, "_route_requires_openssh", return_value=True),
            mock.patch.object(
                remote_gpu,
                "_scan_host_keys_via_openssh",
                return_value="gpu-alias ssh-ed25519 AAAATEST\n",
            ) as routed,
            mock.patch.object(
                remote_gpu,
                "_fingerprints_for_host_keys",
                return_value=["SHA256:abcdefghijklmnopqrstuvwx"],
            ),
            mock.patch.object(remote_gpu, "_scan_host_keys_direct") as direct,
        ):
            remote_gpu._scan_host_keys("gpu-alias", "alice", 2222, "bastion")
        routed.assert_called_once_with("gpu-alias", "alice", 2222, "bastion")
        direct.assert_not_called()

    def test_bootstrap_ssh_disables_agent_x11_and_local_commands(self) -> None:
        value = self.profile()
        known = Path(value["ssh"]["known_hosts_file"])
        known.parent.mkdir(parents=True, exist_ok=True)
        known.write_text("example.test ssh-ed25519 AAAATEST\n", encoding="utf-8")
        rendered = " ".join(
            remote_gpu._ssh_base(
                host=value["ssh"]["host"],
                user=value["ssh"]["user"],
                port=value["ssh"]["port"],
                identity=Path(value["ssh"]["identity_file"]),
                known_hosts=known,
                proxy_jump=None,
            )
        )
        for option in (
            "ForwardAgent=no",
            "ForwardX11=no",
            "ForwardX11Trusted=no",
            "PermitLocalCommand=no",
        ):
            self.assertIn(option, rendered)

    def test_scheduler_confirmation_fails_closed(self) -> None:
        probe = {"tools": {"scontrol": "missing", "qstat": "missing"}}
        with mock.patch.object(remote_gpu, "ask", return_value="kubernetes"):
            with self.assertRaises(remote_gpu.SetupError):
                remote_gpu._require_no_external_scheduler(probe)
        with mock.patch.object(remote_gpu, "ask", return_value="none"):
            remote_gpu._require_no_external_scheduler(probe)
        with self.assertRaises(remote_gpu.SetupError):
            remote_gpu._require_no_external_scheduler(
                {"tools": {"scontrol": "/usr/bin/scontrol", "qstat": "missing"}}
            )

    def test_mig_and_exact_gpu_mapping_fail_closed(self) -> None:
        expected = self.profile()["gpu"]["devices"]
        actual = copy.deepcopy(expected)
        actual[0]["uuid"], actual[1]["uuid"] = actual[1]["uuid"], actual[0]["uuid"]
        ready, _detail = remote_gpu._exact_managed_gpu_mapping(expected, actual)
        self.assertFalse(ready)
        ready, _detail, policy = remote_gpu._mig_readiness(
            {
                "gpus": expected,
                "mig": {0: "disabled", 1: "disabled"},
                "mig_error": None,
            }
        )
        self.assertTrue(ready)
        self.assertEqual(policy, "disabled")
        ready, _detail, policy = remote_gpu._mig_readiness(
            {
                "gpus": expected,
                "mig": {0: "N/A", 1: "Not Supported"},
                "mig_error": None,
            }
        )
        self.assertTrue(ready)
        self.assertEqual(policy, "unsupported")
        for probe in (
            {"gpus": expected, "mig": {}, "mig_error": "query-failed"},
            {"gpus": expected, "mig": {0: "disabled"}, "mig_error": None},
            {
                "gpus": expected,
                "mig": {0: "enabled", 1: "disabled"},
                "mig_error": None,
            },
        ):
            ready, _detail, _policy = remote_gpu._mig_readiness(probe)
            self.assertFalse(ready)

    def test_dashboard_and_tensorboard_port_collisions_fail_closed(self) -> None:
        remote_gpu._validate_dashboard_port_plan(
            dashboard_port=16006,
            proxy_enabled=True,
            proxy_local_port=7890,
            proxy_remote_port=17890,
            tensorboard_port_start=16006,
            tensorboard_port_end=16105,
        )
        invalid = (
            dict(
                dashboard_port=8765,
                proxy_enabled=True,
                proxy_local_port=8765,
                proxy_remote_port=17890,
                tensorboard_port_start=16006,
                tensorboard_port_end=16105,
            ),
            dict(
                dashboard_port=8765,
                proxy_enabled=True,
                proxy_local_port=7890,
                proxy_remote_port=16050,
                tensorboard_port_start=16006,
                tensorboard_port_end=16105,
            ),
            dict(
                dashboard_port=8765,
                proxy_enabled=False,
                proxy_local_port=7890,
                proxy_remote_port=17890,
                tensorboard_port_start=16105,
                tensorboard_port_end=16006,
            ),
        )
        for values in invalid:
            with self.assertRaises(remote_gpu.SetupError):
                remote_gpu._validate_dashboard_port_plan(**values)

    def test_offline_import_initializes_ticket_ledger(self) -> None:
        value = self.profile()
        source = self.root / "profile.json"
        source.write_text(json.dumps(value), encoding="utf-8")
        environment = os.environ.copy()
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "remote_gpu.py"),
                "setup",
                "--from-json",
                str(source),
                "--offline",
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        ticket_root = Path(value["local"]["ticket_root"])
        self.assertTrue((ticket_root / "config.json").is_file())
        self.assertTrue((ticket_root / "state.json").is_file())
        self.assertTrue((ticket_root / "BOARD.md").is_file())
        status = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "remote_gpu.py"),
                "ticket",
                "status",
                "--json",
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertEqual(payload["profile"], value["slug"])
        self.assertEqual(payload["occupied_gpus"], [])


if __name__ == "__main__":
    unittest.main()
