#!/usr/bin/env python3
"""Offline regressions for the two-root remote filesystem boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "skills" / "remote-gpu-dev" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import managed_run  # noqa: E402
import infra_tools  # noqa: E402
import profile as profiles  # noqa: E402
import remote_gpu  # noqa: E402
import remote_path_guard as guard  # noqa: E402
import ssh_remote  # noqa: E402


class RemoteFilesystemGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        identity = root / "id"
        identity.write_text("fixture\n", encoding="utf-8")
        identity.chmod(0o600)
        self.profile = profiles.default_profile(
            name="Guard fixture",
            slug="guard-fixture",
            host="gpu.example.test",
            user="researcher",
            port=22,
            identity_file=str(identity),
            local_projects_root=str(root / "projects"),
            ticket_root=str(root / "tickets"),
            remote_temp_root="/scratch/remote-gpu-dev/guard",
            remote_durable_root="/data/remote-gpu-dev/guard",
            gpu_ids=[0],
            conda_executable="/opt/miniforge/bin/conda",
            monitor_python="/data/remote-gpu-dev/guard/infra/monitor/bin/python",
            host_key_fingerprints=["SHA256:abcdefghijklmnopqrstuvwx"],
            remote_machine_id_sha256="sha256:" + hashlib.sha256(b"guard").hexdigest(),
            gpu_devices=[{
                "index": 0,
                "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "name": "Fixture GPU",
                "memory_mib": 24576,
            }],
        )

    def test_asset_paths_must_stay_in_exact_two_roots(self) -> None:
        self.assertEqual(
            guard.require_managed_remote_path(
                self.profile, "/scratch/remote-gpu-dev/guard/runs/a"
            ),
            "/scratch/remote-gpu-dev/guard/runs/a",
        )
        for path in ("/tmp/a", "/home/researcher/a", "/data/other", "/"):
            with self.subTest(path=path), self.assertRaises(guard.RemotePathError):
                guard.require_managed_remote_path(self.profile, path)

    def test_profile_rejects_derived_path_escape_and_protected_environment(self) -> None:
        escaped = copy.deepcopy(self.profile)
        escaped["remote"]["records_root"] = "/tmp/records"
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(escaped)
        protected = copy.deepcopy(self.profile)
        protected["gpu"]["environment"]["HOME"] = "/tmp"
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(protected)
        protected_pip = copy.deepcopy(self.profile)
        protected_pip["gpu"]["environment"]["PIP_INDEX_URL"] = "https://wrong.invalid/simple"
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_profile(protected_pip)

    def test_all_runtime_asset_values_are_under_temp_except_explicit_null_config(self) -> None:
        temporary = self.profile["remote"]["temp_root"]
        environment = guard.managed_runtime_environment(self.profile)
        for key, value in environment.items():
            if key in {"CONDA_NO_PLUGINS", "PYTHONNOUSERSITE", "PIP_CONFIG_FILE"}:
                continue
            with self.subTest(key=key):
                self.assertTrue(value == temporary or value.startswith(temporary + "/"))
        self.assertEqual(environment["PIP_CONFIG_FILE"], "/dev/null")
        self.assertIn("TMUX_TMPDIR", environment)

    def test_infrastructure_landlock_defaults_strict_and_gpu_mode_is_explicit(self) -> None:
        code = managed_run.LANDLOCK_RUNNER
        self.assertIn("if strict_isolation and abi < 5", code)
        self.assertNotIn('add_path("/dev")', code)
        self.assertIn('add_path("/dev/shm", writable=True)', code)
        self.assertIn('add_path("/dev/ptmx", device=True)', code)
        self.assertIn('add_path("/dev/pts", device=True)', code)
        self.assertIn('"/run"', code)
        self.assertIn('"/proc"', code)
        self.assertIn('required = managed(raw, "required file", canonical=True)', code)
        strict_command = managed_run.build_landlock_command(
            self.profile,
            ["/usr/bin/python3", "-c", "pass"],
            workdir="/scratch/remote-gpu-dev/guard",
        )
        compatible_command = managed_run.build_landlock_command(
            self.profile,
            ["/scratch/remote-gpu-dev/guard/env/bin/python", "train.py"],
            workdir="/scratch/remote-gpu-dev/guard/run",
            required_files=[
                "/scratch/remote-gpu-dev/guard/projects/demo/train.py"
            ],
            strict_isolation=False,
        )
        self.assertTrue(json.loads(shlex.split(strict_command)[-1])["strict_isolation"])
        self.assertFalse(json.loads(shlex.split(strict_command)[-1])["allow_pty"])
        pty_command = managed_run.build_landlock_command(
            self.profile,
            ["/usr/bin/python3", "-c", "pass"],
            workdir="/scratch/remote-gpu-dev/guard",
            allow_pty=True,
        )
        self.assertTrue(json.loads(shlex.split(pty_command)[-1])["allow_pty"])
        compatible_spec = json.loads(shlex.split(compatible_command)[-1])
        self.assertFalse(compatible_spec["strict_isolation"])
        self.assertEqual(
            compatible_spec["required_files"],
            ["/scratch/remote-gpu-dev/guard/projects/demo/train.py"],
        )
        with self.assertRaisesRegex(
            managed_run.ManagedRunError, "allow_pty must be a boolean"
        ):
            managed_run.build_landlock_command(
                self.profile,
                ["/usr/bin/python3", "-c", "pass"],
                workdir="/scratch/remote-gpu-dev/guard",
                allow_pty=1,  # type: ignore[arg-type]
            )
        environment = managed_run.build_spec(
            self.profile,
            {"id": "GPU-fixture", "assigned_gpus": [0], "remote_workdir": "/scratch/remote-gpu-dev/guard/run"},
            workdir="/scratch/remote-gpu-dev/guard/run",
            env_prefix="/scratch/remote-gpu-dev/guard/env",
            script="/scratch/remote-gpu-dev/guard/project/train.py",
            arguments=[],
        )["environment"]
        self.assertNotIn("NCCL_SHM_DISABLE", environment)

    def test_script_and_module_targets_use_a_separate_ticket_workdir(self) -> None:
        ticket = {
            "id": "GPU-fixture",
            "assigned_gpus": [0],
            "remote_workdir": "/scratch/remote-gpu-dev/guard/runs/demo",
        }
        script = "/scratch/remote-gpu-dev/guard/projects/demo/train.py"
        script_spec = managed_run.build_spec(
            self.profile,
            ticket,
            workdir=ticket["remote_workdir"],
            env_prefix="/scratch/remote-gpu-dev/guard/envs/demo",
            script=script,
            arguments=["--epochs", "2"],
        )
        self.assertEqual(
            script_spec["argv"],
            [
                "/scratch/remote-gpu-dev/guard/envs/demo/bin/python",
                script,
                "--epochs",
                "2",
            ],
        )
        self.assertEqual(script_spec["required_files"], [script])

        module_spec = managed_run.build_spec(
            self.profile,
            ticket,
            workdir=ticket["remote_workdir"],
            env_prefix="/scratch/remote-gpu-dev/guard/envs/demo",
            module="torch.distributed.run",
            arguments=["--nproc-per-node", "4", script],
        )
        self.assertEqual(
            module_spec["argv"][:3],
            [
                "/scratch/remote-gpu-dev/guard/envs/demo/bin/python",
                "-m",
                "torch.distributed.run",
            ],
        )
        self.assertEqual(module_spec["required_files"], [])
        for kwargs in (
            {"script": script, "module": "torch.distributed.run"},
            {"script": None, "module": None},
            {"script": None, "module": "torch-distributed.run"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                managed_run.ManagedRunError
            ):
                managed_run.build_spec(
                    self.profile,
                    ticket,
                    workdir=ticket["remote_workdir"],
                    env_prefix="/scratch/remote-gpu-dev/guard/envs/demo",
                    arguments=[],
                    **kwargs,
                )

    def test_python_arguments_reject_obvious_path_escapes(self) -> None:
        ticket = {
            "id": "GPU-fixture",
            "assigned_gpus": [0],
            "remote_workdir": "/scratch/remote-gpu-dev/guard/runs/demo",
        }
        common = {
            "profile": self.profile,
            "ticket": ticket,
            "workdir": ticket["remote_workdir"],
            "env_prefix": "/scratch/remote-gpu-dev/guard/envs/demo",
            "script": "/scratch/remote-gpu-dev/guard/projects/demo/train.py",
        }
        accepted = managed_run.build_spec(
            arguments=[
                "--output=/data/remote-gpu-dev/guard/results/demo.json",
                "/scratch/remote-gpu-dev/guard/data/images",
                "--epochs",
                "2",
            ],
            **common,
        )
        self.assertEqual(accepted["argv"][-4:], [
            "--output=/data/remote-gpu-dev/guard/results/demo.json",
            "/scratch/remote-gpu-dev/guard/data/images",
            "--epochs",
            "2",
        ])
        for arguments in (
            ["--output=/tmp/result.pt"],
            ["/etc/passwd"],
            ["../outside"],
            ["--name=bad\nvalue"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(
                (managed_run.ManagedRunError, guard.RemotePathError)
            ):
                managed_run.build_spec(arguments=arguments, **common)

    def test_ticket_workdir_and_control_session_are_inferred_but_not_overridden(self) -> None:
        ticket = {
            "id": "GPU-fixture",
            "session": "exact-session",
            "remote_workdir": "/scratch/remote-gpu-dev/guard/runs/demo",
            "assigned_gpus": [0],
        }
        self.assertEqual(
            managed_run._ticket_workdir(self.profile, ticket, None),
            ticket["remote_workdir"],
        )
        self.assertEqual(
            managed_run._ticket_workdir(
                self.profile, ticket, ticket["remote_workdir"]
            ),
            ticket["remote_workdir"],
        )
        self.assertEqual(managed_run._ticket_session(ticket, None), "exact-session")
        self.assertEqual(
            managed_run._ticket_session(ticket, "exact-session"), "exact-session"
        )
        with self.assertRaises(managed_run.ManagedRunError):
            managed_run._ticket_workdir(
                self.profile, ticket, "/scratch/remote-gpu-dev/guard/runs/other"
            )
        with self.assertRaises(managed_run.ManagedRunError):
            managed_run._ticket_session(ticket, "other-session")

    def test_gpu_execute_has_one_compatible_mode_and_infers_workdir(self) -> None:
        ticket = {
            "id": "GPU-fixture",
            "status": "running",
            "session": "exact-session",
            "remote_workdir": "/scratch/remote-gpu-dev/guard/runs/demo",
            "assigned_gpus": [0],
        }
        args = managed_run.parse_command_line([
            ticket["id"],
            "--env-prefix", "/scratch/remote-gpu-dev/guard/envs/demo",
            "--script", "/scratch/remote-gpu-dev/guard/projects/demo/train.py",
            "--", "--epochs", "2",
        ])
        completed = subprocess.CompletedProcess(["ssh"], 0)
        with (
            mock.patch.object(managed_run, "_ticket", return_value=ticket),
            mock.patch.object(
                managed_run, "build_landlock_command", return_value="structured"
            ) as builder,
            mock.patch.object(managed_run, "ssh_argv", return_value=["ssh"]),
            mock.patch.object(managed_run.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(managed_run.execute(self.profile, args), 0)
        self.assertEqual(builder.call_args.kwargs["workdir"], ticket["remote_workdir"])
        self.assertFalse(builder.call_args.kwargs["strict_isolation"])
        self.assertEqual(
            builder.call_args.kwargs["required_files"],
            ["/scratch/remote-gpu-dev/guard/projects/demo/train.py"],
        )

        status_args = managed_run.parse_command_line([ticket["id"], "--status"])
        with (
            mock.patch.object(managed_run, "_ticket", return_value=ticket),
            mock.patch.object(managed_run, "_control_job", return_value=0) as control,
        ):
            self.assertEqual(managed_run.execute(self.profile, status_args), 0)
        control.assert_called_once_with(
            self.profile,
            ticket,
            session="exact-session",
            action="status",
            workdir=None,
        )

    def test_run_injects_profile_primary_and_extra_pip_indexes(self) -> None:
        profile = copy.deepcopy(self.profile)
        profile["network"]["pip_index_url"] = "https://primary.example.test/simple"
        profile["network"]["pip_extra_index_urls"] = [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://secondary.example.test/simple",
        ]
        environment = managed_run.build_spec(
            profile,
            {
                "id": "GPU-fixture",
                "assigned_gpus": [0],
                "remote_workdir": "/scratch/remote-gpu-dev/guard/run",
            },
            workdir="/scratch/remote-gpu-dev/guard/run",
            env_prefix="/scratch/remote-gpu-dev/guard/env",
            script="/scratch/remote-gpu-dev/guard/run/train.py",
            arguments=[],
        )["environment"]
        self.assertEqual(
            environment["PIP_INDEX_URL"], "https://primary.example.test/simple"
        )
        self.assertEqual(
            environment["PIP_EXTRA_INDEX_URL"],
            "https://pypi.tuna.tsinghua.edu.cn/simple "
            "https://secondary.example.test/simple",
        )

    def test_setup_never_creates_a_missing_root_parent(self) -> None:
        root = Path(self.temporary.name) / "existing"
        root.mkdir()
        missing_parent = root / "missing-parent"
        target = missing_parent / "managed-root"
        completed = subprocess.run(
            [sys.executable, "-c", remote_gpu.REMOTE_CREATE_MANAGED_ROOTS, str(target), str(root / "durable")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(missing_parent.exists())

    def test_public_ssh_parser_has_no_remote_command_or_interactive_mode(self) -> None:
        parser = ssh_remote.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["echo outside"])
        parsed = parser.parse_args(["--check"])
        self.assertTrue(parsed.check)

    def test_project_conda_environment_is_structured_and_root_bound(self) -> None:
        parser = infra_tools.build_parser()
        parsed = parser.parse_args([
            "create-env", "--prefix",
            "/scratch/remote-gpu-dev/guard/envs/demo", "--python", "3.12",
            "--package", "pytorch=2.9",
        ])
        self.assertEqual(parsed.package, ["pytorch=2.9"])
        command = infra_tools._project_env_command(
            self.profile,
            action="create",
            prefix=parsed.prefix,
            python=parsed.python,
            packages=parsed.package,
            proxy=False,
        )
        self.assertIn("/scratch/remote-gpu-dev/guard/envs/demo", command)
        self.assertIn("CONDA_PKGS_DIRS=/scratch/remote-gpu-dev/guard/runtime/conda-pkgs", command)
        self.assertIn("/bin/rm -rf -- /scratch/remote-gpu-dev/guard/envs/demo", command)
        with self.assertRaises(guard.RemotePathError):
            infra_tools._project_env_command(
                self.profile, action="create", prefix="/tmp/demo",
                python="3.12", packages=[], proxy=False,
            )
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "install-env", "--prefix",
                "/scratch/remote-gpu-dev/guard/envs/demo", "--package",
                "https://example.invalid/pkg.whl",
            ])

    def test_created_environment_is_a_successful_cli_result(self) -> None:
        with (
            mock.patch.object(infra_tools, "load_profile", return_value=self.profile),
            mock.patch.object(
                infra_tools,
                "project_environment",
                return_value={
                    "status": "created",
                    "prefix": "/scratch/remote-gpu-dev/guard/envs/demo",
                },
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    "infra_tools.py",
                    "create-env",
                    "--prefix",
                    "/scratch/remote-gpu-dev/guard/envs/demo",
                    "--python",
                    "3.12",
                ],
            ),
        ):
            self.assertEqual(infra_tools.main(), 0)

    def test_failed_direct_create_is_cleaned_before_proxy_retry(self) -> None:
        failed = subprocess.CompletedProcess(
            ["ssh"], 1, stdout="", stderr="network failed"
        )
        succeeded = subprocess.CompletedProcess(
            ["ssh"], 0, stdout="", stderr=""
        )
        with mock.patch.object(
            infra_tools, "_remote", side_effect=[failed, succeeded]
        ) as remote:
            payload = infra_tools.project_environment(
                self.profile,
                action="create",
                prefix="/scratch/remote-gpu-dev/guard/envs/retry",
                python="3.12",
                packages=["pytorch"],
                use_proxy=False,
            )
        self.assertEqual(payload["route"], "proxy")
        self.assertEqual(remote.call_count, 2)
        for call in remote.call_args_list:
            self.assertIn(
                "/bin/rm -rf -- /scratch/remote-gpu-dev/guard/envs/retry",
                call.args[1],
            )

    def test_unknown_create_outcome_never_attempts_proxy_fallback(self) -> None:
        with mock.patch.object(
            infra_tools,
            "_remote",
            side_effect=infra_tools.InfraError("remote command timed out"),
        ) as remote:
            with self.assertRaises(infra_tools.InfraError):
                infra_tools.project_environment(
                    self.profile,
                    action="create",
                    prefix="/scratch/remote-gpu-dev/guard/envs/unknown",
                    python="3.12",
                    packages=[],
                    use_proxy=False,
                )
        remote.assert_called_once()

    def test_pip_install_is_structured_profile_bound_and_rejects_urls(self) -> None:
        profile = copy.deepcopy(self.profile)
        profile["network"]["pip_index_url"] = "https://primary.example.test/simple"
        command = infra_tools._project_pip_command(
            profile,
            prefix="/scratch/remote-gpu-dev/guard/envs/demo",
            packages=["torch==2.9.0", "tensorboard>=2.20"],
            proxy=False,
        )
        self.assertIn("/envs/demo/bin/python -m pip install", command)
        self.assertIn("PIP_INDEX_URL=https://primary.example.test/simple", command)
        self.assertIn("PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple", command)
        with self.assertRaises(infra_tools.InfraError):
            infra_tools._project_pip_command(
                profile,
                prefix="/scratch/remote-gpu-dev/guard/envs/demo",
                packages=["https://example.invalid/pkg.whl"],
                proxy=False,
            )

    def test_detached_job_has_root_bound_state_and_exact_identity_control(self) -> None:
        source = managed_run.LANDLOCK_RUNNER
        control = managed_run.JOB_CONTROL
        self.assertIn("os.setsid()", source)
        self.assertIn('job_dir / "run.log"', source)
        self.assertIn('job_dir / "identity.json"', source)
        self.assertIn('job_dir / "final.json"', source)
        self.assertIn("process_start_ticks", source)
        self.assertIn("os.killpg(worker_pid, signum)", source)
        self.assertIn('os.kill(identity["pid"], signal.SIGTERM)', control)
        self.assertIn('current_boot != boot_id', control)
        self.assertIn('ticks != identity["process_start_ticks"]', control)
        parsed = managed_run.parse_command_line([
            "GPU-fixture", "--status", "--session", "exact-session"
        ])
        self.assertTrue(parsed.status)
        self.assertEqual(parsed.session, "exact-session")
        launch = managed_run.parse_command_line([
            "GPU-fixture",
            "--env-prefix", "/scratch/remote-gpu-dev/guard/env",
            "--script", "/scratch/remote-gpu-dev/guard/project/train.py",
            "--", "--epochs", "20",
        ])
        self.assertEqual(launch.arguments, ["--epochs", "20"])
        self.assertIsNone(launch.workdir)
        module_launch = managed_run.parse_command_line([
            "GPU-fixture",
            "--env-prefix", "/scratch/remote-gpu-dev/guard/env",
            "--module", "torch.distributed.run",
            "--", "--nproc-per-node", "4", "train.py",
        ])
        self.assertEqual(module_launch.module, "torch.distributed.run")
        self.assertEqual(
            module_launch.arguments, ["--nproc-per-node", "4", "train.py"]
        )
        for invalid in (
            [
                "GPU-fixture", "--env-prefix", "/scratch/x", "--script", "/a.py",
                "--module", "torch.distributed.run",
            ],
            ["GPU-fixture", "--module", "torch-distributed.run"],
            ["GPU-fixture", "--strict-isolation"],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                managed_run.parse_command_line(invalid)
        self.assertEqual(
            managed_run._job_directory(
                self.profile, "GPU-fixture", "exact-session"
            ),
            "/scratch/remote-gpu-dev/guard/runtime/runs/"
            "GPU-fixture/jobs/exact-session",
        )
        ticket = {
            "id": "GPU-fixture",
            "session": "other-session",
            "remote_workdir": "/scratch/remote-gpu-dev/guard/run",
            "assigned_gpus": [0],
        }
        with self.assertRaises(managed_run.ManagedRunError):
            managed_run._control_job(
                self.profile, ticket, session="exact-session", action="stop"
            )


if __name__ == "__main__":
    unittest.main()
