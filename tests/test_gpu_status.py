#!/usr/bin/env python3
"""Offline regressions for fail-closed GPU telemetry."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "skills" / "remote-gpu-dev" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import gpu_status  # noqa: E402
import profile as profiles  # noqa: E402


class GPUStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ssh = mock.patch.object(gpu_status, "ssh_argv", return_value=["ssh"])
        self.ssh.start()
        self.addCleanup(self.ssh.stop)

    @staticmethod
    def profile(mig_policy: str = "disabled") -> dict:
        devices = [
            {
                "index": 0,
                "uuid": "GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "name": "Example GPU",
                "memory_mib": 24576,
            },
            {
                "index": 1,
                "uuid": "GPU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "name": "Example GPU",
                "memory_mib": 24576,
            },
        ]
        profile = profiles.default_profile(
            name="Lab GPU",
            slug="lab-gpu",
            host="gpu.example.test",
            user="researcher",
            port=2222,
            identity_file="/tmp/lab-gpu-key",
            local_projects_root="/tmp/projects/lab-gpu",
            ticket_root="/tmp/tickets/lab-gpu",
            remote_temp_root="/scratch/remote-gpu-dev/lab-gpu",
            remote_durable_root="/data/remote-gpu-dev/lab-gpu",
            gpu_ids=[0, 1],
            conda_executable="/opt/conda/bin/conda",
            monitor_python="/data/remote-gpu-dev/lab-gpu/infra/monitor-env/bin/python",
            host_key_fingerprints=["SHA256:abcdefghijklmnopqrstuvwx"],
            remote_machine_id_sha256="sha256:"
            + hashlib.sha256(b"lab-gpu").hexdigest(),
            gpu_devices=devices,
        )
        profile["gpu"]["mig_mode"] = mig_policy
        return profiles.validate_profile(profile)

    @staticmethod
    def completed(gpu_rows: str, process_rows: str = "") -> subprocess.CompletedProcess[str]:
        output = (
            "__REMOTE_GPU_DEV_GPUS__\n"
            + gpu_rows
            + "__REMOTE_GPU_DEV_PROCESSES__\n"
            + process_rows
        )
        return subprocess.CompletedProcess(["ssh"], 0, stdout=output, stderr="")

    def test_compute_query_failure_is_not_suppressed(self) -> None:
        self.assertNotIn("|| true", gpu_status.REMOTE_COMMAND)
        failure = subprocess.CompletedProcess(
            ["ssh"], 1, stdout="", stderr="NVML compute process query failed"
        )
        with mock.patch.object(gpu_status.subprocess, "run", return_value=failure):
            with self.assertRaises(gpu_status.GPUStatusError):
                gpu_status.snapshot(self.profile())

    def test_exact_index_uuid_mapping_is_required(self) -> None:
        swapped = (
            "0, GPU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, Example GPU, "
            "24576, 0, 0, 0, 30, P8, Disabled\n"
            "1, GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, Example GPU, "
            "24576, 0, 0, 0, 31, P8, Disabled\n"
        )
        with mock.patch.object(
            gpu_status.subprocess, "run", return_value=self.completed(swapped)
        ):
            payload = gpu_status.snapshot(self.profile())
        self.assertFalse(payload["inventory_match"])
        self.assertFalse(payload["safe_to_allocate"])
        self.assertEqual(set(payload["mapping_mismatches"]), {"0", "1"})

    def test_runtime_mig_must_be_disabled(self) -> None:
        rows = (
            "0, GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, Example GPU, "
            "24576, 0, 0, 0, 30, P8, Enabled\n"
            "1, GPU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, Example GPU, "
            "24576, 0, 0, 0, 31, P8, Disabled\n"
        )
        with mock.patch.object(
            gpu_status.subprocess, "run", return_value=self.completed(rows)
        ):
            payload = gpu_status.snapshot(self.profile())
        self.assertTrue(payload["inventory_match"])
        self.assertFalse(payload["mig_policy_match"])
        self.assertFalse(payload["safe_to_allocate"])
        self.assertEqual(payload["unsafe_mig_modes"], {"0": "enabled"})

    def test_explicit_unsupported_mig_policy_is_accepted_and_pinned(self) -> None:
        rows = (
            "0, GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, Example GPU, "
            "24576, 0, 0, 0, 30, P8, N/A\n"
            "1, GPU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, Example GPU, "
            "24576, 0, 0, 0, 31, P8, Not Supported\n"
        )
        with mock.patch.object(
            gpu_status.subprocess, "run", return_value=self.completed(rows)
        ):
            payload = gpu_status.snapshot(self.profile("unsupported"))
        self.assertTrue(payload["mig_policy_match"])
        self.assertTrue(payload["safe_to_allocate"])
        self.assertFalse(payload["mig_disabled"])
        with mock.patch.object(
            gpu_status.subprocess, "run", return_value=self.completed(rows)
        ):
            changed = gpu_status.snapshot(self.profile("disabled"))
        self.assertFalse(changed["mig_policy_match"])
        self.assertFalse(changed["safe_to_allocate"])

    def test_malformed_compute_rows_fail_closed(self) -> None:
        rows = (
            "0, GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, Example GPU, "
            "24576, 0, 0, 0, 30, P8, Disabled\n"
            "1, GPU-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, Example GPU, "
            "24576, 0, 0, 0, 31, P8, Disabled\n"
        )
        malformed = "GPU-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, not-a-pid, python, 100\n"
        with mock.patch.object(
            gpu_status.subprocess,
            "run",
            return_value=self.completed(rows, malformed),
        ):
            with self.assertRaises(gpu_status.GPUStatusError):
                gpu_status.snapshot(self.profile())


if __name__ == "__main__":
    unittest.main()
