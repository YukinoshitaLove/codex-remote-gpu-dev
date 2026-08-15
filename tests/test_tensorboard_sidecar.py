#!/usr/bin/env python3
"""Failure-window tests for the ticket-bound TensorBoard sidecar."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "remote-gpu-dev"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "pytorch_dev_tensorboard_sidecar",
    SKILL_ROOT / "scripts" / "tensorboard_sidecar.py",
)
assert SPEC and SPEC.loader
sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sidecar)


TICKET_ID = "GPU-20260811-120000-abcd-sidecar-test"


def runtime_object(*pairs: tuple[str, object]) -> dict[str, object]:
    return dict(pairs)


class SidecarFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_profile = sidecar.PROFILE
        sidecar.PROFILE = {
            "ssh": {"user": "fixture", "host": "gpu.invalid"},
            "remote": {
                "temp_root": "/srv/remote-gpu-dev-fixture",
                "durable_root": "/archive/remote-gpu-dev-fixture",
            }
        }
        self.addCleanup(setattr, sidecar, "PROFILE", self.original_profile)

    def test_only_ssh_255_and_ssh_timeout_are_transient(self) -> None:
        cases = (
            ("ssh", 255, sidecar.TransientSSHError),
            ("ssh", 1, sidecar.SidecarError),
            ("helper", 255, sidecar.SidecarError),
        )
        for executable, returncode, error_type in cases:
            completed = subprocess.CompletedProcess(
                [executable], returncode, stdout="", stderr="fixture failure"
            )
            with (
                self.subTest(executable=executable, returncode=returncode),
                mock.patch.object(sidecar.subprocess, "run", return_value=completed),
                self.assertRaises(sidecar.SidecarError) as raised,
            ):
                sidecar._run([executable])
            self.assertIs(type(raised.exception), error_type)

        for executable, error_type in (
            ("ssh", sidecar.TransientSSHError),
            ("helper", sidecar.SidecarError),
        ):
            with (
                self.subTest(executable=executable, failure="timeout"),
                mock.patch.object(
                    sidecar.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired([executable], 6),
                ),
                self.assertRaises(sidecar.SidecarError) as raised,
            ):
                sidecar._run([executable], timeout=6)
            self.assertIs(type(raised.exception), error_type)

    def test_remote_readonly_control_retries_five_transient_ssh_failures(self) -> None:
        failures = [
            sidecar.TransientSSHError("ssh exited 255: Connection closed")
            for _ in range(4)
        ]
        completed = mock.Mock(returncode=0, stdout='{"ok": true}', stderr="")
        with (
            mock.patch.object(sidecar, "ssh_argv", return_value=["ssh"]),
            mock.patch.object(
                sidecar, "build_landlock_command", return_value="remote-command"
            ),
            mock.patch.object(sidecar, "_run", side_effect=[*failures, completed]) as run,
            mock.patch.object(sidecar.time, "sleep") as sleep,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            result = sidecar._remote_json(
                "readonly", timeout=35, retry_transient=True
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(run.call_count, 5)
        self.assertEqual(sleep.call_args_list, [mock.call(0.5)] * 4)
        self.assertEqual(stderr.getvalue().count("warning:"), 4)
        self.assertTrue(all(call.kwargs["timeout"] <= 6 for call in run.call_args_list))

    def test_remote_readonly_control_stops_after_five_transient_ssh_failures(self) -> None:
        with (
            mock.patch.object(sidecar, "ssh_argv", return_value=["ssh"]),
            mock.patch.object(
                sidecar, "build_landlock_command", return_value="remote-command"
            ),
            mock.patch.object(
                sidecar,
                "_run",
                side_effect=sidecar.TransientSSHError(
                    "ssh exited 255: Connection closed"
                ),
            ) as run,
            mock.patch.object(sidecar.time, "sleep"),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaisesRegex(
                sidecar.TransientSSHError, "5 consecutive SSH attempts failed"
            ),
        ):
            sidecar._remote_json("readonly", timeout=35, retry_transient=True)
        self.assertEqual(run.call_count, 5)
        self.assertEqual(stderr.getvalue().count("warning:"), 4)

    def test_remote_retry_window_is_one_shared_deadline(self) -> None:
        with (
            mock.patch.object(sidecar, "ssh_argv", return_value=["ssh"]),
            mock.patch.object(
                sidecar, "build_landlock_command", return_value="remote-command"
            ),
            mock.patch.object(
                sidecar,
                "_run",
                side_effect=sidecar.TransientSSHError(
                    "ssh exited 255: Connection closed"
                ),
            ) as run,
            mock.patch.object(sidecar.time, "monotonic", side_effect=[0.0, 0.0, 34.75, 35.1]),
            mock.patch.object(sidecar.time, "sleep") as sleep,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(
                sidecar.TransientSSHError, "expired after 1 of 5 attempts"
            ),
        ):
            sidecar._remote_json("readonly", timeout=35, retry_transient=True)
        self.assertEqual(run.call_count, 1)
        sleep.assert_called_once_with(0.25)

    def test_remote_mutating_control_is_not_blindly_retried(self) -> None:
        with (
            mock.patch.object(sidecar, "ssh_argv", return_value=["ssh"]),
            mock.patch.object(
                sidecar, "build_landlock_command", return_value="remote-command"
            ),
            mock.patch.object(
                sidecar,
                "_run",
                side_effect=sidecar.TransientSSHError(
                    "ssh exited 255: Connection closed"
                ),
            ) as run,
            self.assertRaises(sidecar.TransientSSHError),
        ):
            sidecar._remote_json("mutating", timeout=35)
        self.assertEqual(run.call_count, 1)
        self.assertGreater(run.call_args.kwargs["timeout"], 30)

    def test_only_readonly_remote_call_sites_enable_bounded_retry(self) -> None:
        self.assertIn(
            "retry_transient=True", inspect.getsource(sidecar._inspect_metadata)
        )
        self.assertIn(
            "retry_transient=True",
            inspect.getsource(sidecar._verify_untracked_absent),
        )
        self.assertIn(
            "retry_transient=True", inspect.getsource(sidecar.command_configure)
        )
        self.assertEqual(
            inspect.getsource(sidecar.command_start).count("retry_transient=True"),
            1,
        )
        self.assertNotIn(
            "retry_transient=True", inspect.getsource(sidecar.command_stop)
        )

    def test_remote_launch_port_probe_allows_time_wait_reuse(self) -> None:
        self.assertIn(
            "probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)",
            sidecar.REMOTE_LAUNCH,
        )

    def test_remote_launch_surfaces_tmux_failure_detail(self) -> None:
        self.assertIn(
            'detail = (launched.stderr or launched.stdout).strip()',
            sidecar.REMOTE_LAUNCH,
        )
        self.assertIn(
            'tmux could not create the exact sidecar session (exit %d)',
            sidecar.REMOTE_LAUNCH,
        )

    def test_remote_stop_accepts_pid_reuse_only_after_session_and_port_are_absent(self) -> None:
        safe = (
            'if not pane_pids(session) and not port_listening(port):\n'
            '            emit("stopped", "recorded PID was reused after stop and no session or port residue remains")'
        )
        self.assertIn(safe, sidecar.REMOTE_STOP)

    def test_sensitive_error_matrix_is_suppressed_before_ledger_updates(self) -> None:
        github_classic = "ghp" + "_" + "A" * 32
        private_jwk_strings = tuple(
            json.dumps(jwk)
            for jwk in (
                runtime_object(("kty", "EC"), ("d", "private-fixture")),
                runtime_object(("kty", "OKP"), ("d", "private-fixture")),
                runtime_object(("kty", "RSA"), ("d", "private-fixture")),
                runtime_object(
                    ("kty", "RSA"),
                    ("oth", [runtime_object(("r", "r"), ("d", "d"), ("t", "t"))]),
                ),
                runtime_object(("kty", "oct"), ("k", "symmetric-fixture")),
            )
        ) + (
            "\ufeff" + json.dumps(runtime_object(("kty", "oct"), ("k", "symmetric-fixture"))),
            "\u200b" + json.dumps(runtime_object(("kty", "OKP"), ("d", "private-fixture"))),
        )
        secret_values = (
            "".join(("pass", "phrase=fixture-correct-horse")),
            "".join(("-----BEGIN ", "PRIVATE KEY----- ", "B" * 64)),
            "gho" + "_" + "C" * 24,
            "xox" + "b-" + "D" * 24,
            "AK" + "IA" + "E" * 16,
            "说明" + github_classic + "结束",
            "sk" + "-ant-api03-" + "F" * 48,
            "gl" + "pat-" + "G" * 20,
            "pypi-" + "AgEIcHlwaS5vcmc" + "H" * 32,
            "npm" + "_" + "I" * 36,
            "sk" + "_live_" + "J" * 24,
            "AI" + "za" + "K" * 35,
            "ya" + "29." + "L" * 40,
            "".join(("OPENAI", "_API_KEY=fixture-value")),
            "".join(("MLFLOW_TRACKING_PASS", "WORD=fixture-value")),
            "".join(("CUSTOM_ACCESS_TO", "KEN=fixture-value")),
            "".join(("clientSec", "ret=fixture-value")),
            "".join(("ACCESS_TO", "KENS=fixture-value")),
            "".join(("authTo", "kens=fixture-value")),
            "".join(("clientCred", "entials=fixture-value")),
            "".join(("CREDEN", "TIALS=fixture-value")),
            "".join(("SERVICE_ACCOUNT_", "KEY=fixture-value")),
            "".join(("pass", "word\n=fixture-value")),
            "".join(("Bear", "er\nabc123fixture")),
            "".join(("pass", "word\N{NO-BREAK SPACE}=fixture-value")),
            "".join(("Bear", "er\N{NO-BREAK SPACE}abc123fixture")),
            "".join(("pass", "word\u200b=fixture-value")),
            "".join(("postgresql://fixture-user", ":fixture-password@db.invalid/run")),
            "".join(("redis://fixture-user", ":fixture-password@cache.invalid/0")),
            "".join(("ssh://fixture-user", ":fixture-password@gpu.invalid/home")),
            "".join(("https://", ":fixture-password@example.invalid/simple")),
            "".join(("x://fixture-user", ":fixture-password@host")),
            "".join(("//fixture-user", ":fixture-password@host/path")),
            "".join(("//", ":fixture-password@host/path")),
            *private_jwk_strings,
        )
        suppressed = "sidecar operation failed; sensitive-looking detail was suppressed"
        for secret_value in secret_values:
            with self.subTest(value=secret_value[:12]):
                self.assertEqual(sidecar._safe_last_error(secret_value), suppressed)
                calls: list[dict] = []

                def fake_ledger(_ticket_id: str, _status: str, **fields):
                    calls.append(fields)
                    return {}

                with mock.patch.object(
                    sidecar, "_ticket_tensorboard", side_effect=fake_ledger
                ):
                    self.assertTrue(
                        sidecar._best_effort_ticket_update(
                            TICKET_ID, "failed", secret_value, 1
                        )
                    )
                self.assertEqual(calls[0]["last_error"], suppressed)
                self.assertNotIn(secret_value, json.dumps(calls))

                failed = subprocess.CompletedProcess(
                    ["fixture"], 1, stdout="", stderr=secret_value
                )
                with (
                    mock.patch.object(sidecar.subprocess, "run", return_value=failed),
                    self.assertRaisesRegex(sidecar.SidecarError, "suppressed") as raised,
                ):
                    sidecar._run(["fixture"])
                self.assertNotIn(secret_value, str(raised.exception))

        self.assertEqual(
            sidecar._safe_last_error("training\N{NO-BREAK SPACE}failed"),
            "training failed",
        )
        for safe_url in (
            "x://fixture-user@host",
            "x://fixture-user:@host",
            "x://:@host",
            "x://host:22/path",
            "//fixture-user@host/path",
            "//fixture-user:@host/path",
            "//:@host/path",
            "//host:443/path",
        ):
            with self.subTest(safe_url=safe_url):
                self.assertEqual(sidecar._safe_last_error(safe_url), safe_url)
        for jwk in (
            {"kty": "EC", "crv": "P-256", "x": "x", "y": "y"},
            {"kty": "OKP", "crv": "Ed25519", "x": "x"},
            {"kty": "RSA", "n": "modulus", "e": "AQAB"},
            {"kty": "ec", "d": "descriptive-dimension"},
            {"kty": "RSA", "p": None},
            {"kty": "oct", "k": ""},
        ):
            rendered = json.dumps(jwk)
            with self.subTest(public_jwk=jwk["kty"]):
                self.assertEqual(sidecar._safe_last_error(rendered), rendered)

    @staticmethod
    def _metadata(status: str = "stopped", generation: int = 1) -> dict:
        return {
            "status": status,
            "generation": generation,
            "logdir": "/srv/remote-gpu-dev-fixture/records/test-run/events",
            "env_prefix": "/srv/remote-gpu-dev-fixture/projects/test/.conda/env",
            "remote_port": None if status == "stopped" else 16006,
            "path_prefix": sidecar._tensorboard_path_prefix(TICKET_ID),
            "session": sidecar._session_name(TICKET_ID),
            "pid": None if status == "stopped" else 4321,
            "process_start_ticks": None if status == "stopped" else 9876,
            "boot_id": (
                None
                if status == "stopped"
                else "11111111-2222-3333-4444-555555555555"
            ),
            "version": None if status == "stopped" else "2.20.0",
            "command_sha256": None if status == "stopped" else "a" * 64,
        }

    def test_legacy_unicode_ticket_uses_encoded_path_and_hashed_ascii_session(self) -> None:
        ticket_id = "GPU-20260811-120000-abcd-视觉_实验"
        self.assertEqual(
            sidecar._tensorboard_path_prefix(ticket_id),
            "/tb/GPU-20260811-120000-abcd-%E8%A7%86%E8%A7%89_%E5%AE%9E%E9%AA%8C",
        )
        session = sidecar._session_name(ticket_id)
        self.assertRegex(session, r"^remote-gpu-tb-[0-9a-f]{32}$")
        self.assertNotIn("视觉", session)

    def test_configure_registers_stopped_source_without_launching_frontend(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "running",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": None,
        }
        preflight = {
            "ok": True,
            "workdir": ticket["remote_workdir"],
            "logdir": "/srv/remote-gpu-dev-fixture/records/test-run/events",
            "env_prefix": "/srv/remote-gpu-dev-fixture/projects/test/.conda/env",
            "version": "2.20.0",
        }
        ledger_calls: list[tuple[str, dict]] = []

        def fake_remote(code: str, *_arguments: str, **_kwargs):
            self.assertEqual(code, sidecar.REMOTE_PREFLIGHT)
            return preflight

        def fake_ledger(_ticket_id: str, status: str, **fields):
            ledger_calls.append((status, fields))
            metadata = {
                **self._metadata("stopped", 1),
                "logdir": fields["logdir"],
                "env_prefix": fields["env_prefix"],
            }
            return {**ticket, "tensorboard": metadata}

        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            env_prefix=preflight["env_prefix"],
            logdir=preflight["logdir"],
        )
        output = io.StringIO()
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json", side_effect=fake_remote),
            mock.patch.object(sidecar, "_ticket_tensorboard", side_effect=fake_ledger),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(sidecar.command_configure(args), 0)

        self.assertEqual(len(ledger_calls), 1)
        status, fields = ledger_calls[0]
        self.assertEqual(status, "stopped")
        self.assertEqual(fields["expected_generation"], 0)
        self.assertNotIn("remote_port", fields)
        self.assertNotIn("pid", fields)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["configured"])
        self.assertFalse(payload["frontend_started"])

    def test_configure_refuses_active_generation_before_remote_preflight(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "running",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": self._metadata("live", 1),
        }
        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            env_prefix="/srv/remote-gpu-dev-fixture/projects/test/.conda/env",
            logdir="/srv/remote-gpu-dev-fixture/records/test-run/events",
        )
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json") as remote,
            self.assertRaisesRegex(sidecar.SidecarError, "active or unresolved"),
        ):
            sidecar.command_configure(args)
        remote.assert_not_called()

    def test_running_ticket_can_start_from_retained_stopped_source_without_paths(self) -> None:
        stopped_metadata = self._metadata("stopped", 1)
        ticket = {
            "id": TICKET_ID,
            "status": "running",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": stopped_metadata,
        }
        preflight = {
            "ok": True,
            "workdir": ticket["remote_workdir"],
            "logdir": stopped_metadata["logdir"],
            "env_prefix": stopped_metadata["env_prefix"],
            "version": "2.20.0",
        }
        launched = {
            "ok": True,
            "pid": 4321,
            "process_start_ticks": 9876,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "command_sha256": "a" * 64,
            "log_file": "/srv/remote-gpu-dev-fixture/records/test-run/tensorboard.log",
        }
        ledger_calls: list[tuple[str, dict]] = []

        launch_options: dict[str, object] = {}

        def fake_remote(code: str, *_arguments: str, **kwargs):
            if code == sidecar.REMOTE_PREFLIGHT:
                return preflight
            if code == sidecar.REMOTE_LAUNCH:
                launch_options.update(kwargs)
                return launched
            raise AssertionError("unexpected remote helper")

        def fake_ledger(_ticket_id: str, status: str, **fields):
            ledger_calls.append((status, fields))
            if status == "starting":
                self.assertEqual(fields["logdir"], stopped_metadata["logdir"])
                self.assertEqual(fields["env_prefix"], stopped_metadata["env_prefix"])
                return {
                    **ticket,
                    "tensorboard": {
                        **stopped_metadata,
                        "status": "starting",
                        "generation": 2,
                        "remote_port": 16006,
                    },
                }
            if status == "live":
                return {
                    **ticket,
                    "tensorboard": {
                        **stopped_metadata,
                        "status": "live",
                        "generation": 2,
                        "remote_port": 16006,
                        **fields,
                    },
                }
            raise AssertionError(f"unexpected ledger state {status}")

        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            remote_port=None,
            env_prefix=None,
            logdir=None,
            startup_timeout=2.0,
        )
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json", side_effect=fake_remote),
            mock.patch.object(sidecar, "_ticket_tensorboard", side_effect=fake_ledger),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(sidecar.command_start(args), 0)
        self.assertEqual([status for status, _fields in ledger_calls], ["starting", "live"])
        self.assertIs(launch_options.get("allow_pty"), True)

    def test_start_without_paths_requires_stopped_configuration(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "running",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": None,
        }
        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            remote_port=None,
            env_prefix=None,
            logdir=None,
            startup_timeout=2.0,
        )
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json") as remote,
            self.assertRaisesRegex(sidecar.SidecarError, "stopped TensorBoard"),
        ):
            sidecar.command_start(args)
        remote.assert_not_called()

    def test_expected_generation_mismatch_never_inspects_or_stops_remote(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "completed",
            "tensorboard": self._metadata("live", 2),
        }
        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            stop_timeout=1.0,
            expected_generation=1,
        )
        output = io.StringIO()
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json") as remote,
            mock.patch.object(sidecar, "_ticket_tensorboard") as ledger,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(sidecar.command_stop(args), 0)
        remote.assert_not_called()
        ledger.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "superseded")
        self.assertEqual(payload["observed_generation"], 2)

    def test_generation_advance_after_remote_stop_is_superseded(self) -> None:
        original = {
            "id": TICKET_ID,
            "status": "completed",
            "tensorboard": self._metadata("live", 1),
        }
        newer = {
            **original,
            "tensorboard": self._metadata("live", 2),
        }
        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            stop_timeout=1.0,
            expected_generation=1,
        )
        output = io.StringIO()
        with (
            mock.patch.object(sidecar, "_ticket_status", side_effect=[original, newer]),
            mock.patch.object(
                sidecar,
                "_remote_json",
                return_value={"status": "stopped", "message": "old generation stopped"},
            ),
            mock.patch.object(
                sidecar,
                "_ticket_tensorboard",
                side_effect=sidecar.SidecarError("generation changed"),
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(sidecar.command_stop(args), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "superseded")
        self.assertEqual(payload["observed_generation"], 2)

    def test_live_commit_and_remote_rollback_failure_records_exact_cleanup_identity(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "completed",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": None,
        }
        preflight = {
            "ok": True,
            "workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "logdir": "/srv/remote-gpu-dev-fixture/records/test-run/events",
            "env_prefix": "/srv/remote-gpu-dev-fixture/projects/test/.conda/env",
            "version": "2.20.0",
        }
        launched = {
            "ok": True,
            "pid": 4321,
            "process_start_ticks": 9876,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "command_sha256": "a" * 64,
            "log_file": "/srv/remote-gpu-dev-fixture/records/test-run/tensorboard.log",
        }
        calls: list[tuple[str, dict]] = []

        def fake_remote(code: str, *_arguments: str, **_kwargs):
            if code == sidecar.REMOTE_PREFLIGHT:
                return preflight
            if code == sidecar.REMOTE_LAUNCH:
                return launched
            if code == sidecar.REMOTE_STOP:
                raise sidecar.SidecarError("simulated SSH rollback failure")
            raise AssertionError("unexpected remote helper")

        def fake_ledger(_ticket_id: str, status: str, **fields):
            calls.append((status, fields))
            if status == "starting":
                return {
                    **ticket,
                    "tensorboard": {
                        "status": "starting",
                        "generation": 1,
                        "remote_port": 16006,
                    },
                }
            if status == "live":
                raise sidecar.SidecarError("simulated live ledger failure")
            if status == "cleanup_pending":
                return {**ticket, "tensorboard": {"status": status, **fields}}
            raise AssertionError(f"unexpected ledger state {status}")

        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            remote_port=None,
            env_prefix=preflight["env_prefix"],
            logdir=preflight["logdir"],
            startup_timeout=2.0,
        )
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json", side_effect=fake_remote),
            mock.patch.object(sidecar, "_ticket_tensorboard", side_effect=fake_ledger),
            self.assertRaisesRegex(sidecar.SidecarError, "cleanup is pending"),
        ):
            sidecar.command_start(args)

        cleanup = [fields for status, fields in calls if status == "cleanup_pending"]
        self.assertEqual(len(cleanup), 1)
        self.assertEqual(cleanup[0]["expected_generation"], 1)
        self.assertEqual(cleanup[0]["pid"], launched["pid"])
        self.assertEqual(
            cleanup[0]["process_start_ticks"], launched["process_start_ticks"]
        )
        self.assertEqual(cleanup[0]["boot_id"], launched["boot_id"])
        self.assertEqual(cleanup[0]["command_sha256"], launched["command_sha256"])

    def test_keyboard_interrupt_windows_retry_cleanup_record(self) -> None:
        ticket = {
            "id": TICKET_ID,
            "status": "completed",
            "remote_workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "tensorboard": None,
        }
        preflight = {
            "ok": True,
            "workdir": "/srv/remote-gpu-dev-fixture/records/test-run",
            "logdir": "/srv/remote-gpu-dev-fixture/records/test-run/events",
            "env_prefix": "/srv/remote-gpu-dev-fixture/projects/test/.conda/env",
            "version": "2.20.0",
        }
        launched = {
            "ok": True,
            "pid": 4321,
            "process_start_ticks": 9876,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "command_sha256": "a" * 64,
            "log_file": "/srv/remote-gpu-dev-fixture/records/test-run/tensorboard.log",
        }
        calls: list[tuple[str, dict]] = []
        cleanup_attempts = 0

        def fake_remote(code: str, *_arguments: str, **_kwargs):
            if code == sidecar.REMOTE_PREFLIGHT:
                return preflight
            if code == sidecar.REMOTE_LAUNCH:
                return launched
            if code == sidecar.REMOTE_STOP:
                raise KeyboardInterrupt()
            raise AssertionError("unexpected remote helper")

        def fake_ledger(_ticket_id: str, status: str, **fields):
            nonlocal cleanup_attempts
            calls.append((status, fields))
            if status == "starting":
                return {
                    **ticket,
                    "tensorboard": {
                        "status": "starting",
                        "generation": 1,
                        "remote_port": 16006,
                    },
                }
            if status == "live":
                raise KeyboardInterrupt()
            if status == "cleanup_pending":
                cleanup_attempts += 1
                if cleanup_attempts == 1:
                    raise KeyboardInterrupt()
                return {**ticket, "tensorboard": {"status": status, **fields}}
            raise AssertionError(f"unexpected ledger state {status}")

        args = argparse.Namespace(
            ticket_id=TICKET_ID,
            remote_port=None,
            env_prefix=preflight["env_prefix"],
            logdir=preflight["logdir"],
            startup_timeout=2.0,
        )
        with (
            mock.patch.object(sidecar, "_ticket_status", return_value=ticket),
            mock.patch.object(sidecar, "_remote_json", side_effect=fake_remote),
            mock.patch.object(sidecar, "_ticket_tensorboard", side_effect=fake_ledger),
            self.assertRaisesRegex(sidecar.SidecarError, "cleanup is pending"),
        ):
            sidecar.command_start(args)

        cleanup = [fields for status, fields in calls if status == "cleanup_pending"]
        self.assertEqual(len(cleanup), 2)
        self.assertEqual(cleanup[-1]["expected_generation"], 1)
        self.assertEqual(cleanup[-1]["pid"], launched["pid"])
        self.assertEqual(
            cleanup[-1]["last_error"],
            "ledger live update failed and remote rollback outcome is unknown",
        )


if __name__ == "__main__":
    unittest.main()
