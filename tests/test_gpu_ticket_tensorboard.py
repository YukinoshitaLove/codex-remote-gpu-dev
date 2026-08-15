#!/usr/bin/env python3
"""Isolated state-machine tests for ticket-bound TensorBoard metadata."""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "remote-gpu-dev"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "pytorch_dev_gpu_ticket", SKILL_ROOT / "scripts" / "gpu_ticket.py"
)
assert SPEC and SPEC.loader
ticket_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ticket_tool)


def runtime_object(*pairs: tuple[str, object]) -> dict[str, object]:
    return dict(pairs)


class TensorBoardTicketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name) / "tickets"
        root.mkdir(parents=True)
        self.original_paths = {
            name: getattr(ticket_tool, name)
            for name in (
                "TICKET_ROOT",
                "CONFIG_PATH",
                "STATE_PATH",
                "BOARD_PATH",
                "EVENTS_PATH",
                "LOCK_PATH",
                "TICKET_DIR",
            )
        }
        self.original_profile = ticket_tool.PROFILE
        ticket_tool.PROFILE = {
            "slug": "isolated-test",
            "ssh": {"user": "tester", "host": "gpu.example.test", "port": 22},
            "remote": {
                "temp_root": "/root",
                "durable_root": "/archive/remote-gpu-dev-fixture",
            }
        }
        replacements = {
            "TICKET_ROOT": root,
            "CONFIG_PATH": root / "config.json",
            "STATE_PATH": root / "state.json",
            "BOARD_PATH": root / "BOARD.md",
            "EVENTS_PATH": root / "events.jsonl",
            "LOCK_PATH": root / ".lock",
            "TICKET_DIR": root / "tickets",
        }
        for name, value in replacements.items():
            setattr(ticket_tool, name, value)
        self.addCleanup(self._restore_paths)
        ticket_tool.TICKET_DIR.mkdir()
        ticket_tool.EVENTS_PATH.touch()
        ticket_tool.LOCK_PATH.touch()
        self.config = {
            "schema_version": 1,
            "server": "isolated-test",
            "gpu_ids": [0, 1],
            "reservation_ttl_minutes": 15,
            "heartbeat_grace_minutes": 20,
            "tensorboard_port_start": 16006,
            "tensorboard_port_end": 16007,
        }
        ticket_tool.CONFIG_PATH.write_text(json.dumps(self.config), encoding="utf-8")
        self.state = ticket_tool.initial_state()

    def _restore_paths(self) -> None:
        for name, value in self.original_paths.items():
            setattr(ticket_tool, name, value)
        ticket_tool.PROFILE = self.original_profile

    @staticmethod
    def _silent(function, *args) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            function(*args)

    def _reserve(self, project: str) -> dict:
        before = set(self.state["tickets"])
        args = argparse.Namespace(
            project=project,
            owner="test-owner",
            purpose="TensorBoard state-machine test",
            gpu_ids=None,
            gpus=1,
            expected="1h",
            json=True,
        )
        self._silent(ticket_tool.command_reserve, self.state, self.config, args)
        created = set(self.state["tickets"]) - before
        self.assertEqual(len(created), 1)
        return self.state["tickets"][created.pop()]

    @staticmethod
    def _tb_args(ticket_id: str, status: str, **updates) -> argparse.Namespace:
        values = {
            "ticket_id": ticket_id,
            "status": status,
            "logdir": None,
            "env_prefix": None,
            "remote_port": None,
            "path_prefix": None,
            "session": None,
            "pid": None,
            "process_start_ticks": None,
            "boot_id": None,
            "version": None,
            "command_sha256": None,
            "last_error": None,
            "expected_generation": None,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def _start_metadata(self, ticket_id: str, **updates) -> None:
        fields = {
            "logdir": "/root/tb-test/logs",
            "env_prefix": "/root/tb-test/.conda/env",
            "path_prefix": ticket_tool.tensorboard_path_prefix(ticket_id),
            "session": f"pytorch-tb-{ticket_id}",
        }
        fields.update(updates)
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(ticket_id, "starting", **fields),
        )

    def _mark_live(self, ticket: dict) -> None:
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket["id"],
                "live",
                pid=1234,
                process_start_ticks=5678,
                boot_id="11111111-2222-3333-4444-555555555555",
                version="2.20.0",
                command_sha256="a" * 64,
            ),
        )

    def test_release_preserves_source_and_manual_stop_restart_increments(self) -> None:
        ticket = self._reserve("lifecycle")
        ticket_id = ticket["id"]
        self._start_metadata(ticket_id)
        self.assertEqual(ticket["tensorboard"]["remote_port"], 16006)
        self._mark_live(ticket)
        live = dict(ticket["tensorboard"])
        self._silent(
            ticket_tool.command_release,
            self.state,
            self.config,
            argparse.Namespace(
                ticket_id=ticket_id,
                outcome="cancelled",
                confirmed_stopped=None,
                result="no CUDA process was launched",
            ),
        )
        self.assertEqual(ticket["status"], "cancelled")
        self.assertEqual(ticket["assigned_gpus"], [])
        self.assertEqual(ticket["tensorboard"], live)
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(ticket_id, "stopped"),
        )
        self._start_metadata(ticket_id, logdir="/root/tb-test/logs-reopened")
        self.assertEqual(ticket["tensorboard"]["generation"], 2)
        self.assertEqual(ticket["tensorboard"]["status"], "starting")

    def test_configure_stopped_source_holds_no_port_and_start_reallocates(self) -> None:
        ticket = self._reserve("configured-source")
        ticket_id = ticket["id"]
        configured = self._tb_args(
            ticket_id,
            "stopped",
            logdir="/root/tb-test/configured-events",
            env_prefix="/root/tb-test/.conda/env",
            path_prefix=ticket_tool.tensorboard_path_prefix(ticket_id),
            session="pytorch-tb-configured-source",
        )
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            configured,
        )
        self.assertEqual(ticket["tensorboard"]["status"], "stopped")
        self.assertEqual(ticket["tensorboard"]["generation"], 1)
        self.assertIsNone(ticket["tensorboard"]["remote_port"])
        self.assertEqual(ticket_tool.occupied_tensorboard_ports(self.state), set())

        self._start_metadata(
            ticket_id,
            logdir="/root/tb-test/configured-events",
            session="pytorch-tb-configured-source",
        )
        self.assertEqual(ticket["tensorboard"]["status"], "starting")
        self.assertEqual(ticket["tensorboard"]["generation"], 2)
        self.assertEqual(ticket["tensorboard"]["remote_port"], 16006)

    def test_tensorboard_metadata_never_reconciles_gpu_ticket_lifecycle(self) -> None:
        expiring = self._reserve("overdue-reservation")
        stale_candidate = self._reserve("overdue-heartbeat")
        queued = self._reserve("queued-behind-overdue")
        self.assertEqual(queued["status"], "queued")

        expiring["reservation_expires_at"] = "2000-01-01T00:00:00Z"
        stale_candidate.update(
            {
                "status": "running",
                "reservation_expires_at": None,
                "heartbeat_due_at": "2000-01-01T00:00:00Z",
            }
        )
        before = {
            ticket["id"]: (ticket["status"], list(ticket["assigned_gpus"]))
            for ticket in (expiring, stale_candidate, queued)
        }

        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                expiring["id"],
                "stopped",
                logdir="/root/tb-test/overdue-events",
                env_prefix="/root/tb-test/.conda/env",
                path_prefix=ticket_tool.tensorboard_path_prefix(expiring["id"]),
                session="pytorch-tb-overdue-source",
            ),
        )

        after = {
            ticket["id"]: (ticket["status"], list(ticket["assigned_gpus"]))
            for ticket in (expiring, stale_candidate, queued)
        }
        self.assertEqual(after, before)
        self.assertEqual(queued["status"], "queued")
        pending = ticket_tool.pending_time_transitions(self.state)
        self.assertIn(f"would expire {expiring['id']}", pending)
        self.assertIn(f"would mark stale {stale_candidate['id']}", pending)
        events = ticket_tool.EVENTS_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"event":"auto-expire"', events)
        self.assertNotIn('"event":"auto-stale"', events)
        self.assertNotIn('"event":"auto-reserve"', events)

    def test_reconfigure_stopped_source_creates_clean_generation(self) -> None:
        ticket = self._reserve("reconfigure-source")
        ticket_id = ticket["id"]
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket_id,
                "stopped",
                logdir="/root/tb-test/events-a",
                env_prefix="/root/tb-test/.conda/env",
                path_prefix=ticket_tool.tensorboard_path_prefix(ticket_id),
                session="pytorch-tb-reconfigure-source",
            ),
        )
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket_id,
                "stopped",
                expected_generation=1,
                logdir="/root/tb-test/events-b",
                env_prefix="/root/tb-test/.conda/env",
                path_prefix=ticket_tool.tensorboard_path_prefix(ticket_id),
                session="pytorch-tb-reconfigure-source",
            ),
        )
        metadata = ticket["tensorboard"]
        self.assertEqual(metadata["generation"], 2)
        self.assertEqual(metadata["logdir"], "/root/tb-test/events-b")
        self.assertIsNone(metadata["remote_port"])
        self.assertIsNone(metadata["pid"])
        self.assertIsNone(metadata["command_sha256"])

    def test_initial_source_configuration_has_an_unconfigured_cas_guard(self) -> None:
        ticket = self._reserve("initial-config-cas")
        fields = {
            "logdir": "/root/tb-test/initial-events",
            "env_prefix": "/root/tb-test/.conda/env",
            "path_prefix": ticket_tool.tensorboard_path_prefix(ticket["id"]),
            "session": "pytorch-tb-initial-config-cas",
        }
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket["id"], "stopped", expected_generation=0, **fields
            ),
        )
        self.assertEqual(ticket["tensorboard"]["generation"], 1)
        with self.assertRaisesRegex(ticket_tool.TicketError, "expected unconfigured"):
            self._silent(
                ticket_tool.command_tensorboard,
                self.state,
                self.config,
                self._tb_args(
                    ticket["id"], "stopped", expected_generation=0, **fields
                ),
            )

    def test_port_collision_and_identity_or_path_injection_fail_closed(self) -> None:
        first = self._reserve("first")
        second = self._reserve("second")
        self._start_metadata(first["id"], remote_port=16007)
        with self.assertRaisesRegex(ticket_tool.TicketError, "already registered"):
            self._start_metadata(second["id"], remote_port=16007)
        with self.assertRaises(ticket_tool.TicketError):
            self._start_metadata(second["id"], logdir="/root/../etc")
        self._start_metadata(second["id"], remote_port=16006)
        with self.assertRaisesRegex(ticket_tool.TicketError, "process identity"):
            self._silent(
                ticket_tool.command_tensorboard,
                self.state,
                self.config,
                self._tb_args(second["id"], "live", pid=1234),
            )

    def test_generated_slug_uses_the_shared_url_safe_alphabet(self) -> None:
        self.assertEqual(ticket_tool.slugify("视觉实验 foo_bar"), "foo-bar")
        self.assertRegex(ticket_tool.slugify("ASCII Model 14"), r"^[a-z0-9-]+$")

    def test_ticket_free_text_rejects_secret_values_without_persisting_them(self) -> None:
        github_classic = "ghp" + "_" + "A" * 32
        github_fine_grained = "github" + "_pat_" + "B" * 32
        github_oauth = "gho" + "_" + "C" * 32
        bearer = "Bear" + "er " + "c" * 24
        unicode_adjacent = "说明" + github_classic + "结束"
        sensitive_phrase = "".join(("pass", "phrase=fixture-correct-horse"))
        key_container = "".join(
            ("-----BEGIN ", "PRIVATE KEY----- ", "M" * 64, " -----END ", "PRIVATE KEY-----")
        )
        service_tokens = (
            "sk" + "-ant-api03-" + "N" * 48,
            "gl" + "pat-" + "O" * 20,
            "pypi-" + "AgEIcHlwaS5vcmc" + "P" * 32,
            "npm" + "_" + "Q" * 36,
            "sk" + "_live_" + "R" * 24,
            "AI" + "za" + "S" * 35,
            "ya" + "29." + "T" * 40,
        )
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
                runtime_object(
                    ("keys", [runtime_object(("kty", "EC"), ("d", "nested-private-fixture"))])
                ),
            )
        ) + (
            "\ufeff" + json.dumps(runtime_object(("kty", "oct"), ("k", "symmetric-fixture"))),
            "\u200b" + json.dumps(runtime_object(("kty", "OKP"), ("d", "private-fixture"))),
        )
        public_jwk_strings = tuple(
            json.dumps(jwk)
            for jwk in (
                {"kty": "EC", "crv": "P-256", "x": "x", "y": "y"},
                {"kty": "OKP", "crv": "Ed25519", "x": "x"},
                {"kty": "RSA", "n": "modulus", "e": "AQAB"},
                {"kty": "ec", "d": "descriptive-dimension"},
                {"kty": "RSA", "p": None},
                {"kty": "oct", "k": ""},
            )
        )
        credential_labels = (
            "".join(("OPENAI", "_API_KEY=fixture-value")),
            "".join(("MLFLOW_TRACKING_PASS", "WORD=fixture-value")),
            "".join(("CUSTOM_ACCESS_TO", "KEN=fixture-value")),
            "".join(("clientSec", "ret=fixture-value")),
            "".join(("ACCESS_TO", "KENS=fixture-value")),
            "".join(("authTo", "kens=fixture-value")),
            "".join(("clientCred", "entials=fixture-value")),
            "".join(("CREDEN", "TIALS=fixture-value")),
            "".join(("SERVICE_ACCOUNT_", "KEY=fixture-value")),
        )
        normalized_obfuscations = (
            "".join(("pass", "word\n=fixture-value")),
            "".join(("Bear", "er\nabc123fixture")),
            "".join(("pass", "word\N{NO-BREAK SPACE}=fixture-value")),
            "".join(("Bear", "er\N{NO-BREAK SPACE}abc123fixture")),
            "".join(("pass", "word\u200b=fixture-value")),
        )
        additional_secret_values = (
            "Bas" + "ic " + "d" * 24,
            "hf" + "_" + "E" * 32,
            github_oauth,
            "sk" + "-proj-" + "F" * 24,
            "sk" + "-admin-" + "J" * 24,
            "xox" + "b-" + "1" * 24,
            "xapp" + "-1-" + "2" * 24,
            "xoxe" + "-1-" + "3" * 24,
            "AK" + "IA" + "2" * 16,
            "".join(("https://fixture-user:", "fixture-password@example.invalid/resource")),
            "".join(("postgresql://fixture-user", ":fixture-password@db.invalid/run")),
            "".join(("redis://fixture-user", ":fixture-password@cache.invalid/0")),
            "".join(("ssh://fixture-user", ":fixture-password@gpu.invalid/home")),
            "".join(("https://", ":fixture-password@example.invalid/simple")),
            "".join(("x://fixture-user", ":fixture-password@host")),
            "".join(("//fixture-user", ":fixture-password@host/path")),
            "".join(("//", ":fixture-password@host/path")),
            "".join(("api", "_key=", "G" * 24)),
            "".join(("API", " key: ", "H" * 24)),
            "".join(("AWS", "_ACCESS_KEY_ID=", "I" * 20)),
            sensitive_phrase,
            key_container,
            unicode_adjacent,
            *service_tokens,
            *credential_labels,
            *normalized_obfuscations,
            *private_jwk_strings,
        )
        for secret_value in additional_secret_values:
            with self.subTest(command="clean_text", value=secret_value[:10]):
                with self.assertRaisesRegex(ticket_tool.TicketError, "secret-bearing"):
                    ticket_tool.clean_text(secret_value, "purpose", 240)
        for normal_value in (
            "hf-transformers-benchmark",
            "sk-learn-classification",
            "No API key is stored by this experiment",
            "TOKEN_IDS=input_ids; TOKEN_EMBEDDING_DIMENSION=4096",
            "N_TOKENS=1024; TOKENS_PER_SECOND=42.5",
            "PROMPT_TOKENS=128; COMPLETION_TOKENS=64",
            "token budget\N{NO-BREAK SPACE}planning note",
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
            *public_jwk_strings,
        ):
            with self.subTest(command="clean_text_normal", value=normal_value):
                self.assertEqual(
                    ticket_tool.clean_text(normal_value, "purpose", 240),
                    " ".join(normal_value.strip().split()),
                )
        reserve_values = {
            "project": github_classic,
            "owner": github_fine_grained,
            "purpose": bearer,
        }
        reserve_defaults = {
            "project": "vit-classification",
            "owner": "test-owner",
            "purpose": "Train and validate the image classifier",
            "gpu_ids": None,
            "gpus": 1,
            "expected": "1h",
            "json": True,
        }
        for field, secret_value in reserve_values.items():
            with self.subTest(command="reserve", field=field):
                args = argparse.Namespace(**{**reserve_defaults, field: secret_value})
                before = copy.deepcopy(self.state)
                with self.assertRaisesRegex(ticket_tool.TicketError, "secret-bearing"):
                    self._silent(
                        ticket_tool.command_reserve, self.state, self.config, args
                    )
                self.assertEqual(self.state, before)

        for secret_value in (
            sensitive_phrase,
            key_container,
            unicode_adjacent,
            *service_tokens,
            *credential_labels,
            *normalized_obfuscations,
            *private_jwk_strings,
        ):
            with self.subTest(command="reserve-boundary", value=secret_value[:10]):
                args = argparse.Namespace(
                    **{**reserve_defaults, "purpose": secret_value}
                )
                before = copy.deepcopy(self.state)
                with self.assertRaisesRegex(ticket_tool.TicketError, "secret-bearing"):
                    self._silent(
                        ticket_tool.command_reserve, self.state, self.config, args
                    )
                self.assertEqual(self.state, before)

        ticket = self._reserve("secret-boundary")
        start_defaults = {
            "ticket_id": ticket["id"],
            "confirmed_idle": "0",
            "session": "vit-training-session",
            "remote_workdir": "/root/secret-boundary/run",
            "summary": "Run the fixed-learning-rate training cycle",
            "expected": None,
        }
        for field, secret_value in {
            "session": github_classic,
            "summary": bearer,
        }.items():
            with self.subTest(command="start", field=field):
                args = argparse.Namespace(**{**start_defaults, field: secret_value})
                before = copy.deepcopy(ticket)
                with self.assertRaisesRegex(ticket_tool.TicketError, "secret-bearing"):
                    self._silent(ticket_tool.command_start, self.state, self.config, args)
                self.assertEqual(ticket, before)

        for secret_value in (
            github_fine_grained,
            *service_tokens,
            *credential_labels,
            *normalized_obfuscations,
            *private_jwk_strings,
        ):
            with self.subTest(command="release", value=secret_value[:10]):
                before = copy.deepcopy(ticket)
                with self.assertRaisesRegex(ticket_tool.TicketError, "secret-bearing"):
                    self._silent(
                        ticket_tool.command_release,
                        self.state,
                        self.config,
                        argparse.Namespace(
                            ticket_id=ticket["id"],
                            outcome="cancelled",
                            confirmed_stopped=None,
                            result=secret_value,
                        ),
                    )
                self.assertEqual(ticket, before)

        persisted = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ticket_tool.STATE_PATH,
                ticket_tool.BOARD_PATH,
                ticket_tool.EVENTS_PATH,
                *ticket_tool.TICKET_DIR.glob("*.md"),
            )
            if path.exists()
        )
        for secret_value in (
            github_classic,
            github_fine_grained,
            bearer,
            sensitive_phrase,
            key_container,
            unicode_adjacent,
            *service_tokens,
            *credential_labels,
            *normalized_obfuscations,
            *private_jwk_strings,
        ):
            self.assertNotIn(secret_value, persisted)

    def test_normal_purpose_summary_and_result_text_remain_accepted(self) -> None:
        ticket = self._reserve("vit-classification")
        self._silent(
            ticket_tool.command_start,
            self.state,
            self.config,
            argparse.Namespace(
                ticket_id=ticket["id"],
                confirmed_idle="0",
                session="vit-training-session",
                remote_workdir="/root/vit-classification/run",
                summary="Run Adam at a fixed learning rate for twenty epochs",
                expected=None,
            ),
        )
        self._silent(
            ticket_tool.command_release,
            self.state,
            self.config,
            argparse.Namespace(
                ticket_id=ticket["id"],
                outcome="completed",
                confirmed_stopped="0",
                result="Validation accuracy recorded; metrics.json and checkpoint verified",
            ),
        )
        self.assertEqual(ticket["purpose"], "TensorBoard state-machine test")
        self.assertEqual(
            ticket["command_summary"],
            "Run Adam at a fixed learning rate for twenty epochs",
        )
        self.assertEqual(
            ticket["result"],
            "Validation accuracy recorded; metrics.json and checkpoint verified",
        )
        persisted = ticket_tool.STATE_PATH.read_text(encoding="utf-8")
        self.assertIn(ticket["command_summary"], persisted)
        self.assertIn(ticket["result"], persisted)

    def test_starting_is_single_writer_and_generation_is_compare_and_set(self) -> None:
        ticket = self._reserve("concurrent")
        self._start_metadata(ticket["id"])
        generation = ticket["tensorboard"]["generation"]
        with self.assertRaisesRegex(ticket_tool.TicketError, "invalid TensorBoard transition"):
            self._start_metadata(ticket["id"], logdir="/root/other/logs")
        with self.assertRaisesRegex(ticket_tool.TicketError, "invalid TensorBoard transition"):
            self._silent(
                ticket_tool.command_tensorboard,
                self.state,
                self.config,
                self._tb_args(
                    ticket["id"],
                    "starting",
                    expected_generation=generation,
                    logdir="/root/tb-test/logs",
                    env_prefix="/root/tb-test/.conda/env",
                    path_prefix=ticket_tool.tensorboard_path_prefix(ticket["id"]),
                    session="pytorch-tb-concurrent-writer",
                ),
            )
        with self.assertRaisesRegex(ticket_tool.TicketError, "generation changed"):
            self._silent(
                ticket_tool.command_tensorboard,
                self.state,
                self.config,
                self._tb_args(
                    ticket["id"],
                    "failed",
                    expected_generation=generation + 1,
                    last_error="simulated failure",
                ),
            )

    def test_legacy_unicode_and_underscore_id_has_one_canonical_url_segment(self) -> None:
        ticket = self._reserve("legacy")
        original_id = ticket["id"]
        legacy_id = "GPU-20260811-120000-abcd-视觉_实验"
        self.state["tickets"].pop(original_id)
        ticket["id"] = legacy_id
        self.state["tickets"][legacy_id] = ticket

        prefix = ticket_tool.tensorboard_path_prefix(legacy_id)
        self.assertEqual(
            prefix,
            "/tb/GPU-20260811-120000-abcd-%E8%A7%86%E8%A7%89_%E5%AE%9E%E9%AA%8C",
        )
        self._start_metadata(
            legacy_id,
            path_prefix=prefix,
            session="pytorch-tb-legacy-unicode",
        )
        self.assertEqual(ticket["tensorboard"]["path_prefix"], prefix)
        loaded = ticket_tool.load_state()
        self.assertIn(legacy_id, loaded["tickets"])

        with self.assertRaisesRegex(ticket_tool.TicketError, "invalid format"):
            ticket_tool.clean_ticket_id("GPU-unsafe/path")

    def test_verified_stop_clears_prior_cleanup_error(self) -> None:
        ticket = self._reserve("cleanup")
        self._start_metadata(ticket["id"])
        generation = ticket["tensorboard"]["generation"]
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket["id"],
                "cleanup_pending",
                expected_generation=generation,
                last_error="simulated uncertain cleanup",
            ),
        )
        self._silent(
            ticket_tool.command_tensorboard,
            self.state,
            self.config,
            self._tb_args(
                ticket["id"], "stopped", expected_generation=generation
            ),
        )
        self.assertIsNone(ticket["tensorboard"]["last_error"])

    def test_old_state_status_load_is_zero_write(self) -> None:
        old = {
            "schema_version": 1,
            "updated_at": "2026-08-11T00:00:00Z",
            "tickets": {
                "GPU-old": {
                    "id": "GPU-old",
                    "status": "completed",
                    "created_at": "2026-08-11T00:00:00Z",
                }
            },
        }
        serialized = json.dumps(old, sort_keys=True) + "\n"
        ticket_tool.STATE_PATH.write_text(serialized, encoding="utf-8")
        before = (
            hashlib.sha256(ticket_tool.STATE_PATH.read_bytes()).hexdigest(),
            ticket_tool.STATE_PATH.stat().st_mtime_ns,
        )
        loaded = ticket_tool.load_state()
        self.assertIsNone(loaded["tickets"]["GPU-old"]["tensorboard"])
        self._silent(
            ticket_tool.command_status,
            loaded,
            self.config,
            argparse.Namespace(ticket_id=None, json=True),
        )
        after = (
            hashlib.sha256(ticket_tool.STATE_PATH.read_bytes()).hexdigest(),
            ticket_tool.STATE_PATH.stat().st_mtime_ns,
        )
        self.assertEqual(after, before)
        self.assertNotIn("tensorboard", json.loads(ticket_tool.STATE_PATH.read_text())["tickets"]["GPU-old"])


if __name__ == "__main__":
    unittest.main()
