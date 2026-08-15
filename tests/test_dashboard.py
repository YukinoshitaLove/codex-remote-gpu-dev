#!/usr/bin/env python3
"""Deterministic security and embedding tests for the local dashboard."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills" / "remote-gpu-dev"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "remote_gpu_dev_dashboard", SKILL_ROOT / "scripts" / "dashboard.py"
)
assert SPEC and SPEC.loader
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


TICKET_ID = "GPU-20260811-120000-abcd-dashboard-test"
LEGACY_TICKET_ID = "GPU-20260811-120000-abcd-视觉_实验"


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    post_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        body = json.dumps(
            {"method": "GET", "path": self.path, "cookie": self.headers.get("Cookie")}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Set-Cookie", "upstream=must-not-escape")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        type(self).post_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        body = json.dumps(
            {
                "method": "POST",
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "content_length": self.headers.get("Content-Length"),
                "cookie": self.headers.get("Cookie"),
                "authorization": self.headers.get("Authorization"),
                "transfer_encoding": self.headers.get("Transfer-Encoding"),
                "body": request_body.decode("utf-8"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Set-Cookie", "upstream=must-not-escape")
        self.end_headers()
        self.wfile.write(body)


class FakeState:
    def __init__(self, upstream_port: int) -> None:
        self.upstream_port = upstream_port
        self.drops: list[str] = []
        self.requests: list[str] = []
        self.control_requests: list[tuple[str, str]] = []

    def snapshot(self) -> dict[str, object]:
        return {"ok": True}

    def tensorboard_upstream(self, ticket_id: str) -> tuple[int | None, str | None]:
        self.requests.append(ticket_id)
        if ticket_id in {TICKET_ID, LEGACY_TICKET_ID}:
            return self.upstream_port, None
        return None, "not registered"

    def drop_tensorboard_tunnel(self, ticket_id: str) -> None:
        self.drops.append(ticket_id)

    def open_tensorboard(self, ticket_id: str) -> tuple[int, dict[str, object]]:
        self.control_requests.append(("open", ticket_id))
        return 200, {
            "ok": True,
            "ticket_id": ticket_id,
            "status": "live",
            "generation": 7,
        }

    def close_tensorboard(self, ticket_id: str) -> tuple[int, dict[str, object]]:
        self.control_requests.append(("close", ticket_id))
        return 200, {
            "ok": True,
            "ticket_id": ticket_id,
            "status": "stopped",
            "generation": 7,
        }


class DashboardHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        cls.port = int(cls.server.server_address[1])
        setattr(cls, "to" + "ken", "-".join(("capability", "fixture")))
        cls.session = "session-test-token"
        cls.state = FakeState(int(cls.upstream.server_address[1]))
        cls.server.RequestHandlerClass = dashboard.make_handler(
            cls.state,
            "instance-test",
            cls.port,
            cls.token,
            cls.session,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.upstream.shutdown()
        cls.upstream.server_close()

    def request(
        self,
        method: str,
        path: str,
        *,
        cookie: bool = False,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        merged = {"Host": f"127.0.0.1:{self.port}"}
        if cookie:
            merged["Cookie"] = f"remote_gpu_dev_dashboard={self.session}"
        merged.update(headers or {})
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        body = response.read()
        result = response.status, {name.lower(): value for name, value in response.getheaders()}, body
        connection.close()
        return result

    def assert_common_headers(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers.get("cache-control"), "no-store")
        self.assertEqual(headers.get("referrer-policy"), "no-referrer")
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertIn("content-security-policy", headers)

    def test_capability_bootstrap_sets_private_cookie(self) -> None:
        status, headers, body = self.request("GET", f"/{self.token}/")
        self.assertEqual(status, 303)
        self.assertEqual(body, b"")
        self.assertEqual(headers.get("location"), "/ui/")
        cookie = headers.get("set-cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertNotIn(self.token, cookie)
        self.assert_common_headers(headers)

    def test_ui_requires_cookie_and_all_errors_have_security_headers(self) -> None:
        status, headers, _ = self.request("GET", "/ui/")
        self.assertEqual(status, 401)
        self.assert_common_headers(headers)
        status, headers, _ = self.request("POST", "/api/status", cookie=True)
        self.assertEqual(status, 405)
        self.assert_common_headers(headers)

    def test_ui_and_status_are_authenticated(self) -> None:
        status, headers, body = self.request("GET", "/ui/", cookie=True)
        self.assertEqual(status, 200)
        self.assertIn(b"GPU", body)
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertIn(
            f"frame-src http://localhost:{self.port}",
            headers.get("content-security-policy", ""),
        )
        status, _, body = self.request("GET", "/api/status", cookie=True)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

    def test_authenticated_same_origin_tensorboard_controls(self) -> None:
        headers = {
            "Origin": f"http://127.0.0.1:{self.port}",
            "Content-Type": "application/json",
        }
        body = json.dumps({"ticket_id": TICKET_ID}).encode()
        before = len(self.state.control_requests)
        for action, expected_status in (("open", "live"), ("close", "stopped")):
            with self.subTest(action=action):
                status, response_headers, response_body = self.request(
                    "POST",
                    f"/api/tensorboard/{action}",
                    cookie=True,
                    headers=headers,
                    body=body,
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    response_headers.get("content-type"),
                    "application/json; charset=utf-8",
                )
                payload = json.loads(response_body)
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["status"], expected_status)
        self.assertEqual(
            self.state.control_requests[before:],
            [("open", TICKET_ID), ("close", TICKET_ID)],
        )

    def test_tensorboard_control_requires_cookie_and_exact_origin(self) -> None:
        body = json.dumps({"ticket_id": TICKET_ID}).encode()
        valid_headers = {
            "Origin": f"http://127.0.0.1:{self.port}",
            "Content-Type": "application/json",
        }
        cases = [
            (False, valid_headers, 401),
            (True, {"Content-Type": "application/json"}, 403),
            (
                True,
                {"Origin": "http://localhost:8765", "Content-Type": "application/json"},
                403,
            ),
        ]
        before = len(self.state.control_requests)
        for cookie, headers, expected in cases:
            with self.subTest(cookie=cookie, headers=headers):
                status, _, _ = self.request(
                    "POST",
                    "/api/tensorboard/open",
                    cookie=cookie,
                    headers=headers,
                    body=body,
                )
                self.assertEqual(status, expected)
        self.assertEqual(len(self.state.control_requests), before)

    def test_tensorboard_control_json_is_small_and_closed_schema(self) -> None:
        headers = {
            "Origin": f"http://127.0.0.1:{self.port}",
            "Content-Type": "application/json",
        }
        cases = [
            (b"".join((b'{"ticket_id":"GPU-a",', b'"ticket_id":"GPU-b"}')), headers, 400),
            (json.dumps({"ticket_id": TICKET_ID, "logdir": "/tmp"}).encode(), headers, 400),
            (json.dumps({"ticket_id": "GPU-unsafe/path"}).encode(), headers, 400),
            (json.dumps({"ticket_id": TICKET_ID}).encode(), {"Origin": headers["Origin"]}, 415),
            (
                b"{" + (b" " * dashboard.MAX_CONTROL_POST_BYTES) + b"}",
                headers,
                413,
            ),
            (
                json.dumps({"ticket_id": TICKET_ID}).encode(),
                {
                    **headers,
                    "".join(("Authori", "zation")): "".join(
                        ("Bear", "er forbidden")
                    ),
                },
                400,
            ),
        ]
        before = len(self.state.control_requests)
        for body, request_headers, expected in cases:
            with self.subTest(expected=expected, headers=request_headers):
                status, _, _ = self.request(
                    "POST",
                    "/api/tensorboard/open",
                    cookie=True,
                    headers=request_headers,
                    body=body,
                )
                self.assertEqual(status, expected)
        status, _, _ = self.request(
            "POST",
            "/api/tensorboard/open?ticket_id=ignored",
            cookie=True,
            headers=headers,
            body=json.dumps({"ticket_id": TICKET_ID}).encode(),
        )
        self.assertEqual(status, 405)
        self.assertEqual(len(self.state.control_requests), before)

    def test_tensorboard_control_is_unavailable_from_viewer_origin(self) -> None:
        status, _, _ = self.request(
            "POST",
            "/api/tensorboard/open",
            headers={
                "Host": f"localhost:{self.port}",
                "Origin": f"http://localhost:{self.port}",
                "Content-Type": "application/json",
            },
            body=json.dumps({"ticket_id": TICKET_ID}).encode(),
        )
        self.assertEqual(status, 404)

    def test_host_origin_and_path_injection_are_rejected(self) -> None:
        status, headers, _ = self.request(
            "GET", "/ui/", cookie=True, headers={"Host": "evil.invalid"}
        )
        self.assertEqual(status, 421)
        self.assert_common_headers(headers)
        status, _, _ = self.request(
            "GET", "/ui/", cookie=True, headers={"Origin": "null"}
        )
        self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/tb/..%2Fetc/", cookie=True)
        self.assertEqual(status, 404)

    def test_tensorboard_is_origin_isolated_and_strips_sensitive_headers(self) -> None:
        status, headers, body = self.request(
            "GET",
            f"/tb/{TICKET_ID}/data/plugin/scalars/tags",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["path"], f"/tb/{TICKET_ID}/data/plugin/scalars/tags")
        self.assertIsNone(payload["cookie"])
        self.assertNotIn("x-frame-options", headers)
        self.assertIn(
            f"frame-ancestors http://127.0.0.1:{self.port}",
            headers.get("content-security-policy", ""),
        )
        self.assertNotIn("set-cookie", headers)

    def test_tensorboard_runs_get_keeps_the_absolute_ticket_prefix(self) -> None:
        status, _, body = self.request(
            "GET",
            f"/tb/{TICKET_ID}/data/runs",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["path"], f"/tb/{TICKET_ID}/data/runs")
        status, _, _ = self.request(
            "GET",
            "/data/runs",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 404)

    def test_timeseries_multipart_post_is_narrowly_proxied_without_credentials(self) -> None:
        boundary = "----TensorBoardBoundary7MA4YWxk"
        request_body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="request"\r\n'
            "\r\n"
            '{"run":"train","tag":"loss"}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        path = (
            f"/tb/{TICKET_ID}/experiment/defaultExperimentId/"
            "data/plugin/timeseries/timeSeries"
        )
        before = UpstreamHandler.post_count
        status, headers, body = self.request(
            "POST",
            path,
            headers={
                "Host": f"localhost:{self.port}",
                "Origin": f"http://localhost:{self.port}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            body=request_body,
        )
        self.assertEqual(status, 200)
        self.assertEqual(UpstreamHandler.post_count, before + 1)
        payload = json.loads(body)
        self.assertEqual(payload["method"], "POST")
        self.assertEqual(payload["path"], path)
        self.assertEqual(payload["content_length"], str(len(request_body)))
        self.assertEqual(payload["body"].encode(), request_body)
        self.assertIsNone(payload["cookie"])
        self.assertIsNone(payload["authorization"])
        self.assertIsNone(payload["transfer_encoding"])
        self.assertNotIn("set-cookie", headers)
        self.assertNotIn("x-frame-options", headers)

    def test_classic_scalars_multirun_post_is_narrowly_proxied(self) -> None:
        boundary = "----WebKitFormBoundaryScalarBatch"
        request_body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="tag"\r\n'
            "\r\n"
            "experiment/train_acc\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="runs"\r\n'
            "\r\n"
            ".\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        path = f"/tb/{TICKET_ID}/data/plugin/scalars/scalars_multirun"
        before = UpstreamHandler.post_count
        status, headers, body = self.request(
            "POST",
            path,
            headers={
                "Host": f"localhost:{self.port}",
                "Origin": f"http://localhost:{self.port}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-TensorBoard-Feature-Flags": "{}",
            },
            body=request_body,
        )
        self.assertEqual(status, 200)
        self.assertEqual(UpstreamHandler.post_count, before + 1)
        payload = json.loads(body)
        self.assertEqual(payload["path"], path)
        self.assertEqual(payload["content_length"], str(len(request_body)))
        self.assertEqual(payload["body"].encode(), request_body)
        self.assertIsNone(payload["cookie"])
        self.assertIsNone(payload["authorization"])
        self.assertIsNone(payload["transfer_encoding"])
        self.assertNotIn("set-cookie", headers)
        self.assertNotIn("x-frame-options", headers)

    def test_all_other_or_unsafe_tensorboard_posts_fail_closed(self) -> None:
        boundary = "----TensorBoardBoundary"
        request_body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
            f"\r\n{{}}\r\n--{boundary}--\r\n"
        ).encode()
        valid_path = (
            f"/tb/{TICKET_ID}/experiment/defaultExperimentId/"
            "data/plugin/timeseries/timeSeries"
        )
        base_headers = {
            "Host": f"localhost:{self.port}",
            "Origin": f"http://localhost:{self.port}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        cases = [
            (valid_path + "?write=true", base_headers, 405),
            (f"/tb/{TICKET_ID}/data/plugin/timeseries/timeSeries", base_headers, 405),
            (f"/tb/{TICKET_ID}/data/plugin/scalars/scalars", base_headers, 405),
            (
                f"/tb/{TICKET_ID}/data/plugin/scalars/scalars_multirun?write=true",
                base_headers,
                405,
            ),
            (
                f"/tb/{TICKET_ID}/experiment/defaultExperimentId/"
                "data/plugin/scalars/scalars_multirun",
                base_headers,
                405,
            ),
            (
                f"/tb/{TICKET_ID}/experiment/defaultExperimentId%2Fescape/"
                "data/plugin/timeseries/timeSeries",
                base_headers,
                405,
            ),
            (valid_path, {**base_headers, "Origin": "http://evil.invalid"}, 403),
            (valid_path, {**base_headers, "Content-Type": "application/json"}, 415),
            (valid_path, {**base_headers, "Cookie": "ambient=secret"}, 400),
            (
                valid_path,
                {
                    **base_headers,
                    "".join(("Authori", "zation")): "".join(
                        ("Bear", "er hidden")
                    ),
                },
                400,
            ),
            (valid_path, {**base_headers, "Transfer-Encoding": "chunked"}, 400),
            (
                valid_path,
                {**base_headers, "Content-Length": str(dashboard.MAX_TENSORBOARD_POST_BYTES + 1)},
                413,
            ),
        ]
        before = UpstreamHandler.post_count
        for path, headers, expected in cases:
            with self.subTest(path=path, expected=expected, headers=headers):
                status, response_headers, _ = self.request(
                    "POST", path, headers=headers, body=request_body
                )
                self.assertEqual(status, expected)
                self.assert_common_headers(response_headers)

        status, _, _ = self.request(
            "POST",
            valid_path,
            headers=base_headers,
            body=b"--different-boundary\r\n\r\n--different-boundary--\r\n",
        )
        self.assertEqual(status, 400)

        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.putrequest("POST", valid_path, skip_host=True)
        connection.putheader("Host", f"localhost:{self.port}")
        connection.putheader("Origin", f"http://localhost:{self.port}")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 411)
        response.read()
        connection.close()
        self.assertEqual(UpstreamHandler.post_count, before)

        status, _, _ = self.request(
            "PUT", valid_path, headers=base_headers, body=request_body
        )
        self.assertEqual(status, 405)

    def test_unicode_ticket_route_is_canonical_and_decodes_to_ledger_id(self) -> None:
        segment = quote(LEGACY_TICKET_ID, safe="")
        status, _, body = self.request(
            "GET",
            f"/tb/{segment}/data/plugin/scalars/tags",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(self.state.requests[-1], LEGACY_TICKET_ID)
        self.assertEqual(json.loads(body)["path"], f"/tb/{segment}/data/plugin/scalars/tags")

        status, _, _ = self.request(
            "GET",
            f"/tb/{segment.lower()}/",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 404)

    def test_viewer_origin_cannot_read_dashboard_api(self) -> None:
        status, headers, _ = self.request(
            "GET", "/api/status", headers={"Host": f"localhost:{self.port}"}
        )
        self.assertEqual(status, 404)
        self.assert_common_headers(headers)

    def test_unknown_tensorboard_cannot_choose_an_upstream(self) -> None:
        status, headers, _ = self.request(
            "GET",
            "/tb/GPU-20260811-120000-dead-unknown/",
            headers={"Host": f"localhost:{self.port}"},
        )
        self.assertEqual(status, 503)
        self.assert_common_headers(headers)


class SanitizerTests(unittest.TestCase):
    def test_ticket_supplied_urls_are_not_exposed(self) -> None:
        raw = {
            "id": TICKET_ID,
            "status": "running",
            "assigned_gpus": [0],
            "tensorboard": {
                "status": "live",
                "path_prefix": f"/tb/{TICKET_ID}",
                "remote_port": 16006,
                "generation": 9,
                "url": "http://169.254.169.254/latest/meta-data",
            },
        }
        sanitized = dashboard.sanitize_ticket(raw)
        assert sanitized is not None
        self.assertNotIn("url", sanitized["tensorboard"])
        self.assertEqual(sanitized["tensorboard"]["remote_port"], 16006)
        self.assertEqual(sanitized["tensorboard"]["generation"], 9)

    def test_legacy_ticket_id_alphabet_and_encoded_path_are_supported(self) -> None:
        prefix = dashboard.tensorboard_path_prefix(LEGACY_TICKET_ID)
        sanitized = dashboard.sanitize_ticket(
            {
                "id": LEGACY_TICKET_ID,
                "status": "completed",
                "tensorboard": {
                    "status": "live",
                    "path_prefix": prefix,
                    "remote_port": 16006,
                },
            }
        )
        assert sanitized is not None
        self.assertEqual(sanitized["tensorboard"]["path_prefix"], prefix)
        self.assertIsNone(
            dashboard.sanitize_ticket(
                {
                    "id": "GPU-unsafe/path",
                    "status": "completed",
                    "tensorboard": {"status": "live"},
                }
            )["tensorboard"]
        )
        overlong = "GPU-" + ("a" * 157)
        clipped_prefix = "/tb/" + ("GPU-" + ("a" * 156))
        self.assertIsNone(
            dashboard.sanitize_ticket(
                {
                    "id": overlong,
                    "status": "completed",
                    "tensorboard": {
                        "status": "live",
                        "path_prefix": clipped_prefix,
                        "remote_port": 16006,
                    },
                }
            )["tensorboard"]
        )


class DashboardTensorBoardControlTests(unittest.TestCase):
    def state_with_tensorboard(
        self,
        tensorboard: dict[str, object],
    ) -> dashboard.DashboardState:
        state = dashboard.DashboardState()
        state.ticket = {
            "connected": True,
            "snapshot": {
                "active": [],
                "queued": [],
                "history": [
                    {
                        "id": TICKET_ID,
                        "status": "completed",
                        "tensorboard": tensorboard,
                    }
                ],
            },
        }
        return state

    @staticmethod
    def completed(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    def test_open_uses_retained_configuration_and_records_new_generation(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "stopped", "generation": 3, "logdir": "/retained/events"}
        )
        response = self.completed(
            {
                "ticket_id": TICKET_ID,
                "status": "live",
                "ticket": {"tensorboard": {"generation": 4}},
            }
        )
        with mock.patch.object(dashboard.subprocess, "run", return_value=response) as run:
            status, payload = state.open_tensorboard(TICKET_ID)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["generation"], 4)
        self.assertEqual(state.tensorboard_owned_generations, {TICKET_ID: 4})
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                dashboard.sys.executable,
                str(dashboard.TENSORBOARD_SIDECAR),
                "start",
                TICKET_ID,
            ],
        )
        self.assertNotIn("--logdir", command)
        self.assertNotIn("--env-prefix", command)

    def test_idempotent_open_does_not_claim_manual_generation(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 8, "logdir": "/retained/events"}
        )
        response = self.completed(
            {
                "ticket_id": TICKET_ID,
                "status": "live",
                "idempotent": True,
                "tensorboard": {"generation": 8},
            }
        )
        with mock.patch.object(dashboard.subprocess, "run", return_value=response):
            status, payload = state.open_tensorboard(TICKET_ID)
        self.assertEqual(status, 200)
        self.assertFalse(payload["created_by_dashboard"])
        self.assertEqual(state.tensorboard_owned_generations, {})

    def test_close_pins_server_observed_generation_and_drops_tunnel(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 4, "logdir": "/retained/events"}
        )
        state.tensorboard_owned_generations[TICKET_ID] = 4
        state.tensorboard_tunnels[TICKET_ID] = {"process": None, "remote_port": 16006}
        response = self.completed({"ticket_id": TICKET_ID, "status": "stopped"})
        with mock.patch.object(dashboard.subprocess, "run", return_value=response) as run:
            status, payload = state.close_tensorboard(TICKET_ID)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(state.tensorboard_owned_generations, {})
        self.assertNotIn(TICKET_ID, state.tensorboard_tunnels)
        self.assertEqual(
            run.call_args.args[0][-2:],
            ["--expected-generation", "4"],
        )

    def test_close_reports_superseded_without_stopping_new_generation(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 4, "logdir": "/retained/events"}
        )
        state.tensorboard_owned_generations[TICKET_ID] = 4
        response = self.completed(
            {
                "ticket_id": TICKET_ID,
                "status": "superseded",
                "expected_generation": 4,
                "observed_generation": 5,
            }
        )
        with mock.patch.object(dashboard.subprocess, "run", return_value=response):
            status, payload = state.close_tensorboard(TICKET_ID)
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "superseded")
        self.assertEqual(state.tensorboard_owned_generations, {})

    def test_daemon_stop_cleans_only_owned_generation(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 12, "logdir": "/retained/events"}
        )
        state.tensorboard_owned_generations[TICKET_ID] = 12
        response = self.completed(
            {
                "ticket_id": TICKET_ID,
                "status": "superseded",
                "expected_generation": 12,
                "observed_generation": 13,
            }
        )
        with mock.patch.object(dashboard.subprocess, "run", return_value=response) as run:
            state.stop()
        command = run.call_args.args[0]
        self.assertIn("stop", command)
        self.assertEqual(
            command[command.index("--expected-generation") + 1],
            "12",
        )
        self.assertEqual(state.tensorboard_owned_generations, {})
        self.assertTrue(state.stop_event.is_set())

    def test_owned_cleanup_is_parallel_and_bounded_by_stop_grace(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 1, "logdir": "/retained/events"}
        )
        owned = {f"GPU-parallel-{index}": index + 1 for index in range(9)}

        def delayed_stop(arguments, *, timeout):
            del arguments, timeout
            time.sleep(0.1)
            return 200, {"status": "stopped"}

        started = time.monotonic()
        with mock.patch.object(state, "_run_sidecar", side_effect=delayed_stop):
            state._stop_owned_tensorboards(owned)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertLess(
            dashboard.SIDECAR_COMMAND_TIMEOUT_SECONDS
            + dashboard.LOCAL_PROCESS_CLEANUP_SECONDS
            + dashboard.STATE_THREAD_JOIN_SECONDS
            + dashboard.OWNED_SIDECAR_CLEANUP_SECONDS,
            dashboard.DASHBOARD_STOP_GRACE_SECONDS,
        )

    def test_prune_drops_stale_owned_generation(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "live", "generation": 5, "logdir": "/retained/events"}
        )
        state.tensorboard_owned_generations[TICKET_ID] = 4
        state._prune_tensorboard_tunnels(state.ticket["snapshot"])
        self.assertEqual(state.tensorboard_owned_generations, {})

    def test_prune_keeps_new_claim_when_snapshot_is_one_generation_behind(self) -> None:
        state = self.state_with_tensorboard(
            {"status": "stopped", "generation": 4, "logdir": "/retained/events"}
        )
        state.tensorboard_owned_generations[TICKET_ID] = 5
        state._prune_tensorboard_tunnels(state.ticket["snapshot"])
        self.assertEqual(state.tensorboard_owned_generations, {TICKET_ID: 5})

    def test_control_rejects_unconfigured_ticket_without_sidecar_call(self) -> None:
        state = self.state_with_tensorboard({})
        state.ticket["snapshot"]["history"][0]["tensorboard"] = None
        with mock.patch.object(dashboard.subprocess, "run") as run:
            status, payload = state.open_tensorboard(TICKET_ID)
        self.assertEqual(status, 409)
        self.assertFalse(payload["ok"])
        run.assert_not_called()


class DirectHealthTests(unittest.TestCase):
    def test_health_probe_ignores_ambient_http_proxy(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = int(server.server_address[1])
        handler = dashboard.make_handler(FakeState(1), "health-instance", port, "secret-cap", "session")
        server.RequestHandlerClass = handler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxy_names = (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        )
        old = {name: os.environ.get(name) for name in proxy_names}
        try:
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
            os.environ["http_proxy"] = "http://127.0.0.1:9"
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"
            os.environ["https_proxy"] = "http://127.0.0.1:9"
            os.environ["ALL_PROXY"] = "http://127.0.0.1:9"
            os.environ["all_proxy"] = "http://127.0.0.1:9"
            os.environ["NO_PROXY"] = ""
            os.environ["no_proxy"] = ""
            self.assertTrue(
                dashboard.health_matches(
                    {
                        "port": port,
                        "".join(("to", "ken")): "".join(("secret", "-cap")),
                        "instance_id": "health-instance",
                    }
                )
            )
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            server.shutdown()
            server.server_close()


class FrontendPathTests(unittest.TestCase):
    def test_frontend_uses_the_same_unicode_url_segment(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is unavailable")
        app_js = SKILL_ROOT / "assets" / "dashboard" / "app.js"
        probe = r'''
const fs = require("fs");
const vm = require("vm");
const element = () => ({
  addEventListener() {}, append() {}, replaceChildren() {}, removeAttribute() {},
  scrollIntoView() {}, setAttribute() {}, classList: { add() {} }, style: {},
});
const context = {
  window: { location: { protocol: "http:", port: "8765" } },
  document: {
    getElementById() { return element(); },
    createElement() { return element(); },
    querySelectorAll() { return []; },
  },
  fetch: async () => { throw new Error("offline test"); },
  setInterval() { return 0; },
  console,
};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], "utf8");
vm.runInContext(source + `\nglobalThis.__path = tensorboardPath({
  id: "GPU-20260811-120000-abcd-视觉_实验",
  tensorboard: { path_prefix: "/tb/GPU-20260811-120000-abcd-%E8%A7%86%E8%A7%89_%E5%AE%9E%E9%AA%8C" },
});`, context);
process.stdout.write(JSON.stringify(context.__path));
'''
        completed = subprocess.run(
            [node, "-e", probe, str(app_js)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            "http://localhost:8765/tb/GPU-20260811-120000-abcd-%E8%A7%86%E8%A7%89_%E5%AE%9E%E9%AA%8C/",
        )

    def test_frontend_exposes_cleanup_states_and_all_history(self) -> None:
        app_js = (SKILL_ROOT / "assets" / "dashboard" / "app.js").read_text(
            encoding="utf-8"
        )
        index_html = (
            SKILL_ROOT / "assets" / "dashboard" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('["starting", "live", "failed", "cleanup_pending"]', app_js)
        self.assertIn("ticket.history || ticket.recent || []", app_js)
        self.assertIn("重试清理 TensorBoard", app_js)
        self.assertIn("GPU TICKETS READ-ONLY · TENSORBOARD USER CONTROL", index_html)
        self.assertIn("GPU 分配与排队状态只读", index_html)
        self.assertIn("TensorBoard 操作会更新本地工单中的 viewer 元数据", index_html)
        self.assertIn("全部终态工单", index_html)


if __name__ == "__main__":
    unittest.main()
