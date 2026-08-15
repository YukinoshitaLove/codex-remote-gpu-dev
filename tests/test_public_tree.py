#!/usr/bin/env python3
"""Public release-tree scanner regression tests."""

from __future__ import annotations

import ast
import binascii
import importlib.util
import json
import struct
import tempfile
import time
import unittest
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_public_tree_under_test", REPO_ROOT / "tools" / "check_public_tree.py"
)
assert CHECK_SPEC is not None and CHECK_SPEC.loader is not None
check_public_tree = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(check_public_tree)


def runtime_text(*parts: str) -> str:
    return "".join(parts)


def runtime_object(*pairs: tuple[str, object]) -> dict[str, object]:
    return dict(pairs)


def _dynamic_concat_chain(size: int) -> ast.Module:
    """Return the tree for ``payload = dynamic()`` followed by ``size`` ``+'x'``."""

    value: ast.expr = ast.Call(
        func=ast.Name(id="dynamic", ctx=ast.Load()), args=[], keywords=[]
    )
    for _ in range(size):
        value = ast.BinOp(left=value, op=ast.Add(), right=ast.Constant(value="x"))
    return ast.Module(
        body=[
            ast.Assign(
                targets=[ast.Name(id="payload", ctx=ast.Store())], value=value
            )
        ],
        type_ignores=[],
    )


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def public_png(
    *,
    width: int = 2,
    height: int = 2,
    color_type: int = 2,
    extra_chunks: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    channels = 3 if color_type == 2 else 4
    row = b"\x00" + (b"\x7f" * (width * channels))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0),
        )
        + b"".join(png_chunk(kind, value) for kind, value in extra_chunks)
        + png_chunk(b"IDAT", zlib.compress(row * height))
        + png_chunk(b"IEND", b"")
    )


class PublicTreeTests(unittest.TestCase):
    def test_metadata_free_rgb_and_rgba_pngs_are_accepted(self) -> None:
        for color_type in (2, 6):
            with (
                self.subTest(color_type=color_type),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "diagram.png").write_bytes(public_png(color_type=color_type))
                self.assertEqual(check_public_tree.check(root), [])

    def test_png_spoofs_corruption_and_unsafe_metadata_are_rejected(self) -> None:
        valid = public_png()
        corrupt_crc = bytearray(valid)
        corrupt_crc[-5] ^= 0x01
        fixtures = {
            "signature": b"not a png",
            "truncated": valid[:-3],
            "crc": bytes(corrupt_crc),
            "trailing": valid + b"trailing",
            "text": public_png(extra_chunks=((b"tEXt", b"note\x00value"),)),
            "compressed-text": public_png(extra_chunks=((b"zTXt", b"note\x00\x00"),)),
            "international-text": public_png(extra_chunks=((b"iTXt", b"note"),)),
            "exif": public_png(extra_chunks=((b"eXIf", b"fixture"),)),
            "icc": public_png(extra_chunks=((b"iCCP", b"fixture"),)),
            "c2pa": public_png(extra_chunks=((b"caBX", b"fixture"),)),
        }
        for name, value in fixtures.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "diagram.png").write_bytes(value)
                errors = check_public_tree.check(root)
                self.assertTrue(any("invalid public PNG" in error for error in errors), errors)

    def test_png_dimension_and_pixel_limits_are_rejected(self) -> None:
        fixtures = {
            "wide": public_png(width=check_public_tree.MAX_PUBLIC_IMAGE_EDGE + 1),
            "tall": public_png(height=check_public_tree.MAX_PUBLIC_IMAGE_EDGE + 1),
            "pixels": public_png(width=3000, height=3000),
        }
        for name, value in fixtures.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "diagram.png").write_bytes(value)
                errors = check_public_tree.check(root)
                self.assertTrue(any("invalid public PNG" in error for error in errors), errors)

    def test_unknown_binary_remains_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "diagram.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            errors = check_public_tree.check(root)
            self.assertTrue(errors)
            self.assertTrue(all("diagram.webp" in error for error in errors), errors)

    def test_extensionless_private_key_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key_container_header = runtime_text(
                "-----BEGIN OPENSSH ", "PRIVATE KEY-----"
            )
            (root / "id_ed25519").write_text(
                key_container_header + "\nfixture\n", encoding="utf-8"
            )
            errors = check_public_tree.check(root)
            self.assertTrue(any("secret-like" in error for error in errors), errors)

    def test_common_private_key_formats_are_scanned(self) -> None:
        headers = {
            "pkcs8": runtime_text("-----BEGIN ", "PRIVATE KEY-----"),
            "encrypted-pkcs8": runtime_text(
                "-----BEGIN ENCRYPTED ", "PRIVATE KEY-----"
            ),
            "openpgp": runtime_text("-----BEGIN PGP ", "PRIVATE KEY BLOCK-----"),
            "ssh2": runtime_text("---- BEGIN SSH2 ENCRYPTED ", "PRIVATE KEY ----"),
            "putty": runtime_text("PuTTY-User-", "Key-File-3: ssh-ed25519"),
            "age": runtime_text("AGE-SECRET-", "KEY-1ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"),
        }
        for name, header in headers.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_text(header + "\nfixture\n", encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

    def test_unknown_text_extension_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sensitive_field_name = "pass" + "word"
            (root / "credentials.ini").write_text(
                f'{sensitive_field_name}="fixture-secret"\n', encoding="utf-8"
            )
            errors = check_public_tree.check(root)
            self.assertTrue(any("secret-like" in error for error in errors), errors)

    def test_service_tokens_are_scanned_without_rejecting_project_names(self) -> None:
        github_classic = "ghp" + "_" + "A" * 32
        secrets = (
            "gho" + "_" + "0" * 24,
            "xox" + "b-" + "1" * 24,
            "AK" + "IA" + "2" * 16,
            "sk" + "-proj-" + "A" * 24,
            "sk" + "-admin-" + "C" * 24,
            "xapp" + "-1-" + "4" * 24,
            "xoxe" + "-1-" + "5" * 24,
            runtime_text("API", " key=\"", "B" * 24, "\""),
            "说明" + github_classic + "结束",
            "GHP" + "_" + "C" * 32,
            "sk" + "-ant-api03-" + "D" * 48,
            "gl" + "pat-" + "E" * 20,
            "pypi-" + "AgEIcHlwaS5vcmc" + "F" * 32,
            "npm" + "_" + "G" * 36,
            "sk" + "_live_" + "H" * 24,
            "AI" + "za" + "I" * 35,
            "ya" + "29." + "J" * 40,
            runtime_text("postgresql://fixture-user", ":fixture-password@db.invalid/run"),
            runtime_text("redis://fixture-user", ":fixture-password@cache.invalid/0"),
            runtime_text("ssh://fixture-user", ":fixture-password@gpu.invalid/home"),
            runtime_text("https://", ":fixture-password@example.invalid/simple"),
            runtime_text("x://fixture-user", ":fixture-password@host"),
            runtime_text("//fixture-user", ":fixture-password@host/path"),
            runtime_text("//", ":fixture-password@host/path"),
        )
        for index, value in enumerate(secrets):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "README.md").write_text(value + "\n", encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

        for value in (
            "hf-transformers-benchmark",
            "sk-learn-classification",
            "AIza-documentation-placeholder",
            "ya29.oauth-flow-name",
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
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "README.md").write_text(value + "\n", encoding="utf-8")
                self.assertEqual(check_public_tree.check(root), [])

    def test_unquoted_credential_assignments_are_scanned(self) -> None:
        assignments = (
            runtime_text("AWS", "_SECRET_ACCESS_KEY=", "a" * 40),
            runtime_text("api", "_key=", "b" * 24),
            runtime_text("pass", "phrase=fixture-correct-horse"),
            runtime_text("pass", "word=correct!horse"),
            runtime_text("TO", "KEN=abcdef$ghijk"),
            runtime_text("pass", "wd=abc12"),
            runtime_text("OPENAI", "_API_KEY=fixture-value"),
            runtime_text("MLFLOW_TRACKING_PASS", "WORD=fixture-value"),
            runtime_text("CUSTOM_ACCESS_TO", "KEN=fixture-value"),
            runtime_text("clientSec", "ret=fixture-value"),
            runtime_text("ACCESS_TO", "KENS=fixture-value"),
            runtime_text("authTo", "kens=fixture-value"),
            runtime_text("clientCred", "entials=fixture-value"),
            runtime_text("CREDEN", "TIALS=fixture-value"),
            runtime_text("SERVICE_ACCOUNT_", "KEY=fixture-value"),
            runtime_text("service account ", "key=fixture-value"),
            runtime_text("pass", "word\n=fixture-value"),
            runtime_text("client\\\nSec", "ret=fixture-value"),
            runtime_text("pass", "word\u200b=fixture-value"),
        )
        for index, value in enumerate(assignments):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config.sh").write_text(value + "\n", encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

    def test_assignment_scanner_preserves_normal_text_and_dynamic_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text(
                "passwordless login; token bucket; TOKENIZERS_PARALLELISM=false\n"
                "MAX_TOKEN_COUNT=2048; TOKEN_BUDGET=4096; LOSS_TOKEN_WEIGHT=0.25\n"
                "MAX_NEW_TOKENS=256; PAD_TOKEN=pad; EOS_TOKEN=eos\n"
                "BOS_TOKEN=bos; MASK_TOKEN=mask\n"
                "TOKEN_IDS=input_ids; TOKEN_EMBEDDING_DIMENSION=4096\n"
                "HF_HUB_DISABLE_IMPLICIT_TOKEN=true\n"
                "N_TOKENS=1024; TOKENS_PER_SECOND=42.5\n"
                "PROMPT_TOKENS=128; COMPLETION_TOKENS=64\n"
                "SERVICE_ACCOUNT_KEY_ID=fixture-id; SERVICE_ACCOUNT_KEY_COUNT=2\n"
                "SERVICE_ACCOUNT_KEY_ROTATION_INTERVAL=86400\n",
                encoding="utf-8",
            )
            (root / "safe.py").write_text(
                runtime_text(
                    "to",
                    "ken = build_capability()\napi",
                    "_key: str = load_description()\npass",
                    "word = (describe_setting())\nos.environ['OPENAI_",
                    "API_KEY'] = load_description()\nconfig['pass",
                    "word'] = describe_setting()\nclientSec",
                    "ret = describe_setting()\nauthTo",
                    "kens = describe_setting()\nTOKEN_IDS = 'input_ids'\n",
                    "TOKEN_EMBEDDING_DIMENSION = '4096'\nMAX_NEW_TOKENS = '256'\n",
                    "PAD_TOKEN = 'pad'\nEOS_TOKEN = 'eos'\nBOS_TOKEN = 'bos'\n",
                    "MASK_TOKEN = 'mask'\nN_TOKENS = '1024'\n",
                    "TOKENS_PER_SECOND = '42.5'\nPROMPT_TOKENS = '128'\n",
                    "COMPLETION_TOKENS = '64'\nCONFIG = {'pass",
                    "word': load_description()}\nconnect(api",
                    "_key=load_description())\ndef dynamic(pass",
                    "word=load_description(), *, api",
                    "_key=load_description()):\n    return pass",
                    "word, api_key\nasync def dynamic_async(*, authTo",
                    "kens=load_description()):\n    return authTokens\n",
                    "suffix = load_description()\npass",
                    "word = f'opaque-{suffix}'\napi",
                    "_key = 'opaque-' + suffix\nCONFIG_DYNAMIC = {'pass",
                    "word': f'opaque-{suffix}'}\nconnect(api",
                    "_key='opaque-' + suffix)\ndef dynamic_joined(pass",
                    "word=f'opaque-{suffix}'):\n    return password\n",
                    "token_budget: str = 'auto'\n",
                    "one = os.getenv('OPENAI_",
                    "API_KEY')\ntwo = os.getenv('OPENAI_",
                    "API_KEY', load_description())\nthree = os.environ.get('MLFLOW_TRACKING_PASS",
                    "WORD', default=load_description())\nfour = os.getenv('OPENAI_",
                    "API_KEY', '')\n",
                ),
                encoding="utf-8",
            )
            (root / "app.js").write_text(
                runtime_text('fetch("/status", {creden', 'tials: "same-origin"});\n'),
                encoding="utf-8",
            )
            self.assertEqual(check_public_tree.check(root), [])

            bad_sources = (
                runtime_text("to", 'ken = "abc12"\n'),
                runtime_text("api", '_key: str = "fixture-value"\n'),
                runtime_text("pass", 'word = ("fixture-value")\n'),
                runtime_text("OPENAI", '_API_KEY: str = "fixture-value"\n'),
                runtime_text('os.environ["OPENAI_', 'API_KEY"] = "fixture-value"\n'),
                runtime_text('config["pass', 'word"] = "fixture-value"\n'),
                runtime_text("to", 'ken: Literal["opaque"] = "fixture-value"\n'),
                runtime_text("pass", 'word = \\\n    ("fixture-value")\n'),
                runtime_text("pass", 'word = (\n    "fixture-value"\n)\n'),
                runtime_text("clientSec", 'ret = "fixture-value"\n'),
                runtime_text("authTo", 'kens = "fixture-value"\n'),
                runtime_text("CONFIG = {\"pass", 'word\": \"fixture-value\"}\n'),
                runtime_text("connect(api", '_key="fixture-value")\n'),
                runtime_text("def connect(pass", 'word="fixture-value"):\n    pass\n'),
                runtime_text("def connect(*, api", '_key="fixture-value"):\n    pass\n'),
                runtime_text("async def connect(*, authTo", 'kens="fixture-value"):\n    pass\n'),
                runtime_text("pass", 'word = f"fixture-value"\n'),
                runtime_text("api", '_key = "fixture-" + "value"\n'),
                runtime_text("CONFIG = {\"pass", 'word\": f"fixture-{\'value\'}"}\n'),
                runtime_text("connect(api", '_key="fixture-" + "value")\n'),
                runtime_text("def connect(pass", 'word=f"fixture-{\'value\'}"):\n    pass\n'),
                runtime_text("pass\u200b", 'word = "fixture-value"\n'),
                runtime_text("CONFIG = {\"pass\u200b", 'word\": "fixture-value"}\n'),
                runtime_text("SERVICE_ACCOUNT_", 'KEY = "fixture-value"\n'),
                runtime_text(
                    "value = os.getenv('OPENAI_",
                    "API_KEY', 'fixture-default')\n",
                ),
                runtime_text(
                    "value = os.environ.get('MLFLOW_TRACKING_PASS",
                    "WORD', 'fixture-' + 'default')\n",
                ),
                runtime_text(
                    "value = os.getenv(key='CUSTOM_ACCESS_TO",
                    "KEN', default='fixture-default')\n",
                ),
            )
            for index, source in enumerate(bad_sources):
                with self.subTest(index=index):
                    candidate = root / f"bad-{index}.py"
                    candidate.write_text(source, encoding="utf-8")
                    errors = check_public_tree.check(root)
                    self.assertTrue(
                        any("secret-like" in error for error in errors), errors
                    )
                    candidate.unlink()

    def test_all_reconstructed_static_text_is_scanned_once(self) -> None:
        ticket_state = runtime_object(
            ("schema_version", 1),
            ("updated_at", "2026-08-12T12:00:00Z"),
            ("tickets", {}),
        )
        bad_sources = (
            runtime_text(
                'payload = "x://fixture-user:', '" + "fixture-password@host"\n'
            ),
            runtime_text('payload = "ghp_', '" "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n'),
            runtime_text('payload = "-----BEGIN ', '" + "PRIVATE KEY-----"\n'),
            'payload = b"pass\\xe2\\x80\\x8bword=fixture-value"\n',
            "payload = " + repr(json.dumps(ticket_state)) + "\n",
        )
        for index, source in enumerate(bad_sources):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "source.py").write_text(source, encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(
                    any(
                        "secret-like" in error or "private/runtime" in error
                        for error in errors
                    ),
                    errors,
                )

        safe_source = runtime_text(
            "suffix = load_value()\nfirst = 'x://fixture-user",
            ":' + suffix\nsecond = f'x://fixture-user:",
            "{suffix}@host'\nthird = 'ghp_",
            "' + suffix\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "safe.py").write_text(safe_source, encoding="utf-8")
            self.assertEqual(check_public_tree.check(root), [])

    def test_static_environment_setdefault_and_dict_pairs_are_scanned(self) -> None:
        bad_sources = (
            runtime_text(
                "os.environ.setdefault('OPENAI_",
                "API_KEY', 'fixture-default')\n",
            ),
            runtime_text(
                "os.environ.setdefault(key='MLFLOW_TRACKING_PASS",
                "WORD', default='fixture-' + 'default')\n",
            ),
            runtime_text(
                "payload = dict([('OPENAI_",
                "API_KEY', 'fixture-default')])\n",
            ),
            runtime_text(
                "payload = dict((('MLFLOW_TRACKING_PASS",
                "WORD', 'fixture-' + 'default'),))\n",
            ),
        )
        for index, source in enumerate(bad_sources):
            with self.subTest(blocked=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "case.py").write_text(source, encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

        safe_source = runtime_text(
            "os.environ.setdefault('OPENAI_",
            "API_KEY', load_default())\n",
            "os.environ.setdefault(load_key(), 'fixture-default')\n",
            "os.environ.setdefault('N_TOKENS', '1024')\n",
            "dynamic_value = dict([('OPENAI_",
            "API_KEY', load_default())])\n",
            "dynamic_key = dict([(load_key(), 'fixture-default')])\n",
            "metrics = dict([('N_TOKENS', '1024')])\n",
            "ordinary = dict([('model_name', 'vit')])\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "safe.py").write_text(safe_source, encoding="utf-8")
            self.assertEqual(check_public_tree.check(root), [])

    def test_decoded_json_string_leaves_are_scanned(self) -> None:
        bad_documents = (
            runtime_text(
                '{"note":"x\\u003a\\u002f\\u002ffixture-user\\u003a',
                'fixture-password\\u0040host"}',
            ),
            runtime_text(
                '{"note":"ghp\\u005f', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}'
            ),
            runtime_text('{"pass\\u200b', 'word":"opaque"}'),
            runtime_text('{"SERVICE_ACCOUNT_', 'KEY":"opaque"}'),
        )
        for index, document in enumerate(bad_documents):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config.json").write_text(document, encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

    def test_private_jwk_structure_is_rejected_but_public_jwk_passes(self) -> None:
        private_jwks = (
            runtime_object(("kty", "EC"), ("d", "private-fixture")),
            runtime_object(("kty", "OKP"), ("d", "private-fixture")),
            runtime_object(("kty", "oct"), ("k", "symmetric-fixture")),
            runtime_object(("k\u200bty", "EC"), ("\uff44", "private-fixture")),
            *(
                runtime_object(("kty", "RSA"), (parameter, "private-fixture"))
                for parameter in ("d", "p", "q", "dp", "dq", "qi")
            ),
            runtime_object(
                ("kty", "RSA"),
                ("oth", [runtime_object(("r", "r"), ("d", "d"), ("t", "t"))]),
            ),
            runtime_object(
                ("keys", [runtime_object(("kty", "OKP"), ("d", "nested-private-fixture"))])
            ),
        )
        for index, jwk in enumerate(private_jwks):
            with self.subTest(private_jwk=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "jwk.json").write_text(json.dumps(jwk), encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

        python_private_jwks = (
            runtime_text(
                "payload = {'k", "ty': 'oct', 'k': 'private-fixture'}\n"
            ),
            runtime_text(
                "payload = dict(k", "ty='EC', d='private-fixture')\n"
            ),
            runtime_text(
                "payload = Jwk(k", "ty='RSA', p='private-fixture')\n"
            ),
        )
        for index, source in enumerate(python_private_jwks):
            with self.subTest(python_private_jwk=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "source.py").write_text(source, encoding="utf-8")
                self.assertTrue(
                    any(
                        "secret-like" in error
                        for error in check_public_tree.check(root)
                    )
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "duplicate.json").write_text(
                runtime_text(
                    '{"kty":"oct","k":"private-fixture",', '"k":""}'
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                any("structurally scanned" in error for error in check_public_tree.check(root))
            )

        public_jwks = {
            "keys": [
                {"kty": "EC", "crv": "P-256", "x": "x", "y": "y"},
                {"kty": "OKP", "crv": "Ed25519", "x": "x"},
                {"kty": "RSA", "n": "modulus", "e": "AQAB"},
                {"kty": "ec", "d": "descriptive-dimension"},
                {"kty": "RSA", "p": None},
                {"kty": "oct", "k": ""},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "public-jwks.json").write_text(
                json.dumps(public_jwks), encoding="utf-8"
            )
            (root / "public-and-dynamic.py").write_text(
                runtime_text(
                    "public = dict(k",
                    "ty='EC', x='public-x', y='public-y')\n",
                    "dynamic = dict(k",
                    "ty='EC', d=load_private_value())\n",
                ),
                encoding="utf-8",
            )
            self.assertEqual(check_public_tree.check(root), [])

    def test_static_text_collector_growth_is_bounded(self) -> None:
        elapsed: list[float] = []
        for size in (2_000, 8_000):
            # Build the left-leaning ``dynamic()+'x'+'x'...`` chain directly:
            # Python 3.11's parser exhausts its recursion limit on this depth,
            # and the collector under test is what needs the deep tree.
            tree = _dynamic_concat_chain(size)
            started = time.perf_counter()
            tuple(check_public_tree._iter_maximal_static_text_values(tree))
            elapsed.append(time.perf_counter() - started)
        self.assertLess(elapsed[-1], 5.0)
        self.assertLess(elapsed[-1], max(elapsed[0], 0.01) * 8 + 0.25)

        formatted = ast.parse("payload = f'{1:999999999}'\n").body[0]
        assert isinstance(formatted, ast.Assign)
        started = time.perf_counter()
        self.assertIsNone(check_public_tree._static_text_value(formatted.value))
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_json_credential_fields_are_structurally_scanned(self) -> None:
        credential_documents = (
            runtime_object((runtime_text("client", "Secret"), "opaque")),
            runtime_object((runtime_text("auth", "Tokens"), ["opaque"])),
            runtime_object((runtime_text("client", "Credentials"), "opaque")),
            runtime_text('{"client\\u0053', 'ecret": "opaque"}'),
        )
        for index, document in enumerate(credential_documents):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rendered = document if isinstance(document, str) else json.dumps(document)
                (root / "config.json").write_text(rendered, encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(any("secret-like" in error for error in errors), errors)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "TOKEN_IDS": [1, 2, 3],
                        "TOKEN_EMBEDDING_DIMENSION": 4096,
                        "TOKEN_BUDGET": 8192,
                        "ADDITIONAL_SPECIAL_TOKENS": ["<image>", "<audio>"],
                        "SPECIAL_TOKENS_MAP": "tokenizer.json",
                        "IMAGE_TOKEN_INDEX": 32000,
                        "NUM_IMAGE_TOKENS": 576,
                        "CLS_TOKEN": "[CLS]",
                        "SEP_TOKEN": "[SEP]",
                        "UNK_TOKEN": "[UNK]",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(check_public_tree.check(root), [])

    def test_runtime_artifact_shapes_are_rejected_but_public_examples_pass(self) -> None:
        example = json.loads(
            (REPO_ROOT / "examples" / "profile.example.json").read_text(
                encoding="utf-8"
            )
        )
        schema_text = (
            REPO_ROOT
            / "skills"
            / "remote-gpu-dev"
            / "references"
            / "profile.schema.json"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "renamed-example.txt").write_text(
                json.dumps(example, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            (root / "profile-schema.json").write_text(schema_text, encoding="utf-8")
            (root / "README.md").write_text(
                "# Ticket documentation\n\nA board may have an active queue.\n",
                encoding="utf-8",
            )
            self.assertEqual(check_public_tree.check(root), [])

        ticket = runtime_object(
            ("id", "GPU-20260812-120000-abcd-vision"),
            ("status", "reserved"),
            ("project", "vision"),
            ("owner", "researcher"),
            ("purpose", "train a classifier"),
            ("requested_gpus", 1),
            ("requested_gpu_ids", [0]),
            ("assigned_gpus", [0]),
            ("expected_duration_minutes", 60),
            ("created_at", "2026-08-12T12:00:00Z"),
            ("updated_at", "2026-08-12T12:00:00Z"),
        )
        modified_profile = json.loads(json.dumps(example))
        modified_profile["ssh"]["host"] = "gpu.internal"
        ticket_config = {
            "schema_version": 1,
            "server": "researcher@gpu.internal:22",
            "profile": "private-gpu",
            "coordination_uid": "sha256:" + "a" * 64,
            "gpu_ids": [0],
            "gpu_devices": [{"id": 0, "uuid": "GPU-fixture"}],
            "reservation_ttl_minutes": 30,
            "heartbeat_grace_minutes": 10,
            "recent_terminal_limit": 12,
            "tensorboard_port_start": 16006,
            "tensorboard_port_end": 16105,
        }
        event = {
            "at": "2026-08-12T12:00:00Z",
            "action": "reserve",
            "ticket_id": ticket["id"],
            "status": "reserved",
            "assigned_gpus": [0],
            "detail": "",
        }
        overview = {
            "server": "researcher@gpu.internal:22",
            "profile": "private-gpu",
            "coordination_uid": "sha256:" + "a" * 64,
            "gpu_ids": [0],
            "gpu_devices": [{"id": 0, "uuid": "GPU-fixture"}],
            "occupied_gpus": [0],
            "tickets": [ticket],
            "board": "/private/BOARD.md",
            "snapshot_note": "read-only; run reconcile to apply pending transitions",
        }
        board = runtime_text(
            "# Remote GPU Ticket Board\n\n",
            "Generated at `2026-08-12T12:00:00Z` for `researcher@gpu.internal:22`. ",
            "Use `gpu_ticket.py`; do not edit this board.\n\n",
            "## Active allocations\n\n_None._\n\n",
            "## Queue\n\n_None._\n\n",
            "## Recent terminal tickets\n\n_None._\n",
        )
        ticket_markdown = (
            "---\n"
            f'id: "{ticket["id"]}"\n'
            'status: "reserved"\nproject: "vision"\nowner: "researcher"\n'
            "assigned_gpus: [0]\n"
            'created_at: "2026-08-12T12:00:00Z"\n'
            'updated_at: "2026-08-12T12:00:00Z"\n'
            "---\n\n"
            f'# {ticket["id"]}: vision\n\n'
            "| Field | Value |\n|---|---|\n"
            "| Remote workdir | /private/run |\n"
            "| TensorBoard status | `live` |\n\n"
            "This file is generated from `state.json`. Use `gpu_ticket.py` for updates.\n"
        )
        artifacts = (
            ("profile.txt", json.dumps(modified_profile), "profile"),
            ("config.txt", json.dumps(ticket_config), "ticket config"),
            (
                "ledger.txt",
                json.dumps(
                    {
                        "schema_version": 1,
                        "updated_at": "2026-08-12T12:00:00Z",
                        "tickets": {ticket["id"]: ticket},
                    }
                ),
                "ticket state",
            ),
            ("ticket.json", json.dumps(ticket), "ticket record"),
            ("overview.json", json.dumps(overview), "ticket overview"),
            ("history.txt", json.dumps(event) + "\n" + json.dumps(event), "event log"),
            ("board-copy.md", board, "ticket board"),
            ("ticket-copy.md", ticket_markdown, "ticket Markdown"),
        )
        for name, content, kind in artifacts:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_text(content, encoding="utf-8")
                errors = check_public_tree.check(root)
                self.assertTrue(
                    any("private/runtime" in error and kind in error for error in errors),
                    errors,
                )

        for content, kind in (
            (artifacts[2][1], "ticket state"),
            (artifacts[5][1], "event log"),
            (board, "ticket board"),
            (ticket_markdown, "ticket Markdown"),
        ):
            with self.subTest(static_python=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "embedded.py").write_text(
                    "payload = " + repr(content) + "\n", encoding="utf-8"
                )
                errors = check_public_tree.check(root)
                self.assertTrue(
                    any("private/runtime" in error and kind in error for error in errors),
                    errors,
                )

    def test_oversize_files_stop_before_content_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = runtime_text("-----BEGIN ", "PRIVATE KEY-----\n")
            (root / "oversize.md").write_text(
                marker + "x" * check_public_tree.MAX_PUBLIC_SOURCE_BYTES,
                encoding="utf-8",
            )
            errors = check_public_tree.check(root)
            self.assertEqual(len(errors), 1, errors)
            self.assertIn("size limit", errors[0])

    def test_private_key_regex_has_linear_growth_at_one_million_characters(self) -> None:
        chunk = "-----BEGIN A "
        elapsed: list[float] = []
        for size in (100_000, 1_000_000):
            value = (chunk * (size // len(chunk) + 1))[:size]
            started = time.perf_counter()
            self.assertIsNone(check_public_tree.PRIVATE_KEY_RE.search(value))
            elapsed.append(time.perf_counter() - started)
        self.assertLess(elapsed[-1], 5.0)
        self.assertLess(elapsed[-1], max(elapsed[0], 0.005) * 25 + 0.25)

    def test_unknown_non_utf8_binary_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.bin").write_bytes(b"\x00\xff\x00\xfe")
            errors = check_public_tree.check(root)
            self.assertTrue(any("binary" in error for error in errors), errors)

    def test_utf8_decodable_binary_controls_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.data").write_bytes(b"prefix\x00payload")
            errors = check_public_tree.check(root)
            self.assertTrue(any("binary" in error for error in errors), errors)

    def test_unknown_extension_printable_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "public.notice").write_text(
                "This is a public release note.\n", encoding="utf-8"
            )
            errors = check_public_tree.check(root)
            self.assertTrue(any("unrecognized" in error for error in errors), errors)

    def test_known_plain_utf8_text_remains_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text(
                "This is a public release note.\n", encoding="utf-8"
            )
            self.assertEqual(check_public_tree.check(root), [])


if __name__ == "__main__":
    unittest.main()
