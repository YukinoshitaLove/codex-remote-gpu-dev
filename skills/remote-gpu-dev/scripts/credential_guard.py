#!/usr/bin/env python3
"""Shared, bounded credential detection for profiles, tickets, and errors."""

from __future__ import annotations

import json
import re
import unicodedata


ASCII_TOKEN_LEFT = r"(?<![A-Za-z0-9_])"
ASCII_TOKEN_RIGHT = r"(?![A-Za-z0-9_])"
IDENTIFIER_FIELD_BODY = r"[A-Za-z][A-Za-z0-9]*(?:[_-]+[A-Za-z0-9]+){0,15}"
SPACED_FIELD_BODY = (
    r"(?:access[ \t]+token|api[ \t]+key|auth[ \t]+token"
    r"|client[ \t]+secret|private[ \t]+key"
    r"|service[ \t]+account[ \t]+key"
    r"|aws[ \t]+access[ \t]+key(?:[ \t]+id)?"
    r"|aws[ \t]+secret[ \t]+access[ \t]+key)"
)
FIELD_NAME_BODY = rf"(?:{IDENTIFIER_FIELD_BODY}|{SPACED_FIELD_BODY})"
ASSIGNMENT_CANDIDATE_RE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_])(?P<quote>['\"]?)"
    rf"(?P<field>{FIELD_NAME_BODY})(?P=quote)[ \t]*[:=]"
)
FIELD_SEGMENT_RE = re.compile(r"[A-Za-z0-9]+")
CAMEL_COMPONENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)
PLURAL_COMPONENTS = frozenset(
    {
        "CREDENTIALS",
        "KEYS",
        "PASSPHRASES",
        "PASSWORDS",
        "SECRETS",
        "TOKENS",
    }
)
ALWAYS_CREDENTIAL_COMPONENTS = frozenset(
    {
        "AUTHORIZATION",
        "CREDENTIAL",
        "PASSWD",
        "PASSPHRASE",
        "PASSWORD",
        "SECRET",
    }
)
COMPACT_CREDENTIAL_COMPONENTS = frozenset(
    {
        "ACCESSKEY",
        "ACCESSTOKEN",
        "APIKEY",
        "AUTHTOKEN",
        "CLIENTSECRET",
        "PRIVATEKEY",
        "SECRETACCESSKEY",
    }
)
NON_CREDENTIAL_TOKEN_SUFFIXES = frozenset(
    {
        "BUDGET",
        "BUDGETS",
        "COUNT",
        "COUNTS",
        "DIMENSION",
        "DIMENSIONS",
        "EMBEDDING",
        "EMBEDDINGS",
        "ID",
        "IDS",
        "INDEX",
        "LENGTH",
        "LENGTHS",
        "LEFT",
        "LIMIT",
        "LIMITS",
        "MAP",
        "PARALLELISM",
        "PER",
        "POSITION",
        "POSITIONS",
        "PREFIX",
        "RATE",
        "RATES",
        "RATIO",
        "RATIOS",
        "SIZE",
        "SIZES",
        "SUFFIX",
        "TYPE",
        "TYPES",
        "WEIGHT",
        "WEIGHTS",
        "WINDOW",
        "WINDOWS",
        "RIGHT",
        "SECOND",
        "SECONDS",
    }
)
NON_CREDENTIAL_TERMINAL_TOKEN_PREFIXES = frozenset(
    {
        "BOS",
        "CLS",
        "COMPLETION",
        "EOS",
        "IMAGE",
        "MASK",
        "MAX",
        "MIN",
        "NUM",
        "NUMBER",
        "N",
        "OUTPUT",
        "INPUT",
        "PAD",
        "PROMPT",
        "SEP",
        "SPECIAL",
        "TOTAL",
        "UNK",
    }
)
NON_CREDENTIAL_NEW_TOKEN_PREFIXES = frozenset({"MAX", "MIN"})
CREDENTIAL_TOKEN_PREFIXES = frozenset(
    {"ACCESS", "API", "AUTH", "BEARER", "CLIENT", "ID", "REFRESH", "SESSION"}
)
TOKEN_CONTROL_COMPONENTS = frozenset({"DISABLE", "ENABLE"})
NON_CREDENTIAL_FIELD_COMPONENTS = frozenset({("PASSWORD", "AUTHENTICATION")})
CREDENTIAL_COMPONENT_SEQUENCES = frozenset({("SERVICE", "ACCOUNT", "KEY")})
LINE_CONTINUATION_RE = re.compile(r"\\(?:\r\n|\r|\n)")
MAX_STRUCTURED_SECRET_BYTES = 2 * 1024 * 1024
MAX_STRUCTURED_SECRET_NODES = 100_000
MAX_STRUCTURED_SECRET_DEPTH = 64
MAX_STRUCTURED_SCAN_BYTES = 8 * 1024 * 1024
RSA_PRIVATE_JWK_PARAMETERS = frozenset({"d", "p", "q", "dp", "dq", "qi"})


class StructuredSecretScanError(ValueError):
    """A JSON document could not be safely and completely inspected."""

PEM_HEADER_BODY = (
    r"-{4,5}[ \t]*BEGIN[ \t]+(?:[A-Z0-9]+[ \t]+){0,4}"
    r"PRIVATE[ \t]+KEY(?:[ \t]+BLOCK)?[ \t]*-{4,5}"
)
PRIVATE_KEY_RE = re.compile(rf"(?i){PEM_HEADER_BODY}")
SECRET_VALUE_RE = re.compile(
    rf"(?i)(?:{ASCII_TOKEN_LEFT}(?:bearer|basic)[ \t]+"
    r"(?!(?:authentication|authorization|credentials?|header|token|value)\b)"
    rf"[A-Za-z0-9._~+/=-]{{8,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}hf_[A-Za-z0-9]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}github_pat_[A-Za-z0-9_]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}gh[pousr]_[A-Za-z0-9]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}sk-(?:(?:proj|svcacct|admin)-"
    rf"[A-Za-z0-9_-]{{20,}}|[A-Za-z0-9]{{20,}}){ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}sk-ant-api03-[A-Za-z0-9_-]{{16,}}"
    rf"{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}glpat-[A-Za-z0-9_-]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}pypi-[A-Za-z0-9_-]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}npm_[A-Za-z0-9]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}sk_live_[A-Za-z0-9]{{16,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}AIza[A-Za-z0-9_-]{{35}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}ya29\.[A-Za-z0-9._-]{{20,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}(?:xox[baprs]|xapp|xoxe)-"
    rf"[A-Za-z0-9-]{{10,}}{ASCII_TOKEN_RIGHT}"
    rf"|{ASCII_TOKEN_LEFT}(?:AKIA|ASIA)[0-9A-Z]{{16}}{ASCII_TOKEN_RIGHT}"
    r"|(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,31}://"
    r"[^/?#\s:@]*:[^/?#\s@]+@[^/?#\s]+"
    r"|(?<![A-Za-z0-9/])//[^/?#\s:@]*:[^/?#\s@]+@[^/?#\s]+"
    rf"|{PEM_HEADER_BODY}"
    r"|(?<![A-Za-z0-9_])PuTTY-User-Key-File-(?:1|2|3):[^\r\n]{0,512}"
    rf"|{ASCII_TOKEN_LEFT}AGE-SECRET-KEY-1[0-9A-Z]{{20,}}{ASCII_TOKEN_RIGHT})"
)


def _normalize_unicode_for_secret_scan(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )


def field_components(field: str) -> tuple[str, ...]:
    """Return normalized semantic components from a configuration field."""

    components: list[str] = []
    normalized = _normalize_unicode_for_secret_scan(field)
    for segment in FIELD_SEGMENT_RE.findall(normalized):
        parts = CAMEL_COMPONENT_RE.findall(segment)
        if not parts:
            parts = [segment]
        for part in parts:
            upper = part.upper()
            components.append(upper[:-1] if upper in PLURAL_COMPONENTS else upper)
    return tuple(components)


def is_credential_field_name(field: str) -> bool:
    """Classify credential slots without treating token metrics as secrets."""

    components = field_components(field)
    if not components:
        return False
    if components in NON_CREDENTIAL_FIELD_COMPONENTS:
        return False
    if any(component in ALWAYS_CREDENTIAL_COMPONENTS for component in components):
        return True
    if any(component in COMPACT_CREDENTIAL_COMPONENTS for component in components):
        return True
    if components[-1] == "SERVICEACCOUNTKEY":
        return True
    if any(
        pair in {("ACCESS", "KEY"), ("API", "KEY"), ("PRIVATE", "KEY")}
        for pair in zip(components, components[1:])
    ):
        return True
    if any(
        tuple(components[-len(sequence) :]) == sequence
        for sequence in CREDENTIAL_COMPONENT_SEQUENCES
    ):
        return True

    component_set = set(components)
    for index, component in enumerate(components):
        if component != "TOKEN":
            continue
        suffix = set(components[index + 1 :])
        predecessor = components[index - 1] if index else None
        if predecessor in CREDENTIAL_TOKEN_PREFIXES:
            return True
        if suffix.intersection(NON_CREDENTIAL_TOKEN_SUFFIXES):
            continue
        if predecessor in NON_CREDENTIAL_TERMINAL_TOKEN_PREFIXES:
            continue
        if (
            predecessor == "NEW"
            and index >= 2
            and components[index - 2] in NON_CREDENTIAL_NEW_TOKEN_PREFIXES
        ):
            continue
        if (
            predecessor == "IMPLICIT"
            and component_set.intersection(TOKEN_CONTROL_COMPONENTS)
        ):
            continue
        return True
    return False


def normalize_for_secret_scan(value: str) -> str:
    """Return a scan-only view without changing the caller's stored value."""

    normalized = _normalize_unicode_for_secret_scan(value)
    normalized = LINE_CONTINUATION_RE.sub("", normalized)
    return " ".join(normalized.split())


def _contains_normalized_credential_assignment(value: str) -> bool:
    return any(
        is_credential_field_name(match.group("field"))
        for match in ASSIGNMENT_CANDIDATE_RE.finditer(value)
    )


def contains_credential_assignment(value: str) -> bool:
    """Return whether text assigns a value to a credential-bearing field."""

    return _contains_normalized_credential_assignment(
        normalize_for_secret_scan(value)
    )


def _reject_nonstandard_json_constant(value: str) -> object:
    raise StructuredSecretScanError(f"non-standard JSON constant: {value}")


def _canonical_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Construct one JSON object while rejecting raw and Unicode duplicate keys."""

    result: dict[str, object] = {}
    seen: set[str] = set()
    for key, child in pairs:
        canonical = _normalize_unicode_for_secret_scan(key)
        if canonical in seen:
            raise StructuredSecretScanError("duplicate JSON object key")
        seen.add(canonical)
        result[key] = child
    return result


def _validate_decoded_json_budget(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    decoded_bytes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_STRUCTURED_SECRET_NODES:
            raise StructuredSecretScanError("decoded JSON node budget exceeded")
        if depth > MAX_STRUCTURED_SECRET_DEPTH:
            raise StructuredSecretScanError("decoded JSON depth budget exceeded")
        if isinstance(current, str):
            decoded_bytes += len(current.encode("utf-8"))
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, dict):
            for key, child in current.items():
                decoded_bytes += len(key.encode("utf-8"))
                pending.append((child, depth + 1))
        if decoded_bytes > MAX_STRUCTURED_SCAN_BYTES:
            raise StructuredSecretScanError("decoded JSON text budget exceeded")


def decode_json_for_secret_scan(value: str) -> object:
    """Decode one bounded JSON document with duplicate and resource checks."""

    if len(value) > MAX_STRUCTURED_SECRET_BYTES:
        raise StructuredSecretScanError("JSON source character budget exceeded")
    if len(value.encode("utf-8")) > MAX_STRUCTURED_SECRET_BYTES:
        raise StructuredSecretScanError("JSON source byte budget exceeded")
    try:
        document = json.loads(
            value,
            object_pairs_hook=_canonical_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except StructuredSecretScanError:
        raise
    except (RecursionError, MemoryError) as exc:
        raise StructuredSecretScanError("JSON decoder resource limit reached") from exc
    _validate_decoded_json_budget(document)
    return document


def _contains_text_secret(value: str) -> bool:
    normalized = normalize_for_secret_scan(value)
    return _contains_normalized_credential_assignment(normalized) or bool(
        SECRET_VALUE_RE.search(normalized)
    )


def _contains_structured_secret(
    value: object, *, credential_fields: bool, string_leaves: bool
) -> bool:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    visited = 0
    scanned_bytes = 0
    while pending:
        current, depth = pending.pop()
        if depth > MAX_STRUCTURED_SECRET_DEPTH:
            return True
        if isinstance(current, str):
            if string_leaves:
                scanned_bytes += len(current.encode("utf-8"))
                if scanned_bytes > MAX_STRUCTURED_SCAN_BYTES:
                    return True
                if _contains_text_secret(current):
                    return True
            continue
        if not isinstance(current, (dict, list, tuple)):
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        visited += 1
        if visited > MAX_STRUCTURED_SECRET_NODES:
            return True
        if isinstance(current, (list, tuple)):
            pending.extend((child, depth + 1) for child in current)
            continue
        canonical_items: list[tuple[str, object]] = []
        key_types: list[str] = []
        for key, child in current.items():
            if isinstance(key, str):
                normalized_key = _normalize_unicode_for_secret_scan(key)
                canonical_items.append((normalized_key, child))
                if credential_fields and (
                    is_credential_field_name(normalized_key)
                    or _contains_text_secret(normalized_key)
                ):
                    return True
                if normalized_key == "kty" and isinstance(child, str):
                    key_types.append(_normalize_unicode_for_secret_scan(child))
            pending.append((child, depth + 1))
        string_parameters = {
            key
            for key, child in canonical_items
            if isinstance(child, str) and bool(child)
        }
        has_oth = any(
            key == "oth" and isinstance(child, list) and bool(child)
            for key, child in canonical_items
        )
        if any(kty in {"EC", "OKP"} for kty in key_types) and "d" in string_parameters:
            return True
        if "RSA" in key_types and (
            string_parameters.intersection(RSA_PRIVATE_JWK_PARAMETERS) or has_oth
        ):
            return True
        if "oct" in key_types and "k" in string_parameters:
            return True
    return False


def contains_private_jwk(value: object) -> bool:
    """Return whether a bounded JSON-like value contains private JWK material."""

    return _contains_structured_secret(
        value, credential_fields=False, string_leaves=False
    )


def contains_structured_secret(value: object) -> bool:
    """Scan decoded JSON-like keys, string leaves, and private JWK structure."""

    return _contains_structured_secret(
        value, credential_fields=True, string_leaves=True
    )


def _contains_structured_secret_text(value: str) -> bool:
    if len(value) > MAX_STRUCTURED_SECRET_BYTES:
        return True
    if len(value.encode("utf-8")) > MAX_STRUCTURED_SECRET_BYTES:
        return True
    stripped = _normalize_unicode_for_secret_scan(value).lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        document = decode_json_for_secret_scan(stripped)
    except json.JSONDecodeError:
        return False
    except StructuredSecretScanError:
        return True
    return contains_structured_secret(document)


def contains_secret(value: str) -> bool:
    """Return whether text contains a common credential or private-key marker."""

    return _contains_text_secret(value) or _contains_structured_secret_text(value)
