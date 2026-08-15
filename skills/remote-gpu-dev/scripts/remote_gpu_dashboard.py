#!/usr/bin/env python3
"""User-facing launcher for the active remote GPU profile dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DASHBOARD = Path(__file__).resolve().with_name("dashboard.py")


def dashboard_arguments(arguments: list[str]) -> list[str] | None:
    """Map the friendly launcher surface to the dashboard control CLI."""
    if not arguments or arguments == ["open"]:
        return ["ensure", "--open"]
    if arguments[0] in {"-h", "--help"}:
        return None
    if arguments[0] in {"status", "stop"} and len(arguments) == 1:
        return arguments
    if arguments[0] == "ensure":
        return arguments
    raise ValueError(
        "usage: remote-gpu-dashboard [open|status|stop|ensure [--open]]"
    )


def print_help() -> None:
    print(
        "usage: remote-gpu-dashboard [open|status|stop|ensure [--open]]\n\n"
        "Without arguments, start or reuse the verified singleton and open it.\n"
        "  open            start/reuse and open the dashboard\n"
        "  status          show the current singleton status and URL\n"
        "  stop            stop only the verified singleton process\n"
        "  ensure [--open] start/reuse without opening unless --open is supplied"
    )


def main() -> int:
    try:
        mapped = dashboard_arguments(sys.argv[1:])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if mapped is None:
        print_help()
        return 0
    os.execv(
        sys.executable,
        [sys.executable, str(DASHBOARD), *mapped],
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
