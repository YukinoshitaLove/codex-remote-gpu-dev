#!/usr/bin/env python3
"""Repository wrapper for the skill's managed global installer."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "remote-gpu-dev"
    / "scripts"
    / "manage_global_install.py"
)


if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(SCRIPT), *sys.argv[1:]])
