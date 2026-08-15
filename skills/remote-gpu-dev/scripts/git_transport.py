#!/usr/bin/env python3
"""Validated Git receive-pack SSH transport with inherited Landlock policy."""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from managed_run import ManagedRunError, build_landlock_command
from profile import ProfileError, load_profile
from remote_path_guard import RemotePathError, require_managed_remote_path
from ssh_remote import SSHError, ssh_argv


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bare", required=True)
    parser.add_argument("host")
    parser.add_argument("remote_command")
    try:
        args = parser.parse_args()
        profile = load_profile(args.profile)
        bare = require_managed_remote_path(profile, args.bare, "remote bare repository")
        words = shlex.split(args.remote_command)
        if words != ["git-receive-pack", bare]:
            raise ManagedRunError("Git requested an unexpected remote command")
        allowed_hosts = {
            profile["ssh"]["host"],
            f"{profile['ssh']['user']}@{profile['ssh']['host']}",
        }
        if args.host not in allowed_hosts:
            raise ManagedRunError("Git requested an unexpected SSH host")
        command = build_landlock_command(
            profile,
            ["/usr/bin/git-receive-pack", bare],
            workdir=profile["remote"]["git_bare_root"],
        )
        argv = ssh_argv(profile, batch=True)
        argv.extend(
            [f"{profile['ssh']['user']}@{profile['ssh']['host']}", command]
        )
        os.execvp(argv[0], argv)
    except (ProfileError, RemotePathError, SSHError, ManagedRunError, OSError) as exc:
        print(f"git-transport: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
