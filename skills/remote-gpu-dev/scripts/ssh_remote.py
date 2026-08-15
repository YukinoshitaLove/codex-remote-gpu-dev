#!/usr/bin/env python3
"""Profile-driven, key-only connectivity and temporary forwarding helper.

This public command intentionally has no interactive shell or arbitrary remote
command surface.  Other Skill helpers build their fixed commands directly from
validated profile fields.
"""

from __future__ import annotations

import argparse
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

from profile import ProfileError, load_profile


class SSHError(RuntimeError):
    pass


def _port_pair(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isascii() or not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("forward must be LOCAL_PORT:REMOTE_PORT")
    local, remote = (int(part) for part in parts)
    if not (1 <= local <= 65535 and 1 <= remote <= 65535):
        raise argparse.ArgumentTypeError("forward ports must be between 1 and 65535")
    return local, remote


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def ssh_argv(profile: dict, *, batch: bool) -> list[str]:
    ssh = profile["ssh"]
    identity = Path(ssh["identity_file"])
    if not identity.is_file() or not os.access(identity, os.R_OK):
        raise SSHError(f"SSH identity is missing or unreadable: {identity}")
    mode = stat.S_IMODE(identity.stat().st_mode)
    if mode & 0o077:
        raise SSHError(
            f"refusing identity permissions {mode:o}; run chmod 600 {identity}"
        )
    known_hosts = Path(ssh["known_hosts_file"])
    if not known_hosts.is_file() or not os.access(known_hosts, os.R_OK):
        raise SSHError(f"profile known_hosts file is missing: {known_hosts}")
    argv = [
        "ssh",
        "-p",
        str(ssh["port"]),
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "ForwardX11Trusted=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "RemoteCommand=none",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "VerifyHostKeyDNS=no",
        "-o",
        f"ConnectTimeout={ssh['connect_timeout_seconds']}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if batch:
        argv.extend(["-o", "BatchMode=yes"])
    if ssh.get("proxy_jump"):
        argv.extend(["-J", ssh["proxy_jump"]])
    return argv


def add_profile_proxy_forward(profile: dict, argv: list[str]) -> None:
    """Append the validated on-demand reverse proxy forward to an SSH argv."""

    local = profile["local"]
    remote = profile["remote"]
    if profile["network"]["proxy_policy"] != "on-demand":
        raise SSHError("this profile does not allow on-demand proxy forwarding")
    if not _listening(local["proxy_host"], local["proxy_port"]):
        raise SSHError(
            f"local proxy is not listening at {local['proxy_host']}:{local['proxy_port']}"
        )
    argv.extend(
        [
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"{remote['proxy_host']}:{remote['proxy_port']}:{local['proxy_host']}:{local['proxy_port']}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="server profile slug; otherwise use the active profile")
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="temporarily reverse-forward the profile's local proxy to the remote loopback port",
    )
    parser.add_argument(
        "--local-forward",
        type=_port_pair,
        metavar="LOCAL:REMOTE",
        help="forward local 127.0.0.1:LOCAL to remote 127.0.0.1:REMOTE",
    )
    parser.add_argument("--no-command", action="store_true", help="keep forwarding only (ssh -N)")
    parser.add_argument("--batch", action="store_true", help="forbid interactive authentication")
    parser.add_argument("--check", action="store_true", help="run a read-only identity check")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        profile = load_profile(args.profile)
        if args.no_command and args.check:
            raise SSHError("--no-command cannot be combined with --check")
        if args.no_command and not (args.proxy or args.local_forward):
            raise SSHError("--no-command requires --proxy or --local-forward")
        argv = ssh_argv(profile, batch=args.batch)
        local = profile["local"]
        remote = profile["remote"]
        if args.proxy:
            add_profile_proxy_forward(profile, argv)
        if args.local_forward:
            local_port, remote_port = args.local_forward
            argv.extend(
                [
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-L",
                    f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
                ]
            )
        target = f"{profile['ssh']['user']}@{profile['ssh']['host']}"
        if args.no_command:
            argv.append("-N")
        elif not args.check:
            raise SSHError(
                "interactive shells and arbitrary remote commands are disabled; "
                "use a structured remote-gpu-dev operation"
            )
        argv.append(target)
        if args.check:
            argv.append(
                'printf "connected user=%s host=%s pwd=%s boot_id=%s\\n" '
                '"$(id -un)" "$(hostname)" "$PWD" '
                '"$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || printf unavailable)"'
            )
        os.execvp(argv[0], argv)
    except (ProfileError, SSHError, OSError) as exc:
        print(f"remote-gpu-ssh: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
