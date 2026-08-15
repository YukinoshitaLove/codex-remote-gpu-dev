#!/usr/bin/env python3
"""Read-only nvidia-smi snapshot for the selected remote GPU profile."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from typing import Any

from profile import ProfileError, load_profile, utc_now
from ssh_remote import SSHError, ssh_argv


class GPUStatusError(RuntimeError):
    pass


def _normalize_mig_mode(value: str) -> str | None:
    compact = value.strip().lower().strip("[]").replace(" ", "").replace("_", "")
    if compact == "disabled":
        return "disabled"
    if compact in {"na", "n/a", "notsupported", "unsupported"}:
        return "unsupported"
    if compact == "enabled":
        return "enabled"
    return None


REMOTE_COMMAND = r'''set -eu
printf '%s\n' '__REMOTE_GPU_DEV_GPUS__'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,pstate,mig.mode.current --format=csv,noheader,nounits
printf '%s\n' '__REMOTE_GPU_DEV_PROCESSES__'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
'''


def _rows(text: str) -> list[list[str]]:
    return [[item.strip() for item in row] for row in csv.reader(io.StringIO(text)) if row]


def snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    try:
        argv = ssh_argv(profile, batch=True)
        argv.extend([f"{profile['ssh']['user']}@{profile['ssh']['host']}", REMOTE_COMMAND])
        completed = subprocess.run(
            argv,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, SSHError) as exc:
        raise GPUStatusError(f"nvidia-smi probe could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join(completed.stderr.split())[:400]
        raise GPUStatusError(f"nvidia-smi probe failed: {detail}")
    try:
        gpu_text, rest = completed.stdout.split("__REMOTE_GPU_DEV_GPUS__\n", 1)[1].split(
            "__REMOTE_GPU_DEV_PROCESSES__\n", 1
        )
    except (IndexError, ValueError) as exc:
        raise GPUStatusError("remote nvidia-smi output was malformed") from exc
    gpus: list[dict[str, Any]] = []
    for row in _rows(gpu_text):
        if len(row) != 10:
            raise GPUStatusError("unexpected nvidia-smi GPU column count")
        try:
            gpus.append(
                {
                    "index": int(row[0]),
                    "uuid": row[1],
                    "name": row[2],
                    "memory_total_mib": int(row[3]),
                    "memory_used_mib": int(row[4]),
                    "gpu_utilization_percent": int(row[5]),
                    "memory_utilization_percent": int(row[6]),
                    "temperature_c": int(row[7]),
                    "pstate": row[8],
                    "mig_mode_current": row[9].lower(),
                }
            )
        except ValueError as exc:
            raise GPUStatusError("nvidia-smi GPU row was not numeric") from exc
    processes: list[dict[str, Any]] = []
    for row in _rows(rest):
        if len(row) != 4:
            raise GPUStatusError("unexpected nvidia-smi compute-process column count")
        try:
            processes.append(
                {
                    "gpu_uuid": row[0],
                    "pid": int(row[1]),
                    "process_name": row[2],
                    "used_memory_mib": int(row[3]),
                }
            )
        except ValueError as exc:
            raise GPUStatusError("nvidia-smi compute-process row was not numeric") from exc
    expected_mapping = {
        item["index"]: item["uuid"] for item in profile["gpu"]["devices"]
    }
    actual_mapping = {item["index"]: item["uuid"] for item in gpus}
    mapping_mismatches = {
        str(index): {"expected": uuid, "actual": actual_mapping.get(index)}
        for index, uuid in expected_mapping.items()
        if actual_mapping.get(index) != uuid
    }
    inventory_match = not mapping_mismatches
    normalized_mig_modes = {
        str(item["index"]): _normalize_mig_mode(item["mig_mode_current"])
        for item in gpus
    }
    expected_mig_policy = profile["gpu"]["mig_mode"]
    mig_policy_match = bool(gpus) and all(
        mode == expected_mig_policy for mode in normalized_mig_modes.values()
    )
    unsafe_mig_modes = {
        index: mode
        for index, mode in normalized_mig_modes.items()
        if mode != expected_mig_policy
    }
    expected_uuids = set(expected_mapping.values())
    return {
        "schema_version": 1,
        "sampled_at": utc_now(),
        "profile": profile["slug"],
        "server_uid": profile["trust"]["server_uid"],
        "ssh_trust_uid": profile["trust"]["server_uid"],
        "coordination_uid": profile["trust"]["coordination_uid"],
        "inventory_match": inventory_match,
        "mapping_mismatches": mapping_mismatches,
        "mig_policy": expected_mig_policy,
        "mig_policy_match": mig_policy_match,
        "mig_disabled": expected_mig_policy == "disabled" and mig_policy_match,
        "unsafe_mig_modes": unsafe_mig_modes,
        "safe_to_allocate": inventory_match and mig_policy_match,
        "managed_gpu_devices": profile["gpu"]["devices"],
        "managed_gpu_uuids": sorted(expected_uuids),
        "gpus": gpus,
        "compute_processes": processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = snapshot(load_profile())
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for gpu in payload["gpus"]:
                managed = "managed" if gpu["uuid"] in payload["managed_gpu_uuids"] else "foreign"
                print(
                    f"GPU {gpu['index']} {managed} util={gpu['gpu_utilization_percent']}% "
                    f"memory={gpu['memory_used_mib']}/{gpu['memory_total_mib']}MiB "
                    f"temp={gpu['temperature_c']}C {gpu['uuid']}"
                )
            print(f"compute_processes={len(payload['compute_processes'])}")
            print(
                f"inventory_match={str(payload['inventory_match']).lower()} "
                f"mig_policy_match={str(payload['mig_policy_match']).lower()} "
                f"safe_to_allocate={str(payload['safe_to_allocate']).lower()}"
            )
        return 0 if payload["safe_to_allocate"] else 1
    except (ProfileError, GPUStatusError) as exc:
        print(f"remote-gpu-status: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
