# Remote execution contract

## Compatibility and evolution rule

Keep the normal path short: deploy, reserve/start, run, validate, and release.
Training, testing, inference, CUDA/NCCL/DDP, DataLoader workers, compilation,
checkpoint save/load, logs, and result files must work without project-specific
Skill workarounds. Managed paths, tickets, and infrastructure isolation prevent
accidental writes and GPU conflicts for trusted code; they are not hostile-code
containment, and profile roots are not hard-coded universal paths.

When a concrete workload reproduces a compatibility failure, identify the
narrowest responsible Skill rule or helper, update it locally, and add a focused
regression test. Preserve safeguards unrelated to the failure. Do not add runner
restrictions or extra launch steps without evidence that they are required.
If a restriction blocks a core operation, the AI may directly relax it in the
minimum necessary scope and synchronize the helper, tests, documentation, and
release metadata. Never treat an older restriction as immutable when it prevents
working core functionality.

## Before launch

1. Local worktree and index are clean.
2. Remote bare and execution clone resolve to the same full commit.
3. Inputs and any source checkpoint are hashed outside Git as appropriate.
4. The run directory is new, absolute, and inside the managed scratch root.
5. The project Conda prefix is outside the Git clone.
6. A valid ticket owns all requested physical GPUs.
7. Live GPU UUID/index mapping matches the profile and no foreign process is
   present.

## Launch environment

Use structured `remote-gpu-dev run`; the workdir is inferred from the ticket,
and an explicit `--workdir` is only an exact-match assertion. The script may
remain in the clean execution clone while relative outputs land in the separate
run workdir. Use exactly one of `--script` or a dotted ASCII `--module`, and put
unchanged Python arguments after `--`. Add `--session` only for the exact ticket
session when launching a detached job. The structured supervisor writes its
log, immutable launch identity, and atomic final status below
`remote.temp_root/runtime/runs/TICKET/jobs/SESSION`. Query and stop it only with
`run TICKET --status` or `run TICKET --stop`; they infer the ticket session,
while an explicit `--session` must match it.
Stop requires exact ticket/session/workdir, boot ID, PID, process start ticks,
and process-group leadership before sending `SIGTERM`.
Export only validated variables:

```text
CUDA_VISIBLE_DEVICES=<assigned physical indices>
<profile GPU environment, e.g. NCCL_IB_DISABLE=1 when configured>
HF_ENDPOINT=<profile endpoint when applicable>
PIP_INDEX_URL=<profile primary when configured>
PIP_EXTRA_INDEX_URL=<space-separated profile extras when configured>
```

Do not use an interactive shell, arbitrary SSH command, or shell activation.
The workload runner intentionally uses compatibility mode without Landlock. It
places known cache/temp/config/state and managed job-state paths below
`remote.temp_root`, while leaving CUDA, NCCL, DDP, DataLoader, compilation and
normal artifact saving unobstructed. Persistent assets must stay in the roots
by operator and code-review contract. Workload code must be trusted.
Do not put credentials in argv, logs, summaries, or tickets.

Infrastructure helpers retain Landlock. It does not cover SSH authentication,
account-shell startup, network access, GPU DMA, or a hostile kernel. Trust the
server and account startup configuration.

For DDP, record global rank, local rank, logical CUDA device, physical IDs/UUIDs,
world size, backend, process identity, and an actual collective witness. Profile
variables must reach every rank. A ticket duration is metadata; use a real
timeout when the user requests a hard runtime cap.

## Monitor

Heartbeat from the controlling workflow, inspect log growth and exact process
identity, and compare GPU processes with the ticket. Avoid broad polling that
creates new CUDA contexts. A dashboard is optional and never substitutes for
the ticket or result evidence.

SSH control connections can disconnect transiently while the workload remains
healthy. Count only structured control checks and mark the control channel
`unavailable` after five consecutive failures. Failures one through four emit a
warning and retry; any successful structured check resets the consecutive-failure
count. Control-channel failure is not workload-completion evidence and must never
trigger stop or ticket release. Completion and release still require exact
tracked-process identity, assigned-GPU state, and final-status or validated
artifact evidence.

## Validate completion

Do not trust a directory, last log line, or trap-written zero alone. Require the
experiment's completion contract: expected epochs/steps, finite metrics, result
JSON, exit status, artifact hashes, and—when material—checkpoint strict load and
CUDA inference. Explicitly distinguish training, validation/model selection,
and test access. Release only after the tracked process is gone and assigned
GPUs are clear.
