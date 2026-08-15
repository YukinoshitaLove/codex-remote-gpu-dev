---
name: remote-gpu-dev
description: Configure, coordinate, develop, run, monitor, and validate workloads on one or more SSH-accessible NVIDIA GPU servers. Use when a user wants interactive server onboarding, SSH key guidance, Conda environments, local-authoritative Git deployment, conflict-safe GPU tickets, command-scoped mirrors or proxy forwarding, nvitop telemetry, TensorBoard viewing, remote training, CUDA debugging, or durable experiment records.
---

# Remote GPU Dev

Use server profiles to operate remote NVIDIA GPU machines without embedding a
host, password, private key, or storage layout in the Skill. Keep source editing
and review local; use the server only for execution.

## Keep normal PyTorch work first

Treat this Skill as an evolvable development tool, not a frozen policy. Its
highest priority is a short, working path for normal training, testing,
inference, CUDA/NCCL/DDP, DataLoader workers, compilation, checkpoint save/load,
logging, and result saving. Managed roots, tickets, structured launch, and
infrastructure isolation exist to prevent accidental cross-project writes and
GPU conflicts by trusted AI-authored code; they must not obstruct those normal
operations or be presented as hostile-code containment. Root locations are
profile configuration, not universal paths hard-coded by the Skill.

When a real compatibility failure is reproduced, fix the narrowest responsible
Skill instruction, helper, or guard and add a focused regression test. Do not
make project code carry a workaround for a stale Skill rule. Preserve unrelated
credential, ticket-ownership, exact-process, and destructive-action safeguards,
and keep the common path concise. If the runtime already supports the workload,
clarify the contract instead of changing the runner without evidence.
If any restriction blocks a core operation, the AI is authorized to relax that
restriction directly in the minimum necessary scope and synchronize the helper,
tests, documentation, and release metadata. Do not treat an older restriction
as immutable or ask project code to sacrifice core functionality to preserve it.

## Enforce the remote filesystem boundary

Treat `remote.temp_root` and `remote.durable_root` as the only locations for
remote user assets: projects, code, documents, records, datasets, weights,
checkpoints, logs, event files, environments, downloads, caches, temporary
files, sockets, and generated outputs. Every accepted or derived asset path
must pass the shared lexical guard and remote canonical/symlink check.

Never open an interactive remote shell or accept an arbitrary SSH command. The
public `ssh` operation is limited to strict connection checking and
forwarding-only connections. Run project Python only through structured `run`.
It requires a running ticket, uses the ticket's exact workdir, and accepts a
Conda prefix plus either a script below the two roots or a dotted ASCII module
name:

```bash
remote-gpu-dev --profile SERVER run TICKET \
  --env-prefix /managed/temp/envs/project \
  --script /managed/temp/projects/project/train.py -- --epochs 20

remote-gpu-dev --profile SERVER run TICKET \
  --env-prefix /managed/temp/envs/project \
  --script /managed/temp/projects/project/train.py \
  --session EXACT_TICKET_SESSION -- --epochs 20

remote-gpu-dev --profile SERVER run TICKET \
  --status
remote-gpu-dev --profile SERVER run TICKET \
  --stop
```

The script may live in the clean execution clone while the ticket workdir is a
separate run directory. Relative checkpoint, log, and result paths therefore
land in the run directory. Omit `--workdir` to use the ticket value; when
provided, it is an exact assertion. Arguments after `--` pass through
unchanged. Status and stop infer the recorded ticket session; an explicit
`--session` must match it exactly.

GPU Python uses compatibility mode: the runner validates the ticket, exact
workdir, Conda Python and an explicit script when used, redirects common
ML/build caches and temporary/config/state paths, sets the ticket-owned GPU
visibility, uses no shell, and records exact detached-process identity. It does
not apply Landlock to the workload process tree, because normal CUDA, NCCL,
DDP, DataLoader workers, compilation and checkpoint/log/result writes take
priority. Workload code must therefore be trusted, and keeping all user assets
inside the two roots is an operator and code-review contract.

Git, Conda/nvitop, dashboard collection, and TensorBoard complex children keep
the fail-closed Landlock write boundary. Conda and system binaries are
read/execute exceptions. `/dev/shm` is writable for infrastructure children
that need IPC. Persistent user assets still belong in the two roots.

The infrastructure boundary is a same-SSH-user trusted-code guard against
accidental path mistakes, not a hostile-code security boundary. OpenSSH authentication and the account
shell's command startup happen before an infrastructure Landlock wrapper, so
the account and server startup configuration must be trusted. Infrastructure
Landlock covers only its launched process tree; it does not restrict network
access, GPU DMA, or a hostile kernel. Do not claim stronger isolation.
Landlock does not mediate every metadata operation (chmod/chown/xattr/utime),
and pre-opened file descriptors remain usable. Describe the enforced claims as
file content read/write, create/remove/rename, truncate, execute, and device
ioctl restrictions—not absolute malicious-code isolation.

## Resolve the command

Prefer the installed command:

```bash
remote-gpu-dev --help
```

If it is not on `PATH`, use this Skill's `scripts/remote_gpu.py` with the current
Python interpreter. Never invent a profile or silently select among ambiguous
profiles.

## First-use workflow

Run `remote-gpu-dev profiles`. When no suitable profile exists, onboard the
server interactively. Ask one question at a time, in this order:

1. Server display name and stable short slug.
2. SSH host/address, user, port, optional ProxyJump, and identity-file path.
3. Acquire candidate host keys through the exact connection route (`ssh-keyscan`
   only for a direct route; OpenSSH for an alias/ProxyJump), display their
   SHA-256 fingerprints, and require the user to verify them through a trusted
   channel before recording them.
4. Test strict key-only SSH. Classify DNS, route, timeout, refused-port,
   host-key, and public-key failures separately.
5. Only for a public-key failure, guide the user to choose or generate a
   server-specific Ed25519 key and run `ssh-copy-id` in their own terminal.
   Never request, read, save, echo, or log a password, private-key body, or key
   passphrase.
6. Ask for the local multi-project root and the shared local ticket root.
7. Ask for remote scratch/temporary and durable bases. Manage only the derived
   `remote-gpu-dev/<profile>` children, never the mount root itself.
8. Ask explicitly whether Slurm, PBS, Kubernetes, a cloud controller, or another
   scheduler already allocates the GPUs, even when no scheduler binary was
   discovered. If so, stop and use it. Then ask whether coordination is limited
   to multiple sessions on this controller
   or spans multiple computers. A local ledger is valid only for one controller;
   multi-controller use requires a genuinely shared filesystem supporting
   `flock` and atomic rename. If the server has Slurm/PBS/Kubernetes, stop and
   use that scheduler instead of claiming the local ledger coordinates it.
9. Discover GPU index, UUID, memory, MIG mode, Git, tmux, flock, Conda, and
   remote identity. Let the user choose the managed physical GPU set. This
   version fails closed on enabled MIG.
10. Ask for the network preset, optional command-scoped proxy, server-specific
    multi-GPU variables, dashboard port, TensorBoard port-pool start/end,
    reservation TTL, and heartbeat grace. Reject a reversed pool, a local
    dashboard/local-proxy collision, or a remote proxy/TensorBoard-pool collision.
11. Confirm a redacted summary, write the profile atomically, initialize the
    ledger, optionally install nvitop into a dedicated Conda monitor prefix, and
    run `remote-gpu-dev doctor`.

For a human-driven terminal wizard, run `remote-gpu-dev setup`. Read
[onboarding.md](references/onboarding.md) before handling authentication or an
unusual scheduler/topology. A successful setup proves configuration readiness,
not CUDA correctness; a CUDA witness still requires a GPU ticket.

## Choose a profile explicitly

Use `remote-gpu-dev --profile <slug> ...` in automation. `remote-gpu-dev use
<slug>` only sets the convenience default for a human launcher. Keep ticket,
dashboard, and runtime state isolated by the managed GPU coordination identity,
not by SSH host keys. SSH aliases or host-key rotations that manage the same
exact GPU UUID/index inventory must share one ledger and dashboard runtime. Any
overlapping managed GPU UUID set must fail closed unless that mapping and ticket
root are identical.

Run the read-only checks before work:

```bash
remote-gpu-dev --profile SERVER doctor
remote-gpu-dev --profile SERVER gpu --json
remote-gpu-dev --profile SERVER ticket status --json
```

Do not interpret an empty `nvidia-smi` process list alone as authorization to
use a GPU. The current physical index-to-UUID mapping must exactly match the
profile, MIG must still match the recorded `disabled` or explicitly
`unsupported` policy, and process-query errors are blockers. The ledger and all
live evidence must agree.

## Develop and deploy source

The local Git repository is authoritative. Edit, review, test, and commit
locally. The remote execution clone must be clean and detached at the exact
full commit:

```bash
remote-gpu-dev --profile SERVER project deploy PROJECT
remote-gpu-dev --profile SERVER project verify PROJECT --commit FULL_SHA
```

Each project is a first-level directory below the configured local projects
root and gets its own remote bare repository and execution clone. Git contains
source, small configs, launchers, tests, and useful documentation only. Never
track Conda environments, datasets, weights, checkpoints, logs, TensorBoard
events, records, or generated results. Deployment also rejects ignored files:
`.gitignore` is not proof that runtime data is safe to leave inside the source
worktree. Do not edit source on the server and do not run a remote coding agent.
Read [git-and-storage.md](references/git-and-storage.md).

## Reserve and run GPUs

Every CUDA operation, including a smoke test, needs a ticket. Follow this order:

1. Inspect `ticket status --json`; run `ticket reconcile` only when intentionally
   applying pending expiry/stale transitions.
2. Reserve the required count or exact physical IDs. Respect `queued`; never
   bypass it.
3. Re-query the assigned IDs/UUIDs with `gpu --json`. If a foreign process or
   unexplained memory is present, do not start and never kill it automatically.
4. Verify the remote execution clone's exact commit and a fresh, project-specific
   Conda prefix outside Git.
5. Mark the reservation running only after the live idle check. The workdir
   must be below one of the two managed roots:

```bash
remote-gpu-dev --profile SERVER ticket start TICKET \
  --confirmed-idle 0,1 --session EXACT_SESSION \
  --remote-workdir /absolute/run/path --summary 'sanitized command summary'
```

6. Launch only through structured `run`; add the exact ticket `--session` for a
   detached job. A structured supervisor writes
   exact identity, log, and final-status files below the temporary root. Use
   `run TICKET --status` and `run TICKET --stop`; both infer the recorded
   session, while stop verifies boot ID, PID, process start ticks, and
   process-group leadership before `SIGTERM`. It never uses PID-only or
   name-based killing.
   The runner sets `CUDA_VISIBLE_DEVICES`; within the process, logical CUDA
   devices are remapped to `0..N-1`. Record command, cwd, environment prefix,
   PID/session, log, commit, and input hashes. For multi-GPU work, validate every
   rank and collective; never assume NCCL works merely because
   `torch.distributed.is_available()` is true.
7. Heartbeat before the configured grace expires. A stale ticket still owns its
   GPUs. Never release or kill a job only because its heartbeat is stale.
   Treat transient SSH control-plane disconnects as normal monitoring noise.
   Declare the control channel `unavailable` only after five consecutive
   structured control checks fail; failures one through four only emit a warning
   and retry, and any successful structured check resets the count. A failed
   control check never proves that training ended and never authorizes stop or
   ticket release. Those actions still require exact tracked-process identity,
   assigned-GPU state, and final-status or completion-artifact evidence.
8. At completion, bind results to the ticket, verify logs/artifacts/checkpoint
   load as appropriate, stop only the exact tracked process/session, and check
   the assigned GPUs again. Release with exact `--confirmed-stopped` IDs and an
   honest completed/failed/cancelled result.

Read [gpu-tickets.md](references/gpu-tickets.md) for state semantics and
[execution.md](references/execution.md) for the launch and validation contract.

## Manage environments and downloads

Use Conda for every remote environment. The dedicated monitor environment is
server infrastructure; each research project gets a separate `--prefix` env.
Do not modify system Python or install research dependencies in Conda `base`.
Never invoke Conda through an arbitrary SSH command. Use structured operations
so environments, package caches, config, home, and temp stay below the roots.

```bash
remote-gpu-dev --profile SERVER infra create-env \
  --prefix /managed/temp/envs/PROJECT --python 3.12 --package pytorch
remote-gpu-dev --profile SERVER infra install-env \
  --prefix /managed/temp/envs/PROJECT --package tensorboard
remote-gpu-dev --profile SERVER infra pip-install-env \
  --prefix /managed/temp/envs/PROJECT --package tensorboard
```

Package arguments use a closed package-spec grammar; paths, URLs, requirement
files, and arbitrary pip options are rejected. `pip-install-env` always invokes
the selected prefix's `python -m pip`. Add `--proxy` only when the profile
permits its on-demand loopback proxy. A failed `create-env` removes only the
exact prefix that the command first proved absent before a direct-to-proxy
retry, so a partial first attempt cannot be mistaken for a user environment.

Apply mirrors and proxies per command:

- Use the profile's HF endpoint when it can serve the same repository/object.
- If another PyPI index is required, keep the configured TUNA URL as an extra
  index; do not silently replace an explicit primary index.
- Structured pip installs and training runs receive the profile's explicit
  primary index and ordered extra-index list; ambient local pip settings are
  never inherited.
- Try Conda directly first unless the profile says otherwise.
- If a large download is unacceptably slow and no equivalent HF/domestic source
  can accelerate the same bytes, use forwarding-only `ssh --proxy --no-command`
  alongside that structured operation.
- Verify size and upstream checksum. Closing SSH must remove the reverse
  forwarding; do not persist proxy variables in shell startup files.

Read [conda-and-network.md](references/conda-and-network.md).

## Dashboard and TensorBoard

Training code owns TensorBoard event files. A ticket stores only a validated
source and viewer lifecycle metadata. Configure a source after the ticket has a
remote workdir:

```bash
remote-gpu-dev --profile SERVER tensorboard configure TICKET \
  --env-prefix /absolute/experiment/conda-prefix \
  --logdir /absolute/run/events
```

The user alone starts or stops the frontend:

```bash
remote-gpu-dev --profile SERVER dashboard open
remote-gpu-dev --profile SERVER dashboard status
remote-gpu-dev --profile SERVER dashboard stop
```

The dashboard binds local loopback, reads GPU telemetry through a persistent
key-only SSH stream, and lets the user start/stop exact generation-pinned
TensorBoard viewers. Dashboard lifecycle never reserves, starts, heartbeats,
reconciles, releases, or kills GPU workloads. Ending a GPU ticket does not
delete event files. TensorBoard sidecar preflight, status, and exact-absence
checks are idempotent and may make up to five attempts after SSH exit 255 or an
SSH timeout. Launch and stop are single-attempt mutations: an unknown outcome is
never blindly replayed and remains generation-fenced as `cleanup_pending`.
Read [dashboard-and-tensorboard.md](references/dashboard-and-tensorboard.md).

## Storage lifecycle

Start new runs, downloads, and exploratory environments under the managed
temporary root. Promote only useful records, documentation, code-independent
datasets, and checkpoints to the managed durable root after their value is
known. Promotion is an explicit, exact-path operation with checksums; never
bulk-delete or broadly move a storage root. Git source remains in Git rather
than being copied into a durable artifact tree.

## Fail closed

- Do not use passwords, `sshpass`, `StrictHostKeyChecking=no`, broad `pkill`, or
  wildcard deletion.
- Do not initialize CUDA during a read-only doctor or ticket audit.
- Treat NVML errors, changed machine identity, missing GPU UUIDs, dirty Git,
  missing Conda, unresolved TensorBoard identity, and ambiguous process state as
  blockers.
- Do not claim success from directory presence or a zero-looking trap exit code;
  validate the workload's actual completion artifacts and result contract.
- Preserve unrelated user files and processes. Stop/delete only exact recorded
  identities and managed child paths.
