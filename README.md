# Codex Remote GPU Dev

English | [简体中文](README.zh-CN.md)

`remote-gpu-dev` is a reusable Codex Skill and local CLI for safely developing
against SSH-accessible NVIDIA GPU servers. It turns server-specific details into
private profiles and keeps source, GPU allocation, runtime artifacts, and
monitoring ownership separate.

The repository is intentionally public-ready: it contains no raw server
profiles or ticket ledgers, credentials or other secrets, datasets, weights,
checkpoints, training logs, or TensorBoard event files. The visual tour includes
two generated architecture diagrams, two sanitized historical dashboard
screenshots, and one simulated setup screen. The connection endpoint is
pixel-mosaicked, and the raw browser captures are not included.

## Compatibility-first development contract

This Skill is an evolvable development tool, not a frozen policy. Its highest
priority is a short, out-of-the-box path for normal training, testing, inference,
CUDA/NCCL/DDP, DataLoader workers, compilation, checkpoint save/load, logging,
and result saving. Profile-managed roots, GPU tickets, structured launch, and
infrastructure isolation prevent accidental writes and allocation conflicts by
trusted AI-authored code; they must not impede normal PyTorch. The root paths are
configured per server profile rather than hard-coded by the Skill.

If a reproducible compatibility problem appears, update the narrowest affected
Skill instruction, helper, and focused regression test instead of requiring a
project-specific workaround. Keep unrelated credential, ownership, process
identity, and destructive-action safeguards intact. This is a trusted-code
workflow, not hostile-code containment.

If any restriction blocks a core operation, the AI may directly relax it in the
minimum necessary scope and synchronize the helper, tests, documentation, and
release metadata. An older restriction is not immutable and does not take
priority over working core functionality.

Remote monitoring tolerates common SSH control-plane jitter: the control channel
is declared unavailable only after five consecutive structured control checks
fail. Failures one through four only warn and retry, and a successful check resets
the count. A control-check failure never proves training ended and never permits
stop or ticket release without exact process, GPU, and final-status evidence.

## What it provides

- step-by-step server onboarding with SSH failure classification and public-key
  guidance;
- profile-specific strict known-host verification and key-only SSH;
- SSH trust identity plus a host-key-independent GPU coordination identity;
- one file-locked GPU ticket ledger shared by multiple Codex sessions;
- local-authoritative Git source deployment to one bare repository and one exact
  execution clone per project;
- Conda-only research environments and a dedicated nvitop infrastructure env;
- HF/PyPI mirror policy and temporary SSH reverse proxy forwarding;
- profile-specific multi-GPU environment variables such as
  `NCCL_IB_DISABLE=1` when the host requires them;
- a loopback-only GPU/ticket dashboard with manually controlled TensorBoard
  viewers;
- explicit scratch-first and later durable-promotion storage policy;
- a two-root remote asset policy, structured ticket-bound PyTorch runner, and
  fail-closed Landlock for infrastructure helpers; arbitrary remote commands
  and interactive shells are disabled.

## How the pieces fit

![End-to-end remote GPU development workflow](docs/assets/diagrams/system-workflow.png)

*What this diagram explains.* Source remains authoritative in local Git, then a
clean execution clone is deployed before GPU reservation. The ticket-bound
runner launches normal PyTorch, CUDA, NCCL, DDP, and DataLoader work in a
separate recorded run directory so checkpoints, logs, and results have one
predictable destination. Read-only GPU telemetry and TensorBoard event data feed
monitoring without changing ticket ownership. Validation checks final artifacts,
exact process identity, and assigned-GPU state before the ticket is released.
The numbered transitions therefore cover deploy, reserve, launch, observe,
validate, and release rather than hiding those boundaries in one remote shell.

## Trust model and limits

- Passwords, private-key contents, passphrases, tokens, and authenticated proxy
  URLs are never stored. Profiles contain only an identity-file path.
- The user must verify first-use SSH host-key fingerprints through a trusted
  channel. `ssh-keyscan` alone is not authentication.
- The v1 ledger coordinates multiple sessions on one controller computer. It
  coordinates multiple computers only when its directory is on a genuinely
  shared filesystem that supports `flock` and atomic rename. Prefer Slurm/PBS or
  another existing scheduler when present.
- V1 allocates full physical GPUs and fails closed when MIG is enabled.
- The dashboard is not a GPU scheduler. It never reserves, heartbeats, releases,
  or stops CUDA work.

## Ticket ledger and ownership

![GPU ticket ledger and state machine](docs/assets/diagrams/ticket-system.png)

*What this diagram explains.* Multiple controller sessions coordinate through a
file-locked ledger that is replaced atomically, so each physical GPU and each
sidecar port have one recorded owner. A ticket moves through `QUEUED`, `RESERVED`,
and `RUNNING` before ending as `COMPLETED`, `FAILED`, or `CANCELLED`. A stale
heartbeat retains ownership until exact process and GPU checks establish a safe
terminal result; SSH uncertainty alone never frees a GPU or port. Only the
release gate clears ownership. TensorBoard consumes read-only event data
independently, while the SSH control path tolerates up to five consecutive
transient failures without replaying state-changing launch or stop operations.

## Requirements

Local:

- Codex with Skill support;
- Python 3.11+;
- OpenSSH client, `ssh-keyscan`, `ssh-keygen`, and Git;
- a browser for the optional dashboard.

Remote:

- Linux, OpenSSH server, NVIDIA driver plus `nvidia-smi`;
- Linux Landlock ABI 5 or newer for the infrastructure helpers;
- Git, tmux, flock, and an existing Conda/Miniforge/Miniconda installation;
- permission to create dedicated managed subdirectories.

The wizard does not silently install Conda or change system Python.

## Install globally from an existing checkout

```bash
cd /absolute/path/to/codex-remote-gpu-dev
python3 tools/manage_install.py install
python3 tools/manage_install.py check
```

This performs a validated, atomic copy into:

```text
${CODEX_HOME:-$HOME/.codex}/skills/remote-gpu-dev
```

It also installs `remote-gpu-dev`, `remote-gpu-dashboard`, and a user-local
desktop entry. Production installation uses a copy rather than a symlink, so
moving the checkout does not break the Skill.

After publishing, the standard Codex skill installer can also install this
public repository URL:

```text
https://github.com/YukinoshitaLove/codex-remote-gpu-dev/tree/main/skills/remote-gpu-dev
```

The managed installer is recommended when you want verified updates, command
launchers, the desktop entry, and recoverable uninstall.

Restart Codex after first installation so global skill discovery refreshes.

## First server

From a terminal:

```bash
remote-gpu-dev setup
```

![Simulated interactive setup wizard](docs/assets/screenshots/setup-wizard-simulation.png)

*What this screenshot demonstrates.* This is a browser-rendered simulation of
the one-question-at-a-time onboarding flow. Every hostname, path, fingerprint,
GPU identifier, and port is fictional (`example.invalid`); it performs no SSH
connection and writes no profile. It exists to show the order and meaning of
the prompts without publishing a real server configuration.

The wizard asks one question at a time, starting with the server name and SSH
address. If public-key authentication fails, it prints a safe `ssh-keygen` and
`ssh-copy-id` workflow; the password prompt, if any, stays inside your own
OpenSSH terminal.

Profiles are written outside the repository:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/remote-gpu-dev/profiles/<slug>.json
${XDG_CONFIG_HOME:-$HOME/.config}/remote-gpu-dev/known_hosts/<slug>
```

Tickets and dashboard runtime state are also outside Git. Profile and ticket
paths are mode `0600`/`0700` where applicable.

Run the read-only readiness check:

```bash
remote-gpu-dev --profile my-server doctor
remote-gpu-dev --profile my-server gpu --json
```

## Daily workflow

```bash
# Edit, test, review, and commit locally first.
remote-gpu-dev --profile my-server project deploy my-project

# Create the project environment under one managed root.
remote-gpu-dev --profile my-server infra create-env \
  --prefix /managed/temp/envs/my-project --python 3.12 --package pytorch
remote-gpu-dev --profile my-server infra pip-install-env \
  --prefix /managed/temp/envs/my-project --package tensorboard

# Inspect then reserve.
remote-gpu-dev --profile my-server ticket status --json
remote-gpu-dev --profile my-server ticket reserve \
  --project my-project --owner "$USER" --purpose "training run" --gpus 2 --expected 4h

# After checking the assigned GPUs are live-idle, record the exact launch.
remote-gpu-dev --profile my-server ticket start TICKET_ID \
  --confirmed-idle 0,1 --session exact-session \
  --remote-workdir /absolute/run/path --summary "sanitized command summary"

# Heartbeat while it runs, then validate, stop the exact process, recheck GPUs,
# and release with an honest result.
remote-gpu-dev --profile my-server ticket heartbeat TICKET_ID --expected 2h
remote-gpu-dev --profile my-server ticket release TICKET_ID \
  --outcome completed --confirmed-stopped 0,1 --result "result.json and checkpoint verified"
```

Interactive shells and arbitrary remote commands are intentionally unavailable.
Use the ticket-bound structured runner:

```bash
remote-gpu-dev --profile my-server run TICKET \
  --env-prefix /managed/temp/envs/demo \
  --script /managed/temp/projects/demo/train.py -- --epochs 20
```

The runner reads its working directory from the ticket. This keeps the clean
execution clone and the run directory separate, so relative checkpoints, logs,
and results land in the recorded run directory. `--workdir` remains available
as an optional assertion and must exactly match the ticket. Use `--module` with
a dotted ASCII module name when Python's module form is needed, for example
`--module torch.distributed.run`. Arguments after `--` pass through unchanged.

Add `--session EXACT_TICKET_SESSION` for a detached job. The structured
supervisor records `identity.json`, `run.log`, and `final.json` below
`remote.temp_root/runtime/runs/TICKET/jobs/SESSION`. Inspect or stop it only
through the same ticket. Status and stop infer the recorded session; an
explicit `--session` is accepted only when it exactly matches:

```bash
remote-gpu-dev --profile my-server run TICKET --status
remote-gpu-dev --profile my-server run TICKET --stop
```

Stop verifies boot ID, supervisor PID, process start ticks, and process-group
leadership before sending `SIGTERM`; it never falls back to PID-only or
name-based killing. Use `ssh --proxy --no-command` only as a forwarding-only
companion; the forwarding ends with that SSH process.

All remote user assets—including code, documents, records, datasets, weights,
checkpoints, logs, environments, downloads, caches, temporary files, sockets,
and generated results—must stay below the temporary or durable root. Complex
GPU workloads run in compatibility mode without Landlock so CUDA, NCCL, DDP,
DataLoader workers, compilation, and normal checkpoint/log/result writes work
without per-system-call exceptions. The runner still validates the ticket,
managed workdir, Conda Python and script, redirects caches and temporary state,
sets GPU visibility, uses no shell, and records exact detached-process identity.
Workload code must be trusted; keeping persistent assets in the two roots is an
operator and code-review contract. Infrastructure helpers retain Landlock.

The infrastructure boundary is a trusted-code, same-SSH-user guard against
accidental path mistakes, not a VM/container sandbox. SSH authentication and
account-shell command startup precede the client-started Landlock wrapper, so the remote
account and startup configuration must be trusted. Network access, GPU DMA, and
a hostile kernel are outside this boundary.

## Dashboard and TensorBoard

Training writes event files. Configure a source against its ticket:

```bash
remote-gpu-dev --profile my-server tensorboard configure TICKET_ID \
  --env-prefix /absolute/conda/env --logdir /absolute/run/events
```

Only the user starts/stops the frontend:

```bash
remote-gpu-dev --profile my-server dashboard open
remote-gpu-dev --profile my-server dashboard status
remote-gpu-dev --profile my-server dashboard stop
```

### Dashboard overview

![Sanitized dashboard overview](docs/assets/screenshots/dashboard-overview.png)

*What this screenshot demonstrates.* The loopback-only dashboard renders one
card per GPU with utilization, memory, temperature, power, and process context,
plus active/queued ticket counts and ticket history. It is a sanitized
historical capture from 2026-08-13, not evidence of the server's current state.
The connection endpoint at the top is pixel-mosaicked and the raw capture is
excluded. The dashboard remains read-only: the zero allocation/queue counters
shown here do not reserve or release anything.

### Ticket-scoped TensorBoard viewer

![Historical scratch20 TensorBoard viewer](docs/assets/screenshots/dashboard-scratch20-tensorboard.png)

*What this screenshot demonstrates.* A completed and released 20-epoch CIFAR-10
ViT scratch run (`scratch20`) is selected from ticket history, and its real
scalar curves are embedded through the manually started TensorBoard sidecar.
The viewer can be started or stopped without reopening the job or changing its
released ticket state. This is also a historical 2026-08-13 capture, not proof
that TensorBoard, the job, or any GPU is running now; no event file or raw
browser capture is shipped in this repository.

The TensorBoard sidecar retries only idempotent read-only preflight, status, and
exact-absence checks after SSH exit 255 or an SSH timeout, with at most five
attempts under one deadline. Launch and stop are never blindly replayed after an
unknown outcome; the same generation remains fenced as `cleanup_pending` until
an exact status or absence check resolves it.

The desktop launcher opens the currently selected convenience profile (`use
<slug>`). Automated commands should always pass `--profile` explicitly.
Aliases that manage the same exact GPU UUID/index inventory must share one
ticket root and dashboard runtime; SSH host-key rotation does not create a new
allocation namespace.

## Update and uninstall

```bash
git pull --ff-only
python3 tools/manage_install.py update
python3 tools/manage_install.py check

# Recoverable: moves the installed Skill to $CODEX_HOME/skill-backups.
python3 "${CODEX_HOME:-$HOME/.codex}/skills/remote-gpu-dev/scripts/manage_global_install.py" uninstall
```

Uninstall keeps profiles, tickets, records, datasets, and weights. User data is
never recursively removed by this tool.

## Development

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 tools/check_public_tree.py
python3 tools/manage_install.py validate-source
python3 /path/to/skill-creator/scripts/quick_validate.py skills/remote-gpu-dev
```

The test suite uses local fakes and temporary directories; normal CI does not
connect to a real server or initialize CUDA.

## Repository layout

```text
skills/remote-gpu-dev/   installable Codex Skill
tests/                   unit/security/state-machine tests
tools/                   public-tree and managed-install helpers
examples/                non-secret example profile
docs/assets/             reviewed diagrams and sanitized public screenshots
docs/demos/              offline, fictional setup-wizard simulation
.github/workflows/       local-only CI
```

See [SECURITY.md](SECURITY.md) before publishing a fork.
