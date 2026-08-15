# GPU ticket contract

The ledger uses `flock`, a temporary file, file and directory `fsync`, and
`os.replace`. `status` is zero-write; `reconcile` explicitly applies time-based
transitions.

## States

- `queued`: waits FIFO and owns no GPU.
- `reserved`: owns exact physical IDs during a short launch window.
- `running`: workload is tracked and heartbeats are current.
- `stale`: heartbeat is late but the ticket still owns its GPUs.
- `completed`, `failed`, `cancelled`, `expired`: terminal and own no GPU.

TensorBoard metadata has an independent state machine. Releasing a GPU ticket
does not start or stop a viewer.

## Reserve

Supply a sanitized project, owner, purpose, expected duration, and count or exact
physical IDs. Never include tokens, credentials, full secret-bearing commands,
or private dataset names that should not appear on the board. A queued response
is not authorization to run.

## Start

Immediately before start, bind these facts:

- ticket ID and assigned physical indices;
- the corresponding expected GPU UUIDs;
- remote host identity and timestamp;
- no foreign compute process or unexplained memory on each assigned GPU;
- exact tmux/session name, absolute workdir, Conda prefix, source commit, and
  sanitized command summary.

`--confirmed-idle` must exactly match the assignment. An empty process table and
a ledger reservation are both required; neither is sufficient alone.

## Heartbeat and stale recovery

Heartbeat before grace expires. If stale, inspect the recorded tmux pane, PID
tree, command, cwd, logs, boot ID, and `nvidia-smi`. If the workload is still
live, heartbeat and keep the GPU held. Session absence alone or an empty
`nvidia-smi` table alone is not enough to release.

## Release

Stop only the exact tracked process/session when stop was requested or the
workload naturally exited. Recheck process identity and every assigned GPU.
`--confirmed-stopped` must exactly equal the assigned physical IDs. State the
honest outcome and bind it to logs, result files, checkpoint hashes, or failure
evidence.

Never auto-kill, auto-release, or reassign merely because a ticket is stale.
