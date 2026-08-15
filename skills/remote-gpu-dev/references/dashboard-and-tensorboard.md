# Dashboard and TensorBoard

## Ownership split

- Training owns and updates event files.
- The ticket ledger records a validated event source and exact viewer identity.
- The user-controlled local dashboard starts/stops viewers and SSH tunnels.
- GPU allocation and workload processes are independent of all viewer states.

Codex may configure the event source as part of a training setup. It must not
automatically open or keep the frontend alive for every session. The user uses
the desktop entry or `remote-gpu-dashboard`/`dashboard open`.

## Security boundaries

- Remote TensorBoard binds `127.0.0.1` and uses a profile port pool.
- Local dashboard binds `127.0.0.1` and uses a capability bootstrap plus a
  private same-origin session cookie.
- TensorBoard is displayed on an isolated localhost viewer origin.
- Only the minimal read queries used by TensorBoard scalars/time-series are
  proxied; cookies, authorization, hop-by-hop headers, and ambient HTTP proxies
  are stripped or bypassed.
- Stop requires matching ticket ID, generation, tmux session, boot ID, PID,
  process start ticks, command hash, and registered port. A newer generation is
  left untouched.
- The remote TensorBoard/tmux process tree requires Landlock ABI 5+, uses the
  managed `TMUX_TMPDIR`, and can write user files only in the two managed roots.
- The strict launch helper opts into only `/dev/ptmx` and `/dev/pts` so tmux can
  allocate its detached pseudoterminal. It does not expose the whole `/dev` tree,
  and other strict infrastructure helpers keep this exception disabled.

Closing a panel merely unloads the iframe. Closing the dashboard stops only the
viewer generations it created and its SSH tunnels. It does not stop training or
change GPU ticket allocation. Event files remain after viewer or ticket stop.

## Failure states

`starting`, `failed`, and `cleanup_pending` remain visible and offer exact
generation-pinned cleanup. Do not reuse a registered port until cleanup is
confirmed. After a server reboot, old process identities cannot be signaled; the
sidecar must prove the exact session and port are absent before recording stop.

Read-only sidecar preflight, status, and exact-absence checks retry only SSH exit
255 and SSH timeout failures. They make at most five consecutive attempts under
one shared deadline, each attempt is capped at six seconds, failures one through
four emit a warning, and any successful structured response ends the failure
streak. Other errors fail immediately.

Launch and stop change remote process state and therefore run exactly once. If
SSH disconnects before their result is known, the sidecar does not blindly
replay them: it preserves the same generation and records `cleanup_pending`
until an idempotent status or absence check can resolve the remote state.
