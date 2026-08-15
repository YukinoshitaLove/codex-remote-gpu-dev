# Server onboarding

## Connection triage

Test in layers and do not confuse them:

1. Resolve the host and reach the SSH TCP port.
2. Obtain candidate host keys with `ssh-keyscan` only for a direct literal route.
   For an SSH config alias, `ProxyJump`, or `ProxyCommand`, acquire the candidate
   key through that exact OpenSSH route into a temporary private known-hosts
   file. The candidate connection disables authentication and does not run a
   remote workload.
3. Show SHA-256 fingerprints and require out-of-band verification. A keyscan is
   not proof of identity.
4. Save keys in the profile-specific known-hosts file.
5. Run OpenSSH with `BatchMode=yes`, `IdentitiesOnly=yes`, password and
   keyboard-interactive authentication disabled, and strict known-host matching.
6. Run a minimal identity probe before hardware discovery.

Only `Permission denied (publickey)` enters the public-key repair branch. DNS,
timeout, refused port, no route, VPN, ProxyJump, and host-key mismatch require
different fixes.

## Public-key repair

Prefer an existing server-specific Ed25519 identity. Otherwise the user may run:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_SERVER -C remote-gpu-dev
ssh-copy-id -i ~/.ssh/id_ed25519_SERVER.pub -p PORT USER@HOST
```

Those commands run in the user's own terminal. A password prompt belongs to
OpenSSH, never to Codex or a profile file. Encrypted keys use `ssh-agent` or the
OS keyring. Private-key mode must be `0600` or stricter.

If public keys cannot be installed, automated operation is blocked. Do not fall
back to stored passwords.

## Discovery questions and probes

- Does SSH need a VPN, config alias, or ProxyJump?
- Is there an existing scheduler such as Slurm, PBS, Kubernetes, or a cloud job
  controller? Ask explicitly even when scheduler client binaries are absent. If
  yes, use it as the allocator instead of a parallel file ledger.
- Is this physical server also reachable through another alias/profile?
- Will one controller computer or multiple computers submit work?
- Are GPUs full physical devices or MIG instances? V1 requires MIG disabled.
- Which GPU UUIDs are managed and which are intentionally excluded?
- Where are scratch and truly persistent mounts, what are their quotas, and what
  survives reboot or instance destruction?
- Which Conda executable is authoritative? Do not assume interactive `.bashrc`
  makes `conda` available over non-interactive SSH.
- Are `git`, `tmux`, `flock`, reverse forwarding, and loopback ports available?
- Does the host require variables such as `NCCL_IB_DISABLE=1`? These are
  profile-specific, never universal defaults.
- Are Git submodules or LFS genuinely required? They are disabled by default.

## Readiness levels

Report these separately:

- `configured`: profile and strict SSH identity are valid.
- `environment_ready`: paths, Git, tmux, Conda, and monitor environment are ready.
- `cuda_witness_passed`: a ticketed single-GPU operation ran successfully.
- `distributed_ready`: a ticketed multi-rank collective ran successfully with
  rank/device/process evidence.
- `dashboard_ready`: optional nvitop and local viewer are working.

Never upgrade one level based only on a lower-level probe.
