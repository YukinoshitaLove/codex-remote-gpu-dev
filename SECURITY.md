# Security policy

## Reporting

Do not open a public issue containing a credential, internal hostname, private
IP, host key, identity path, ticket content, or experiment data. Use the private
security-reporting channel configured by the repository owner.

## Credential model

This project must never persist:

- SSH passwords or private-key bodies;
- private-key passphrases;
- API/Hugging Face/GitHub tokens;
- proxy credentials or credential-bearing URLs;
- environment dumps that may contain credentials.

Profiles may store the absolute path to a private key. OpenSSH/ssh-agent handles
the key and any passphrase. The project never uses `sshpass` or disables host-key
checking.

Ticket free-text fields are screened for common credential labels, bearer/basic
authorization values, service-token prefixes, and credential-bearing URLs before
they are persisted. This is a defense-in-depth guard, not a general secret or
entropy detector: paste only sanitized summaries into ticket fields.

## Before publishing

Start from a clean repository with no copied `.git` history. Run:

```bash
python3 tools/check_public_tree.py
# Optional: one exact private deny value per line, stored outside this checkout.
python3 tools/check_public_tree.py --local-deny-file /absolute/private/deny-values.txt
git grep -n -I -E 'BEGIN .*PRIVATE KEY|password[[:space:]]*[:=]|token[[:space:]]*[:=]'
git log -p --all
```

The optional deny file must be UTF-8, mode `0600` or stricter, and outside the
release tree. Never embed workstation paths, hostnames, account names, or
credentials in the checker itself, even as split or encoded literals.

The public-tree checker accepts only its explicit UTF-8 text file allowlist. It
rejects unknown file types, non-UTF-8 or binary-control content, and common PEM,
OpenPGP, SSH2, PuTTY, and age private-key containers. Add a new public file type
to the allowlist only together with a focused regression test and manual review.
It is an accidental-release prevention check for common static credentials and
runtime artifacts, not a proof against arbitrary programs, encoding, encryption,
or deliberate obfuscation.

Inspect the full Git history, not only the current tree. `.gitignore` is not a
security boundary. Never commit profiles, known-hosts files, ticket ledgers,
runtime state, logs, records, data, weights, or checkpoints.

## Runtime boundaries

- Compatibility is the highest-priority runtime requirement: ordinary training,
  testing, inference, CUDA/NCCL/DDP, DataLoader workers, compilation, checkpoint
  save/load, logging, and result saving must work through a short normal path.
  Managed paths, tickets, and infrastructure isolation are trusted-code guards
  against accidental writes and allocation conflicts, not hostile-code
  containment. A reproduced compatibility failure is a reason to update the
  narrowest responsible Skill/helper and add a regression test, not to impose a
  permanent project workaround. Profile roots are configurable, not hard-coded
  universal paths. If a restriction blocks a core operation, the AI may directly
  relax it in the minimum necessary scope and synchronize the helper, tests,
  documentation, and release metadata; an older restriction is not immutable.
- The ticket ledger is not a substitute for Slurm/PBS/Kubernetes.
- Dashboard and TensorBoard bind loopback only.
- Interactive SSH and arbitrary remote commands are disabled. GPU Python uses
  a ticket-bound compatibility runner without Landlock so normal CUDA, NCCL,
  DDP, DataLoader and artifact saving are not obstructed. The runner validates
  the canonical ticket workdir, Conda Python, and explicit script when used,
  and redirects common caches and temporary state, but
  workload code must be trusted. Infrastructure helpers retain Landlock ABI 5+
  and may write only under the profile temporary or durable root.

The infrastructure boundary is a same-SSH-user trusted-code guard against
accidental path mistakes, not a hostile-code or VM/container boundary. OpenSSH authentication and account-shell
command startup occur before the client-started Landlock wrapper. The remote
account, SSH server, startup configuration, kernel, and GPU driver remain in the
trusted computing base. Landlock does not restrict network access or GPU DMA.
It also does not mediate every metadata operation (for example chmod, chown,
xattr, or timestamps), and file descriptors opened before restriction remain
usable. The fail-closed claims cover file content read/write, create, remove,
rename/refer, truncate, execution, and device ioctl rights handled by ABI 5.
- A stale ticket continues to own its GPUs.
- A process is stopped only after exact boot/PID/start-ticks/cmdline/session/
  generation verification.
- Detached training records ticket/session/workdir, PID, process start ticks,
  boot ID, log, and final status inside the managed temporary root. Its stop
  path signals only the still-matching supervisor; that supervisor forwards to
  its separately-led worker process group and remains alive to record final status.
- Remote managed paths must be absolute children more specific than a mount
  root. Broad or wildcard deletion is prohibited.
