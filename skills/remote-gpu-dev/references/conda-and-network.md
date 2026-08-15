# Conda and network policy

## Conda

- Reuse an existing Conda/Miniforge/Miniconda installation discovered by its
  absolute executable path.
- Install nvitop into the profile's dedicated infrastructure prefix. Do not
  modify `base` unless the user explicitly overrides this policy.
- Create each project's research environment with `conda create --prefix` below
  a scratch or durable managed root. Save an `environment.yml` and, when useful,
  an explicit package spec outside the environment directory.
- Run only through a structured operation that places Conda itself inside the
  Landlock boundary. Conda's executable is read/execute-only; prefixes, package
  caches, config, home, and temp paths remain below the managed roots.
- Use pip only inside a selected Conda prefix and only for a package unavailable
  from the chosen Conda channels.

Create and extend project prefixes only through the structured commands:

```bash
remote-gpu-dev --profile SERVER infra create-env \
  --prefix /managed/temp/envs/PROJECT --python 3.12 --package pytorch
remote-gpu-dev --profile SERVER infra install-env \
  --prefix /managed/temp/envs/PROJECT --package tensorboard
remote-gpu-dev --profile SERVER infra pip-install-env \
  --prefix /managed/temp/envs/PROJECT --package tensorboard
```

All three commands reject paths outside the profile roots, redirect runtime caches,
and run Conda/pip and package hooks inside the same Landlock write boundary.
`pip-install-env` uses the prefix's `python -m pip` and accepts only closed
package specs, never URLs, local paths, requirement files, or arbitrary options.
The profile's primary PyPI index (when set) and ordered extra-index list are
injected into both pip installation and managed training; ambient pip config is
disabled. Thus an explicit primary remains primary while TUNA can remain extra.

For `direct-then-proxy`, a failed Conda create removes only the exact target
prefix that the operation first proved absent, then retries through the
on-demand proxy. It never deletes or overwrites a prefix that predated the
operation. If transport loss makes the first outcome unknowable, the operation
stops rather than retrying, leaving the exact prefix for inspection.

## Domestic and mirror routing

The China preset uses HF Mirror when it serves the same Hub object and keeps the
TUNA PyPI URL as an extra index when another primary index is configured. Conda
uses upstream directly first. Do not permanently rewrite a user's Conda or shell
configuration merely to accelerate one experiment.

Before substituting a dataset/model source, prove it is the same object through
an authoritative checksum, byte size, or upstream manifest. A similarly named
dataset is not equivalent.

## On-demand proxy

When a large download is slow and no equivalent domestic/HF source can serve the
same bytes:

1. Verify the configured local proxy is listening.
2. Start one forwarding-only SSH connection from the configured remote
   loopback port to the configured local loopback proxy.
3. Inject `HTTP_PROXY`/`HTTPS_PROXY` only into the structured operation and keep
   `NO_PROXY=127.0.0.1,localhost`.
4. Verify the downloaded bytes.
5. Stop the forwarding-only SSH process and confirm the remote forwarding port is
   gone.

Never persist proxy credentials, proxy URLs with embedded authentication, or
proxy exports in `.bashrc`, Conda config, tickets, or profiles.
