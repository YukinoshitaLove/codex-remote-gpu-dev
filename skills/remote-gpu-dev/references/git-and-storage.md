# Git and storage

## Source topology

For each project:

```text
local:              <local.projects_root>/<project>
remote bare:        <remote.git_bare_root>/<project>.git
remote execution:   <remote.projects_root>/<project>
```

The deploy helper accepts only a clean local root, a full commit, a clean remote
clone, the exact expected origin, and a detached exact checkout. It rejects
common data/weight/checkpoint paths, large tracked files, ignored runtime files,
and tracked symlinks whose targets are not commit content. Submodules and LFS
are disabled unless the profile explicitly opts in and the caller verifies all
objects.

Remote Git and its receive-pack transport run inside the same fail-closed
Landlock boundary as workloads, so hooks or configuration cannot write user
assets outside the temporary/durable roots.

Never edit source on the remote machine. Bring a needed change back to the local
repository, review and commit it, then deploy the new commit.

## Artifact topology

Start with scratch paths:

```text
<remote.temp_root>/runs/<project>/<run-id>
<remote.temp_root>/envs/<project>/<env-id>
<remote.temp_root>/downloads/<project>/<name>
```

Promote only useful, verified assets:

```text
<remote.records_root>/<project>/<run-id>
<remote.durable_root>/datasets/<dataset>
<remote.durable_root>/checkpoints/<model-or-project>
<remote.durable_root>/documents/<project>
```

Promotion is a later judgment, not an automatic consequence of successful
training. Record source path, destination, size, checksums, and whether the
scratch copy remains. Never treat the managed root or mount root as a recursive
delete target.

Git source and durable artifacts are separate. Useful documentation may live in
Git when it is small and source-oriented; raw run records, event files, data,
weights, and generated results do not.
