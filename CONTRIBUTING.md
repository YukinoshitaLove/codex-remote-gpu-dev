# Contributing

Keep runtime dependencies in the Python standard library. New remote operations
must construct argv from validated fields, avoid `shell=True`, preserve strict
SSH identity checks, and include isolated tests.

Before a change:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 tools/check_public_tree.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile skills/remote-gpu-dev/scripts/*.py
bash -n skills/remote-gpu-dev/scripts/ssh_remote.sh
node --check skills/remote-gpu-dev/assets/dashboard/app.js
```

Do not add real profiles or a live integration test to ordinary CI. Real GPU
smokes require a separate opt-in environment, a valid ticket, and explicit
cleanup evidence.
