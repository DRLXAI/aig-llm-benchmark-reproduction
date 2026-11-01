# AIG Measurement Lab (`aiglab`)

Evidence-safe benchmark and compatibility harness for Agent Infrastructure Guide.

Start with [`docs/measurement-lab/README.md`](../docs/measurement-lab/README.md): methodology,
commands, evidence rules, Base44 sync and the Estate job flow all live there.

```bash
cd lab
uv run aiglab list                                   # runtimes (detected honestly) and workloads
uv run aiglab validate manifests/*.json
uv run aiglab dry-run manifests/<manifest>.json      # preflight only; executes nothing
uv run aiglab run manifests/<manifest>.json          # genuine execution on this host
uv run pytest && uv run ruff check . && uv run mypy
```
