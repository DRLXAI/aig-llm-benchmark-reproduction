# Agent Infra Guide — reproducible local-LLM benchmark source and data

**Public source and benchmark release for the historical Apple M4 Pro Ollama study.** The measurement index on https://agentinfraguide.com/benchmarks was refreshed **October 30, 2026** and currently contains **15 distinct measured configurations** and **75 measured repetitions** (5 per configuration). The older October 3 snapshot contained 12 and 60 and is no longer the current publication.

## What is included

- `lab/src/aiglab/`: the **original historical measurement harness** (model identity and runtime preflight, streaming inference, process RSS sampling, workload execution, deterministic grading, integrity sealing and aggregation). This source is a copied snapshot of the measuring code in Agent Infra Guide commit `c3cdc057f3b7f31` (Oct 2); the lab source tree was unchanged in the later Oct 19 measuring commit `2a4b0f67c7393b8efe34fa4466f810650c500784`. The original repo is private; this public copy permits independent inspection.
- `lab/manifests/`: configurations, including exact Qwen2.5-Coder **1.5B, 3B, 7B** Q4_K_M, 4096-context, single-client manifests. The 3B and 7B manifests match the original run's `manifest.json` exactly.
- `lab/src/aiglab/workloads/`: **complete prompts, fixture sources, task implementations and deterministic checkers** for coding, research, RAG, automation and browser tasks. The complete source and fixture-digest values are cross-checked against **all 15** published runs.
- `lab/tests/`: source-level harness/workload tests; no model downloads needed.
- `public-measurements/`: an unmodified copy of the **public, redacted/sanitized** measurement index, all 15 per-run measurement extracts and `SHA256SUMS`, captured from the production website November 1, 2026.
- `scripts/verify_release.py`: offline SHA-256, record-count, repetition-count and exact workload-definition hash verifier.
- `docs/measurement-lab/`: original methodology and extension notes. Files under `lab/base44/`, private raw/sealed run artifacts, users' original model blobs and unpublished harness changes are **not distributed**.

## 1. Verify published measurements (no model or internet required)

Requires Python 3.11+ (macOS needed only for executing untrusted generated Python safely):

```bash
python3 scripts/verify_release.py
# expected: 15 runs, 75 measured repetitions, 16 checksum-verified public data files,
#           15 matching workload-definition hashes
```

This verifies the checksum **of the sanitized public extracts**. The `source_seal_digest` fields reference the private originals, not an independently provable copy of that raw seal; do not confuse export verification with independently auditing unpublished original artifacts.

## 2. Inspect and test the original workload and checker code

```bash
cd lab
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest ruff mypy  # optional dev tools
PYTHONPATH=src python -m aiglab list --no-models
PYTHONPATH=src python -m aiglab validate manifests/*.json
PYTHONPATH=src python -m pytest tests/test_workloads.py tests/test_manifest.py tests/test_adapters_runner.py tests/test_aggregate.py
```

The package runtime is **stdlib-only**. No online services or models are needed for source-level tests. Published task definitions include the three coding prompts (`slugify`, `merge_intervals`, `parse_duration`) and exact grading assertions. No previously hidden evaluation logic is withheld from this historical reproduction copy.

## 3. Run a real replication (local M4 Pro plus pinned Ollama/models required)

On an Apple M4 Pro with 24 GiB unified memory, install a model and the required Ollama version deliberately. **Nothing in the harness downloads model weights automatically.** The published October 2 runs used Ollama 0.35.0; the October 19 Qwen2.5-Coder runs used Ollama 0.40.3. Model artifact SHA-256 values and runtime versions are listed in each corresponding file in `public-measurements/`; the same model tag may resolve to different bytes later. For a strict replication, pin `model.expected_artifact_digest` from the published run; the preflight will reject a mismatched artifact. The host must be able to use Apple's `sandbox-exec` for the coding checker.

```bash
cd lab
PYTHONPATH=src python -m aiglab validate manifests/qwen2.5-coder-3b-q4km-m4pro24-ollama-ctx4k-c1-coding.json
PYTHONPATH=src python -m aiglab dry-run manifests/qwen2.5-coder-3b-q4km-m4pro24-ollama-ctx4k-c1-coding.json
# Inspect preflight and environment; run only after intentionally installing/pinning
# the model and runtime, and ensuring no other benchmark or critical host work is running:
PYTHONPATH=src python -m aiglab run manifests/qwen2.5-coder-3b-q4km-m4pro24-ollama-ctx4k-c1-coding.json
```

Do **not** run this last command simply to audit the archive: it performs inference and uses host memory/CPU. Compare the resulting new measured run with the published per-task results and speed ranges; close numerical identity is not guaranteed on a shared Mac (background load, runtime version, model digest and effective context affect results). This publication is a *reproduction method and source snapshot*, not proof a third party has replicated the study or that the historical private run directories have been released.

## Limitations and correction history

- This is one host, not a cross-hardware ranking, and the original measurements are historical. Apple Silicon uses shared unified memory; sampled process RSS is not GPU VRAM or a full-machine power metric.
- Inference runs retain failed tasks and warmups. The coding suite contains 3 distinct function specifications repeated 5 times: **15 checks per run are not 15 independent tasks**.
- Original self-contained harness source and test definitions are now available, addressing the *source transparency* concern, but hardware, model weights, exact old Ollama binaries and unpublished raw run artifacts are not redistributed. The public data is sanitized and does not include generated answer text.
- The October 30 production snapshot is the source of truth for the current table. If you saw 12/60 and the October 3 date, that was a **previous deployment**, not this archive. Re-open https://agentinfraguide.com/benchmarks and download the `index.json` and checksums for current facts.

Questions and corrections are welcome. This is a technical evidence package, **not an affiliate link or paid download**.
