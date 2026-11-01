# Extending the lab, and reproducing a result

## Add a model

1. Make it available to the runtime **deliberately** (the harness never downloads anything):
   `ollama pull <ref>` — a multi-GB download you start yourself.
2. Copy a manifest in `lab/manifests/`, set `model.family/name/runtime_ref/quantization`
   (and `parameter_count` if known), give it a new `id` equal to the file name.
3. `uv run aiglab validate manifests/<id>.json && uv run aiglab dry-run manifests/<id>.json`.
   Dry-run shows the runtime-reported quantization, trained context and capabilities; a declared
   quantization that differs blocks the run.
4. Optional strict pin: copy `artifact_digest` from the dry-run's `preflight.identity` into
   `model.expected_artifact_digest`.
5. For Base44: create/match the ModelVariant in Planner Data Studio and add
   `"<lab key>": "<base44 id>"` to `lab/base44/id-map.json` (the lab key is printed in the
   blocked reason of `aiglab base44-plan`).

## Add a hardware target

The hardware profile key is derived from the host (`apple-m4-pro-24gb`): run the lab **on that
machine**. Clone the repo, `cd lab && uv sync`, then use the same manifests (set
`target.hardware_profile` to describe it; the observed key is what is recorded). Map the key to
a Base44 HardwareProfile id in `id-map.json`.

Cloud hosts: `target.kind: cloud` is rejected at preflight. Running on a rented GPU means
running the lab *on* that host (it is then "local" to the harness) — which needs explicit
authority to provision and pay for the instance. Record `provider`, `region`, `instance`, and
`hourly_usd` + `price_source_url` in the manifest; the price is never turned into
`cost_per_task` inside an `aig_tested` record.

## Add a runtime

1. Implement `RuntimeAdapter` (`lab/src/aiglab/adapters/base.py`) in `adapters/<id>.py`: detect,
   resident models, model identity from **disk** (artifact digest), load, loaded state (effective
   context, serving PID, served artifact), streaming chat with runtime timings, log excerpt.
2. Register it in `adapters/registry.py` (`implemented=True`) and `get_adapter`.
3. Add identity checks that can actually fail for that runtime (see `_identity_checks`).
4. Only after a real run has been reviewed: add the id to `EXECUTING_ADAPTERS` in
   `evidence.py`. That line is the methodology decision that the adapter measures rather than
   computes; until it is added, the adapter's runs are `unknown`.
5. Tests: an integration test against a fake API proving a mock runtime stays `unknown`
   (pattern: `tests/test_adapters_runner.py::test_mock_runtime_never_yields_aig_tested`).

Detected on the AIG Mac 2026-10-02: only Ollama 0.35.0. llama.cpp, MLX-LM, LM Studio and vLLM
are not installed (Ollama bundles its own `llama-server`, which the Ollama adapter observes but
does not drive directly).

## Add a workload

1. New module in `lab/src/aiglab/workloads/` subclassing `Workload`: `id` = a Planner workload
   key, `version = "1.0.0"`, fixed tasks, deterministic checker, and any external service
   isolated in `setup()`/`teardown()` (declare it in `isolated_services`; a genuinely external
   dependency goes in `external_dependencies` and must be named in the methodology).
2. Use fictional or self-contained data so a model cannot pass from memory.
3. Register it in `workloads/__init__.py`; add tests for the checker (pass and fail cases).
4. Changing an existing workload's prompts, data or checker ⇒ bump its `version`. Old manifests
   then fail validation with a pointer to reproduce from the git commit that defined the old
   version.

## Reproduce a prior result

```bash
cd lab
uv run aiglab reproduce <run_id> --out /tmp/repro.json
```

Prints what to pin: methodology version, harness version **and git commit**, workload version
and definition hash, runtime version, model artifact digest, hardware key. To reproduce exactly:

1. `git checkout <harness commit>` (or confirm `lab/src` is unchanged since).
2. Same runtime version (`ollama --version`), same artifact (set
   `model.expected_artifact_digest`; preflight blocks if it differs), same hardware key.
3. `uv run aiglab run /tmp/repro.json` — a new run id; compare `aggregate.json` medians and
   ranges. Differences within the original run's min–max are reproduction; outside it, look at
   host conditions (`host_at_start`, power source, other processes) first.
