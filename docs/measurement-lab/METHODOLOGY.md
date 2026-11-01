# AIG Measurement Lab — Methodology 1.0.0

`METHODOLOGY_VERSION = "1.0.0"` in `lab/src/aiglab/methodology.py`. Records are only compared
with records of the same **major** version. Changing what a metric means, how it is sampled, or
which runs count requires a new version; changing a workload's prompts, data or checker
requires a new workload version (its `definition_sha256` changes either way and is recorded in
every run).

## What one run is

A **run** executes one manifest: one model artifact, one runtime, one host, one context window,
one concurrency, one workload version.

1. **Preflight** (also what `aiglab dry-run` does; nothing is executed):
   runtime reachable · model present · declared quantization equals runtime-reported ·
   context ≤ trained context · artifact on disk · artifact pin (if set) · coding sandbox present ·
   **no other model resident** (contention) · memory pressure not critical · target is local.
   Any blocking failure ⇒ the run does not start and is recorded as *blocked*
   (`evidence/measurements/blocked/`).
2. **Host lock** (`~/.aiglab/run.lock`): one benchmark per host, shared by CLI, queue and Estate.
3. **Artifact identity**: re-hash the model blob (sha256) and compare with the runtime's manifest
   layer digest.
4. **Load**: one tiny request with the manifest's options. Recorded as cold start when the model
   was not resident before (`load_ms` is runtime-reported).
5. **Loaded-state identity checks** (all must pass for `aig_tested`):

   | check | passes when |
   |---|---|
   | `manifest_digest` | API digest == sha256 of the runtime's manifest file on disk |
   | `serving_requested_artifact` | the serving process's `--model` blob == the model layer digest |
   | `artifact_hash` | sha256 of the blob == layer digest |
   | `effective_context` | runtime-reported per-slot context == manifest `context` |
   | `model_resident` | runtime lists the model as loaded |
   | `memory_sampling_target` | exactly one serving process for this artifact was found |
   | `no_reload_during_run` | same serving PID at start and end |
   | `no_contention_during_run` | no other model resident at end |

6. **Repetitions**: `warmup` repetitions, then `measured` repetitions. Each repetition runs
   `concurrency` lanes in parallel; each lane runs every task of the workload in order. Warmups
   are retained in `run.json` but never enter statistics.
7. **Seal**: every raw file is hashed into `SHA256SUMS`, written last, then the directory is
   atomically renamed from `.<run_id>.partial` to `<run_id>`. A crashed run is still sealed with
   `status: aborted` (and is never `aig_tested`).
8. **Aggregate** (`aggregate.json`, derived, outside the seal, records the seal digest it used).

### Prompt-cache isolation (why prompts start with `[session …]`)

Ollama 0.35 / llama.cpp reuse KV cache for a matching prompt prefix, across requests and even
across different prompts. Before this rule, measured repetitions replayed identical prompts and
reported **~29 ms "TTFT" for a 162-token prompt that takes ~430 ms to process cold** — a cache
measurement, not a model measurement (found on the first real batch, 2026-10-02; that batch was
discarded before publication). Every repetition × lane therefore starts its first message with
a unique session line. Nothing is shared *across* repetitions; turns *within* one conversation
and tasks within one session still share their prefix, as a real agent's would. The runtime's
`prompt_eval_cached_count` is recorded per request and summed per repetition
(`prompt_cached_tokens_total`), so any residual cache effect is visible.

## Metrics

All durations are milliseconds unless stated; "runtime-reported" means the runtime's own timer
on this host during this run; "client" means the harness's monotonic clock.

| metric | definition | source | used for Base44 |
|---|---|---|---|
| `decode_tokens_per_s` | per request: `eval_count / eval_duration`; median of successful requests | runtime-reported | `tokens_sec`, `observed_tokens_sec` |
| `aggregate_output_tokens_per_s` | output tokens of all lanes ÷ repetition wall time | client | notes |
| `ttft_ms` | request sent → first **content** chunk, streaming; responses whose first chunk is a tool call are excluded (tool calls arrive complete, so their first-chunk time is not a first-token latency) | client | `ttft` |
| `prompt_eval_ms` | prompt processing time | runtime-reported | — (lab record) |
| `task_completion_s` | one task's wall time: all model turns plus the agent's own tool/retrieval steps; **excludes grading** | client | `completion_time` (seconds) |
| `tool_latency_ms` | median duration of one tool execution (browser_tool: HTTP GET to the local fixture site; rag: retrieval step) | client | `tool_latency` |
| `peak_rss_bytes` | max over samples of the serving process's RSS (`ps -o rss=`, every 0.25 s); on Apple silicon this includes resident memory-mapped weights used in place by Metal | OS | `observed_peak_memory`, `peak_memory` (**decimal GB, max over measured repetitions**) |
| `peak_phys_footprint_bytes` | max over samples of `footprint -p` (dirty + compressed; excludes clean mapped weights), every 2 s | OS | — (lab record: KV cache + buffers proxy) |
| `success_rate`, failures | deterministic checker per task | harness | `failures` = failed tasks across measured repetitions |
| runtime completion | `completed` if every request returned a final chunk without error; `partial` / `failed` / `unknown` otherwise | harness | `completion_status` |
| runtime-reported memory | KV cache, model and compute buffer sizes parsed from runtime log lines written during the run | runtime log | lab record only |

Peaks are **lower bounds** (maximum of samples). Sample counts and intervals are stored next to
every value.

**Not measured, recorded as unavailable with a reason** (never estimated):

* **power** — `powermetrics` needs root; the harness never escalates privileges.
* **cost_per_task** — local hardware has no observed per-task price. A cloud target may declare
  `hourly_usd` + `price_source_url`, but no cloud adapter exists in 1.0.0, so no cost is ever
  produced; a future version would publish it as a separate `calculated` value, never inside an
  `aig_tested` record.
* **GPU memory attributed outside the process** — Metal allocations the OS does not attribute to
  the serving process are not visible without privileges.

## Aggregation

* Every measured repetition is retained (`values[]`).
* **Outliers**: modified z-score `0.6745·|x−median|/MAD > 3.5`; when MAD = 0, any value > 10 %
  from the median. Checked on continuous metrics only. Outliers are **flagged and kept**.
* **Exclusions** only by an explicit operator command with a written reason
  (`aiglab aggregate <run_dir> --exclude <rep>:<reason>`). Statistics are reported for `all`
  repetitions and for `included` ones side by side.
* Headline values: median; memory uses the maximum (capacity planning needs the peak).

### Confidence

* `high`: ≥ 5 included measured repetitions, no flagged outliers in decode rate / peak RSS /
  TTFT, decode-rate CV ≤ 10 %, no exclusions, host memory pressure normal at start.
* `medium`: ≥ 3 included repetitions (or a `high` candidate that failed one condition above —
  the reason is listed).
* `low`: fewer than 3. `unknown`: nothing measured.

## Host conditions

Recorded at run start: memory pressure level and free %, power source (AC/battery), load
average, chip, core counts, GPU core count, total memory, OS build, harness version, git commit
and whether the measuring code (`lab/src`, `lab/pyproject.toml`) had uncommitted changes. The
AIG Mac is a shared desktop, not a dedicated benchmark host; that is stated on every record via
these fields and caps confidence at `medium` when memory pressure is not normal.

## Workloads 1.0.0

Each workload id is also the Planner's workload key. All external services are isolated: no
workload depends on anything outside the harness process except the runtime under test.

| id | tasks | success = | tool latency |
|---|---|---|---|
| `coding` | 3 fully specified Python functions (`slugify`, `merge_intervals`, `parse_duration`) | hidden unit tests pass, run in a `sandbox-exec` profile with **no network and writes confined to a temp dir**, 20 s limit | n/a |
| `research` | 2 questions over 3 short **fictional** sources each (one superseded figure) | all required facts present + ≥ 2 sources cited | n/a |
| `browser_tool` | 2 multi-hop browsing tasks via `fetch_page` / `save_finding` tools; max 8 model turns | saved finding and final answer contain the right values and no distractors; ≥ 2 fetches | each tool call; `fetch_page` is a real HTTP GET to a fixture site the harness serves on 127.0.0.1 (off-site URLs refused) |
| `rag` | 3 questions over an 11-document fictional KB with one distractor; deterministic BM25 top-3 | required values present, distractor absent; retrieval recall recorded | the retrieval step |
| `automation` | 3 fictional support emails → JSON record via the runtime's schema-constrained output | every field matches, incl. resolved relative dates | n/a |

Sampling is deterministic (`temperature 0`, fixed `seed`), but runtimes are not guaranteed to be
bit-reproducible across versions or hardware; the checker, not the text, defines success.

## What a number means — and does not

* A `peak_memory` of 5.35 GB for `qwen2.5:7b-instruct` Q4_K_M at context 8192 means: the serving
  process on this host, with this runtime version, held at least that much resident while
  running this workload. It is not a prediction for another runtime, context, or concurrency.
* `context` is the **allocated** window. The workloads' prompts are short (recorded as
  `prompt_tokens_max`); long-context *processing* time is not measured by 1.0.0 workloads.
* Concurrency is client concurrency. The runtime's own parallel slots are recorded
  (`server_parallel_slots`); Ollama 0.35 on this host loaded with 1 slot, so concurrent requests
  queue inside the runtime — the measured result reflects that, and the note says so.
