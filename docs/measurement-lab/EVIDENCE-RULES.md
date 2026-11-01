# Evidence rules

The Planner's evidence classes are exactly: `aig_tested`, `primary_documented`, `calculated`,
`vendor_claim`, `unknown` (`lab/src/aiglab/evidence.py`; a test pins them to the live Base44
enum). The lab decides the class in **one** function, `classify_run`, from the sealed run on
disk. There is no flag, environment variable or CLI option that upgrades a class.

## A result is `aig_tested` only if **all** of these hold

1. `origin == "executed"` — produced by the harness driving a real runtime, not a fixture,
   simulation, estimate or import.
2. The adapter id is in `EXECUTING_ADAPTERS` (today: `ollama`) **and** the adapter class declares
   `executes_real_runtime = True`.
3. The run directory's `SHA256SUMS` seal matches every raw file on disk, and `aggregate.json`
   was computed from that same sealed data.
4. Harness status is `completed` (an aborted run is never evidence).
5. The methodology version is supported.
6. At least one measured repetition executed, and every request inside it has origin `executed`.
7. Runtime identity checks exist and **every one passed** (artifact hash, served artifact,
   effective context, residency, sampling target, no reload, no contention — see METHODOLOGY).

Any failure ⇒ `unknown`, with every reason attached to the record. A measured *failure* (a task
that fails, a request that errors, an OOM) is still `aig_tested` evidence of that failure: the
class says the measurement is genuine, not that the outcome was good.

## Never `aig_tested`

| data | class | why |
|---|---|---|
| fixture adapter output (tests, pipeline dry-runs) | `unknown`, never synced | `origin=fixture`, adapter not executing |
| simulated / synthetic numbers | `unknown`, never synced | `origin=simulated` |
| planning heuristics, VRAM calculator, sizing engine | `calculated` | `origin=estimated` |
| vendor or model-card numbers | `vendor_claim` / `primary_documented` | `origin=imported`; an import claiming `aig_tested` is downgraded to `unknown` |
| a real run with a failed identity check (e.g. another model was loaded, context mismatch) | `unknown` | integrity failure; excluded from Base44 sync unless `--include-unknown` |
| a real run edited after sealing | `unknown` | seal mismatch |
| a real adapter pointed at a fake/mock API | `unknown` | no serving process loads the verified artifact |
| results from runtimes listed as "not implemented" | n/a | they cannot run; dry-run blocks them |

Regression tests: `lab/tests/test_evidence.py` (fixture through the full pipeline, forged and
re-sealed runs, tampering), `test_adapters_runner.py::test_mock_runtime_never_yields_aig_tested`,
and the site-side guard `src/data/measurement-evidence.test.ts` (runs in `npm test`, which Estate
runs as AIG's registered `test` check).

## Where evidence may live

* `evidence/measurements/runs/` — only sealed runs with `origin=executed`.
  `aiglab check-evidence` fails (exit 2) on anything else.
* Fixtures and synthetic runs exist only in pytest temp dirs.
* Blocked attempts are recorded in `evidence/measurements/blocked/` (they contain no measurements).

## Limits of the guarantee

The seal and identity checks make accidental contamination and casual editing detectable. They
do not make deliberate forgery impossible: someone with write access could fabricate a complete
run directory, including identity checks, and re-seal it. That is why runs land in the repo by
reviewed commit (or via `aiglab import-runs`, which re-verifies seals), and why every run
records the harness commit, runtime version and artifact digest needed to reproduce it.
