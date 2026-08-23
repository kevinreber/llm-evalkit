# llm-evalkit

A Python library for evaluating LLM outputs at scale. Loads and fingerprints a dataset, runs whatever is under evaluation across it concurrently, scores the results with pluggable evaluators, and reports a mean with a bootstrap confidence interval around it — so "73%" comes with the range that says whether it differs from last week's 71%.

It is a library, not a service. `pip install`, import it, run it against your own code. Nothing to deploy, nothing in anyone's request path.

## Status

**Week 1 of 4 — `0.1.0.dev0`, not yet released.** Datasets and the runner work and are tested. Evaluators, scoring, storage, regression detection, W&B and the HTML report are the next three phases and are not built yet; the `evalkit` command currently exposes only `datasets`.

## Where this came from

It was extracted, not invented. [Navi](https://github.com/kevinreber/navi) — a planning-orchestrator agent — had grown three hand-rolled eval runners totalling 939 lines, and each one re-implemented the same loop: load cases, call the thing under test, score, print a scorecard. The generic half of that loop is this package. The domain half stays in Navi as evaluator callables.

That origin is why the interfaces look the way they do. The three suites disagree about almost everything an eval framework normally assumes, and the interface had to bend to fit all three rather than the other way round:

| | `extraction` | `suggestions` | `notes` |
|---|---|---|---|
| what it calls | a domain function that owns its own prompt and model | a pure ranking function | nothing — the artifact was harvested from production |
| what comes back | a Pydantic model | an ordered list of ids | already in the input file |
| what a score is | six field scores plus an invention **count** | a list of labelled pass/fail **checks** | several rates, each of which can be **not applicable** |
| labels | hand-corrected gold | relational assertions (`A outranks B`) | none at all |
| cost per run | real API spend | free | free |

A framework built around `(prompt) -> str` and `expected == actual` serves one of those three.

## How it works

```mermaid
flowchart LR
    subgraph LOAD [dataset.py]
        F["cases/*.json<br/>or cases.jsonl"] --> AD[adapter]
        AD --> EX["Example<br/>id / input / expected / why"]
        EX --> FP["SHA-256 fingerprint<br/>+ attachment bytes"]
    end

    subgraph RUN [runner.py]
        EX --> SEM["asyncio.Semaphore(10)"]
        SEM --> T[["Task<br/>async (Example) -> output"]]
        T --> RR["RunResult<br/>output / latency / usage / error"]
    end

    T -.-> A["anthropic_task<br/>(tasks.py)"]
    T -.-> D[your own domain function]
    T -.-> N[no model at all]

    RR --> SC["evaluators + scorer<br/>(week 2)"]
    SC --> ST["SQLite store<br/>+ regression detection<br/>(week 3)"]
    ST --> REP["W&B, HTML report, CLI<br/>(week 4)"]

    style SC stroke-dasharray: 4 4
    style ST stroke-dasharray: 4 4
    style REP stroke-dasharray: 4 4
```

## Quickstart

```python
import asyncio
from anthropic import AsyncAnthropic
from llm_evalkit import ModelPricing, anthropic_task, load_jsonl, run

dataset = load_jsonl("datasets/qa.jsonl")
task = anthropic_task(
    AsyncAnthropic(),
    model="claude-haiku-4-5-20251001",
    system="Answer in one word.",
    pricing={"claude-haiku-4-5-20251001": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0)},
)

result = asyncio.run(run(dataset, task, concurrency=10))
print(f"{len(result.succeeded)}/{len(result)} ok, ${result.known_cost_usd:.4f}")
```

Evaluating something that isn't a model call is the same code with a different task:

```python
async def rank(example):
    return my_ranker(example.input["references"], example.input["window"])

result = asyncio.run(run(load_dir("evals/suggestions/cases"), rank))
```

Inspect a dataset from the shell:

```
evalkit datasets evals/extraction/cases
```

## Design decisions worth knowing

- **The unit of work is a task, not a model call.** `async (Example) -> output`. The runner never touches an SDK. This is what lets a suite with no model at all — a deterministic ranker, or a scorecard over artifacts harvested from production — get the concurrency cap, the failure isolation and the timing for free, with no API key and no spend. `anthropic_task` is one implementation of the interface, in its own module, not the runner's body.

- **`None` is not zero, in either direction.** A plan that recorded no notes has no orphan-note rate; averaging it in as `0.0` would report a *better* score for a planner that did nothing. Likewise an unpriced model's cost is recorded as unknown rather than free, and a run reports `known_cost_usd` alongside `unpriced_results` so the total reads as the floor it is. Costs should only ever be wrong in the expensive direction.

- **The fingerprint hashes attachment contents, not attachment paths.** An extraction case points at a screenshot beside it on disk. Hashing only the JSON leaves the fingerprint unchanged when the image behind it is swapped — precisely the silent drift a fingerprint exists to catch. It also excludes absolute paths, so two machines agree.

- **Datasets declare whether they are expected to hold still.** A `golden` set changes only when a human edits it, so a fingerprint mismatch means two runs measured different things and the comparison should be refused. A `harvested` set is re-sampled from production every time it is collected, so its fingerprint always differs — refusing that comparison would make an organic suite permanently incomparable to itself. Week 3's regression detection reads this flag.

- **Drafts do not score.** A case file ending `.draft.json` is excluded by default. Bootstrapping a gold set by pre-filling the model's own output as the expected labels is roughly ten times faster than authoring JSON by hand — and rubber-stamping it blesses today's bugs as correct. The rename is the human's signature.

- **Running and scoring are separate passes.** Results carry raw outputs, so a scoring change can be re-applied without paying the model twice.

- **One bad example cannot abort a run.** A 500-example run that dies on example 3 has spent real money and produced nothing. Failures are captured onto their own result, results come back in dataset order regardless of completion order, and a broken progress callback is logged rather than allowed to replace a result that was already paid for.

- **The console script is `evalkit`, not `eval`.** `eval` is a builtin in bash and zsh; the shell would swallow the command before it reached the package.

- **`temperature` defaults to 0.** An eval that samples at the provider default measures the model *and* the sampler, then attributes the resulting wobble to whatever changed between runs.

## Roadmap

| Phase | Contents | State |
|---|---|---|
| 1 | `dataset.py`, `runner.py`, `tasks.py` | done |
| 1 (cont.) | retry with backoff on 429/5xx, pre-flight cost estimation with confirm-on-threshold | next |
| 2 | evaluator protocol, exact / regex / LLM-judge / embedding, **bootstrap confidence intervals**, judge position-bias measurement | |
| 3 | `Store` protocol + SQLite, regression detection, refuse cross-fingerprint comparison | |
| 4 | W&B runs, `run` / `compare` / `report` / `datasets` CLI, Jinja2 HTML report, PyPI `0.1.0` | |

The non-negotiable one is bootstrap CI. Without it an eval score is noise dressed as signal: 73% against 71% on 100 examples is meaningless when the 95% interval is `[68%, 80%]` for both.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check src tests
```

The test suite runs entirely against a fake client. No API key, no spend, no network — which is what makes it repeatable and what lets it run in CI, unlike the suites this package was extracted from.

## License

MIT
