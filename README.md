# llm-evalkit

A Python library for evaluating LLM outputs at scale. Loads and fingerprints a dataset, runs whatever is under evaluation across it concurrently, scores the results with pluggable evaluators, and reports a mean with a bootstrap confidence interval around it — so "73%" comes with the range that says whether it differs from last week's 71%.

It is a library, not a service. `pip install`, import it, run it against your own code. Nothing to deploy, nothing in anyone's request path.

## Status

**Bootstrap scaffold.** Nothing is implemented yet. This commit exists to get packaging, linting, tests and CI green before any behaviour lands.

## Where this is going

It is being extracted, not invented. [Navi](https://github.com/kevinreber/navi) — a planning-orchestrator agent — has three hand-rolled eval runners totalling 939 lines, and each re-implements the same loop: load cases, call the thing under test, score, print a scorecard. The generic half of that loop becomes this package; the domain half stays in Navi as evaluator callables.

| Phase | Contents |
|---|---|
| 1 | `dataset.py` (JSONL + case-directory loading, SHA-256 fingerprint), `runner.py` (async batch execution), retry, pre-flight cost estimation |
| 2 | evaluator protocol, exact / regex / LLM-judge / embedding, **bootstrap confidence intervals**, judge position-bias measurement |
| 3 | `Store` protocol + SQLite, regression detection, refuse cross-fingerprint comparison |
| 4 | W&B runs, `run` / `compare` / `report` / `datasets` CLI, Jinja2 HTML report, PyPI `0.1.0` |

The non-negotiable one is bootstrap CI. Without it an eval score is noise dressed as signal: 73% against 71% on 100 examples is meaningless when the 95% interval is `[68%, 80%]` for both.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check src tests
```

## License

MIT
