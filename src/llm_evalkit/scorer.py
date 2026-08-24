"""Aggregation and bootstrap confidence intervals.

This module is why the package exists. Without an interval, an eval score is noise
dressed as signal: 73% against 71% on 100 examples is meaningless when the 95%
interval is [68%, 80%] for both, and every "the prompt change helped" conclusion
drawn from that pair is unfounded.

The interval is a **percentile bootstrap**: resample the per-example scores with
replacement 1000 times, take the mean of each resample, and report the 2.5th and
97.5th percentiles of those means. It is used instead of a normal approximation
because eval scores are small-n and rarely normal — accuracies pile up near 1.0,
and a symmetric interval around a mean of 0.95 happily predicts scores above 1.0.

**What it measures, and what it does not.** The bootstrap resamples *examples*, so
it answers "how much would this number move if I had drawn a different sample of
cases from the same population?" It says nothing about run-to-run variance from
sampling a model at temperature > 0 — that needs repeated runs, and
:func:`repeat_variance` is where that number comes from. Reporting a bootstrap
interval as though it captured model nondeterminism would overstate the honesty
this module exists to provide. Two of the three suites this was extracted from are
fully deterministic, where the bootstrap is *only* the dataset-sampling question.
"""

from __future__ import annotations

import asyncio
import inspect
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluators.base import Aggregation, Component, Score, components_by_name
from .runner import Run

__all__ = [
    "Aggregate",
    "Interval",
    "ScoreCard",
    "bootstrap_ci",
    "repeat_variance",
    "score_run",
    "summarize",
]

DEFAULT_RESAMPLES = 1000
DEFAULT_CI = 0.95
# Seeded so a scorecard reproduces. An interval that shifts when you re-print it
# invites arguing with the number instead of with the result.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    confidence: float = DEFAULT_CI

    @property
    def width(self) -> float:
        return self.high - self.low

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap.

        Non-overlap is sufficient to call a difference real; overlap is *not*
        sufficient to call it absent, which is a common misreading. Two overlapping
        intervals can still come from significantly different means.
        """
        return self.low <= other.high and other.low <= self.high

    def __str__(self) -> str:
        return f"[{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    scores: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> Interval | None:
    """Percentile bootstrap interval for the mean of ``scores``.

    Returns ``None`` for an empty input — there is no interval for no data, and a
    caller must not be handed ``[0.0, 0.0]`` to print as though there were.

    A single observation yields a degenerate interval at that value. That is the
    honest answer: one example tells you nothing about spread, and reporting
    ``[x, x]`` at least makes the absence visible rather than implying precision.
    """
    if not scores:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    observations = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    # One vectorised draw of (n_resamples x n) indices beats a Python loop by a
    # wide margin, and a 1000x resample of a 500-example run is otherwise slow
    # enough that people turn it down — which defeats the point.
    picks = rng.integers(0, observations.size, size=(n_resamples, observations.size))
    means = observations[picks].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(low=float(low), high=float(high), confidence=confidence)


def repeat_variance(run_means: Sequence[float]) -> Interval | None:
    """Spread across REPEATED runs of the same dataset — model nondeterminism.

    Distinct from :func:`bootstrap_ci`, which resamples examples within one run.
    Only repeated runs can measure how much a sampled model wobbles, and conflating
    the two is the single easiest way for an eval report to mislead. Needs at least
    two runs; returns ``None`` below that rather than inventing a spread.
    """
    if len(run_means) < 2:
        return None
    mean = statistics.fmean(run_means)
    # Half-width of one standard deviation either side — deliberately descriptive
    # rather than inferential, since a handful of runs cannot support more.
    spread = statistics.stdev(run_means)
    return Interval(low=mean - spread, high=mean + spread, confidence=0.0)


@dataclass(frozen=True)
class Aggregate:
    """One component name, rolled up across every example that had it."""

    name: str
    aggregation: Aggregation
    value: float
    n: int
    n_total: int
    interval: Interval | None = None

    @property
    def applicability(self) -> float:
        """Share of examples this component actually applied to.

        Worth printing. A metric at 100% that only ever applied to 2 of 40 examples
        is not a result, and the ranking suite's port to a hosted platform showed
        exactly this failure: three metrics computed from a single case each.
        """
        return self.n / self.n_total if self.n_total else 0.0

    def __str__(self) -> str:
        body = f"{self.value:.3f}" if self.aggregation == "mean" else f"{self.value:g}"
        ci = f" {self.interval}" if self.interval else ""
        na = "" if self.n == self.n_total else f"  (n={self.n}/{self.n_total})"
        return f"{self.name}: {body}{ci}{na}"


@dataclass(frozen=True)
class ScoreCard:
    """Everything one run's scores add up to."""

    n_examples: int
    components: tuple[Aggregate, ...]
    overall: float | None
    interval: Interval | None
    passed: int
    gateable: int

    def component(self, name: str) -> Aggregate | None:
        return next((c for c in self.components if c.name == name), None)

    @property
    def all_passed(self) -> bool:
        """True only when something was gateable and all of it passed.

        A run with nothing gateable is not a pass. An evaluator that stopped
        producing checks would otherwise turn the suite green by going silent.
        """
        return self.gateable > 0 and self.passed == self.gateable


async def score_run(
    run: Run,
    evaluators: Sequence[Callable[..., Any]],
    *,
    concurrency: int = 8,
    include_failed: bool = True,
) -> list[Score]:
    """Apply every evaluator to every result, merging components per example.

    Accepts sync and async evaluators in the same list. Most scoring is pure local
    computation and should not be forced into a coroutine, but a judge is network
    I/O and would serialise a large run into an afternoon if it were not concurrent.
    Detecting the difference here keeps that off the caller.

    A failed result still produces a Score, carrying an ``error`` component that is
    **not applicable** rather than zero. Scoring an API timeout as a wrong answer
    silently converts an infrastructure problem into a quality regression — which is
    exactly the class of lie this package exists to refuse. Pass
    ``include_failed=False`` to drop them instead, but know that a shrinking
    denominator is then the only trace left.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def score_one(result: Any) -> Score:
        if not result.ok:
            return Score.from_components(
                [Component("error", None, detail=repr(result.error))],
                detail={"error": repr(result.error)},
            )
        components: list[Component] = []
        detail: dict[str, Any] = {}
        for evaluator in evaluators:
            outcome = evaluator(result.example, result.output)
            if inspect.isawaitable(outcome):
                async with semaphore:
                    outcome = await outcome
            components.extend(outcome.components)
            detail.update(outcome.detail)
        return Score.from_components(components, detail=detail)

    results = [r for r in run.results if include_failed or r.ok]
    return list(await asyncio.gather(*(score_one(r) for r in results)))


def summarize(
    scores: Sequence[Score],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> ScoreCard:
    """Roll per-example scores into a scorecard with intervals.

    Every component is grouped by name and reduced by its own declared aggregation,
    so counts sum and rates average. Not-applicable components are skipped and the
    surviving denominator is recorded, never silently treated as zero.
    """
    grouped = components_by_name(scores)
    n_examples = len(scores)

    aggregates: list[Aggregate] = []
    for name, items in grouped.items():
        applicable: list[Component] = [c for c in items if c.applicable]
        aggregation = items[0].aggregation
        if not applicable:
            # Present but never applicable. Recorded at n=0 rather than dropped, so
            # a metric that has stopped applying is visible instead of absent.
            aggregates.append(
                Aggregate(name=name, aggregation=aggregation, value=0.0, n=0, n_total=len(items))
            )
            continue
        values = [c.value for c in applicable]  # type: ignore[misc]
        if aggregation == "sum":
            rolled = float(sum(values))
            interval = None  # a total has no sampling interval worth printing
        else:
            rolled = float(sum(values) / len(values))
            interval = bootstrap_ci(
                values, n_resamples=n_resamples, confidence=confidence, seed=seed
            )
        aggregates.append(
            Aggregate(
                name=name,
                aggregation=aggregation,
                value=rolled,
                n=len(applicable),
                n_total=len(items),
                interval=interval,
            )
        )

    per_example = [s.value for s in scores if s.value is not None]
    overall = float(sum(per_example) / len(per_example)) if per_example else None
    overall_interval = bootstrap_ci(
        per_example, n_resamples=n_resamples, confidence=confidence, seed=seed
    )

    verdicts = [s.passed for s in scores if s.passed is not None]
    return ScoreCard(
        n_examples=n_examples,
        components=tuple(sorted(aggregates, key=lambda a: a.name)),
        overall=overall,
        interval=overall_interval,
        passed=sum(1 for v in verdicts if v),
        gateable=len(verdicts),
    )
