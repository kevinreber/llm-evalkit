"""The async batch runner: take a dataset and a task, produce one result per example.

The unit of work is a **task** — ``async (Example) -> output`` — not "an Anthropic
message". That is the load-bearing decision in this module, and it comes straight
from what real suites look like. Of the three this framework was extracted from, one
calls a domain function that owns its own prompt, model, temperature and vision
variant; one calls a pure ranking function with no model at all; and one calls
nothing, because the artifact under evaluation was already produced in production and
harvested afterwards. A runner whose body was ``client.messages.create(...)`` would
serve exactly one of those.

Injecting the task instead buys three things at once. Suites with no model get the
concurrency cap, the error isolation and the timing for free, with no API key and no
spend. Suites with a model keep their own prompt, where it belongs. And the
Anthropic-backed task becomes one implementation among several — see :mod:`llm_evalkit.tasks`.

Nothing here scores anything. Running and scoring are separate passes so a scoring
change can be re-applied to stored responses without paying for the model twice.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .dataset import Dataset, Example

_log = logging.getLogger(__name__)

__all__ = [
    "Metered",
    "Run",
    "RunResult",
    "Task",
    "Usage",
    "run",
]


@dataclass(frozen=True)
class Usage:
    """What one task invocation consumed.

    ``cost_usd`` is ``None`` for *unknown*, distinct from ``0.0`` for *free*. An
    unpriced model must not quietly contribute zero to a total — that reports a
    cheaper run than actually happened, which is the one direction a cost number
    should never be wrong in.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Metered:
    """Wrap a task's output to report usage alongside it.

    A task returns either a bare value or this. Mechanical tasks return bare values
    and are recorded with no usage at all, rather than being forced to fabricate a
    zeroed token count they never spent.
    """

    value: Any
    usage: Usage


class Task(Protocol):
    """Whatever is under evaluation. May call a model, or may not."""

    def __call__(self, example: Example) -> Awaitable[Any]: ...


@dataclass
class RunResult:
    """One example's outcome. A failure is a result, not a missing row."""

    example: Example
    output: Any = None
    error: BaseException | None = None
    latency_ms: float = 0.0
    usage: Usage | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def id(self) -> str:
        return self.example.id


@dataclass
class Run:
    """Everything one pass over a dataset produced."""

    dataset: str
    fingerprint: str
    stability: str
    results: list[RunResult] = field(default_factory=list)
    concurrency: int = 1
    label: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[RunResult]:
        return iter(self.results)

    @property
    def succeeded(self) -> list[RunResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[RunResult]:
        return [r for r in self.results if not r.ok]

    @property
    def wall_ms(self) -> float:
        if not (self.started_at and self.finished_at):
            return 0.0
        return (self.finished_at - self.started_at).total_seconds() * 1000.0

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.results if r.usage)

    @property
    def known_cost_usd(self) -> float:
        """Summed cost of the results that reported one.

        Read together with :attr:`unpriced_results`. A total of ``$0.41`` with four
        unpriced results is a lower bound, and the reporter is expected to say so.
        """
        return float(
            sum(r.usage.cost_usd for r in self.results if r.usage and r.usage.cost_usd is not None)
        )

    @property
    def unpriced_results(self) -> int:
        """Results that consumed tokens but could not be priced."""
        return sum(1 for r in self.results if r.usage and r.usage.cost_usd is None)

    @property
    def metered_results(self) -> int:
        return sum(1 for r in self.results if r.usage)


async def run(
    dataset: Dataset,
    task: Task,
    *,
    concurrency: int = 10,
    label: str | None = None,
    on_result: Callable[[RunResult], None] | None = None,
) -> Run:
    """Execute ``task`` over every example in ``dataset``, at most ``concurrency`` at once.

    One example's failure is captured onto its own result and never aborts the run.
    A 500-example run that dies on example 3 has spent real money and produced
    nothing; a run that records the exception and keeps going produces 499 usable
    results and names the one that broke.

    Results come back in dataset order regardless of completion order, so two runs
    over the same fingerprint line up row for row.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(example: Example) -> RunResult:
        async with semaphore:
            started = time.perf_counter()
            try:
                raw = await task(example)
            except Exception as exc:  # noqa: BLE001 - isolation is the whole point
                result = RunResult(
                    example=example,
                    error=exc,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            else:
                elapsed = (time.perf_counter() - started) * 1000.0
                if isinstance(raw, Metered):
                    result = RunResult(
                        example=example,
                        output=raw.value,
                        latency_ms=elapsed,
                        usage=raw.usage,
                    )
                else:
                    result = RunResult(example=example, output=raw, latency_ms=elapsed)

        if on_result is not None:
            # A broken progress callback is a reporting bug, and reporting is not the
            # run. Letting it propagate would replace a result that was already paid
            # for with a ValueError from the printer.
            try:
                on_result(result)
            except Exception:
                _log.warning("on_result callback failed for example %s", example.id, exc_info=True)
        return result

    started_at = datetime.now(UTC)
    examples = list(dataset)
    # return_exceptions is a backstop, not the mechanism: _one already catches
    # everything an example can raise. It covers the case where the wrapper itself
    # fails — an on_result callback that throws, say — so a bug in reporting cannot
    # discard results that were already paid for.
    gathered = await asyncio.gather(*(_one(ex) for ex in examples), return_exceptions=True)
    finished_at = datetime.now(UTC)

    results: list[RunResult] = []
    for example, item in zip(examples, gathered, strict=True):
        if isinstance(item, RunResult):
            results.append(item)
        elif isinstance(item, asyncio.CancelledError):
            # Cancellation voids the run; it is not a per-example failure. gather()
            # with return_exceptions=True hands CancelledError back like any other
            # exception, and recording it would turn a Ctrl-C at example 200 of 500
            # into a complete-looking Run with 300 "errors" — a number someone could
            # then store as a baseline.
            raise item
        else:
            results.append(RunResult(example=example, error=item))

    return Run(
        dataset=dataset.name,
        fingerprint=dataset.fingerprint,
        stability=dataset.stability,
        results=results,
        concurrency=concurrency,
        label=label,
        started_at=started_at,
        finished_at=finished_at,
    )
