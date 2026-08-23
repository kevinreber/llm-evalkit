"""The evaluator protocol and the ``Score`` type.

An evaluator is one callable: ``(example, output) -> Score``. That is the whole
interface. No base class to subclass, no registration, no framework knowledge —
anyone can add one with a plain function.

``Score`` is where the real design work is, because the three suites this package
was extracted from disagree about what a score even is:

* one produces six named field scores **plus a count** of "fields invented that
  should have been empty" — and the count is the headline metric, reported as
  ``3 inventions``, never as a per-case average;
* one produces a list of labelled pass/fail **checks** asserting relative ordering,
  several sharing a name within a single case (``ranked_above`` twice), aggregated
  both overall and grouped by name;
* one produces several rates, each of which can be genuinely **not applicable** —
  a plan that recorded no notes has no orphan-note rate, and averaging that in as
  ``0.0`` would report a *better* score for a planner that did nothing.

So a ``Score`` carries a **list** of components, not a dict: names repeat. Each
component declares how it rolls up, because mean-versus-sum is a real distinction
that must be stated rather than guessed. And ``value=None`` means not-applicable
and is skipped in aggregation rather than counted as zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..dataset import Example

__all__ = [
    "Aggregation",
    "Component",
    "Evaluator",
    "Score",
    "check",
    "components_by_name",
]

# How a component combines across examples.
#
# ``mean``  average of the values that are not None. The default: accuracies,
#           rates, pass fractions.
# ``sum``   add the values. For counts, where the headline is "3 inventions" and
#           "0.12 inventions per case" would be a different, less useful claim.
Aggregation = Literal["mean", "sum"]


@dataclass(frozen=True)
class Component:
    """One named part of a score.

    ``value=None`` means *not applicable to this example*, which is distinct from
    ``0.0``. Aggregation skips it and reports the reduced denominator, so a metric
    quietly applying to fewer and fewer examples stays visible.
    """

    name: str
    value: float | None
    aggregation: Aggregation = "mean"
    detail: str | None = None

    @property
    def applicable(self) -> bool:
        return self.value is not None


def check(name: str, ok: bool, detail: str | None = None) -> Component:
    """A pass/fail assertion as a component. Sugar — a check is a 0/1 mean."""
    return Component(name=name, value=1.0 if ok else 0.0, aggregation="mean", detail=detail)


@dataclass(frozen=True)
class Score:
    """What an evaluator returns for one example."""

    components: tuple[Component, ...] = ()

    # Free-form payload for the report: expected-versus-got pairs, the ids that
    # ranked wrong, a judge's reasoning. Never scored — it is what makes a red
    # result actionable instead of merely red.
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, value: float | None, *, name: str = "score", detail: str | None = None) -> Score:
        """The common case: one unnamed number."""
        return cls(components=(Component(name=name, value=value, detail=detail),))

    @classmethod
    def from_components(
        cls, components: Iterable[Component], *, detail: dict[str, Any] | None = None
    ) -> Score:
        return cls(components=tuple(components), detail=dict(detail or {}))

    @property
    def value(self) -> float | None:
        """The headline number: the mean of every applicable ``mean`` component.

        ``sum`` components are excluded on purpose. An invention count is not on
        the same scale as an accuracy and averaging them together would produce a
        number that means nothing.
        """
        values = [c.value for c in self.components if c.applicable and c.aggregation == "mean"]
        if not values:
            return None
        return sum(values) / len(values)  # type: ignore[arg-type]

    @property
    def passed(self) -> bool | None:
        """All-or-nothing, for suites used as a gate rather than a scorecard.

        ``None`` when nothing applicable was measured — which must not read as a
        pass, or an evaluator that silently stopped applying would turn green.
        """
        applicable = [c for c in self.components if c.applicable and c.aggregation == "mean"]
        if not applicable:
            return None
        return all(c.value == 1.0 for c in applicable)

    def named(self, name: str) -> list[Component]:
        return [c for c in self.components if c.name == name]


class Evaluator(Protocol):
    """Score one example's output. The entire extension interface."""

    def __call__(self, example: Example, output: Any) -> Score: ...


# Non-async on purpose. Scoring is almost always pure local computation, and
# forcing every trivial evaluator to be a coroutine would be interface tax. An
# evaluator that genuinely needs a model — an LLM judge — wraps its own event loop
# or is run through the async runner as a task instead.
EvaluatorFn = Callable[[Example, Any], Score]


def components_by_name(scores: Sequence[Score]) -> dict[str, list[Component]]:
    """Group every component across a run by name, preserving repeats.

    A dict of name to single value cannot represent a suite whose case asserts
    ``ranked_above`` twice, which is why components are a list to begin with.
    """
    grouped: dict[str, list[Component]] = {}
    for score in scores:
        for component in score.components:
            grouped.setdefault(component.name, []).append(component)
    return grouped
