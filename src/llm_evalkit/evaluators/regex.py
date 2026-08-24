"""Pattern-matching evaluators.

Useful when the answer is embedded in prose rather than being the whole response:
"does it name the right city anywhere", "does it emit a well-formed date", "does it
avoid the phrase we told it never to use".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..dataset import Example
from .base import Component, Score

__all__ = ["must_not_match", "pattern_match", "patterns_from_expected"]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def pattern_match(pattern: str | re.Pattern[str], *, name: str = "pattern_match") -> Any:
    """Build an evaluator scoring 1.0 when ``pattern`` is found in the output."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def evaluate(example: Example, output: Any) -> Score:
        found = compiled.search(_text(output)) is not None
        return Score.of(
            float(found),
            name=name,
            detail=None if found else f"no match for {compiled.pattern!r}",
        )

    return evaluate


def must_not_match(pattern: str | re.Pattern[str], *, name: str = "must_not_match") -> Any:
    """The inverse: 1.0 when the pattern is **absent**.

    Scored as a rate rather than a count so it composes with everything else, but
    note that the interesting number for a safety-style check is usually "how many
    examples tripped it", which is what the ``detail`` payload carries.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def evaluate(example: Example, output: Any) -> Score:
        hit = compiled.search(_text(output))
        return Score.of(
            float(hit is None),
            name=name,
            detail=None if hit is None else f"matched {hit.group(0)!r}",
        )

    return evaluate


def patterns_from_expected(*, name: str = "expected_patterns", flags: int = re.IGNORECASE) -> Any:
    """Treat each example's ``expected`` as the pattern (or list of patterns) to find.

    Each pattern becomes its own component, so a run reports which specific
    expectation is failing rather than one blended number. An example with no
    patterns scores ``None`` — not applicable, not zero — because a case that
    asserts nothing has not been passed, it has been skipped.
    """

    def evaluate(example: Example, output: Any) -> Score:
        expected = example.expected
        patterns: Iterable[Any]
        if expected is None:
            patterns = []
        elif isinstance(expected, (str, re.Pattern)):
            patterns = [expected]
        else:
            patterns = expected

        text = _text(output)
        components: list[Component] = []
        for pattern in patterns:
            compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
            found = compiled.search(text) is not None
            components.append(
                Component(
                    name=name,
                    value=float(found),
                    detail=None if found else f"missing {compiled.pattern!r}",
                )
            )
        if not components:
            components.append(Component(name=name, value=None, detail="no patterns asserted"))
        return Score.from_components(components)

    return evaluate
