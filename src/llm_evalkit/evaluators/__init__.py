"""Evaluators: one callable each, ``(example, output) -> Score``.

The concrete ones (exact, regex, LLM judge, embedding) build on :mod:`.base`,
which holds the protocol and the ``Score`` type everything else agrees on.
"""

from __future__ import annotations

from .base import Aggregation, Component, Evaluator, Score, check, components_by_name

__all__ = [
    "Aggregation",
    "Component",
    "Evaluator",
    "Score",
    "check",
    "components_by_name",
]
