"""Evaluators: one callable each, ``(example, output) -> Score``.

:mod:`.base` holds the protocol and the ``Score`` type everything agrees on. The
rest are ordinary functions — ``exact`` and ``regex`` are a few lines each, which is
the point: adding an evaluator requires no framework knowledge.

:mod:`.llm_judge` is the exception in one respect: it is async, because judging is
network I/O. Run it through :func:`llm_evalkit.scorer.score_run`, which takes sync
and async evaluators in the same list.
"""

from __future__ import annotations

from .base import Aggregation, Component, Evaluator, Score, check, components_by_name
from .embedding import Embedder, cosine, embedding_similarity
from .exact import canonical, exact_match, normalized_match, set_f1, set_overlap
from .llm_judge import (
    JudgeScore,
    PairwiseVerdict,
    PositionBias,
    grading_judge,
    instructor_caller,
    judge_pairwise,
    measure_position_bias,
)
from .regex import must_not_match, pattern_match, patterns_from_expected

__all__ = [
    "Aggregation",
    "Component",
    "Embedder",
    "Evaluator",
    "JudgeScore",
    "PairwiseVerdict",
    "PositionBias",
    "Score",
    "canonical",
    "check",
    "components_by_name",
    "cosine",
    "embedding_similarity",
    "exact_match",
    "grading_judge",
    "instructor_caller",
    "judge_pairwise",
    "measure_position_bias",
    "must_not_match",
    "normalized_match",
    "pattern_match",
    "patterns_from_expected",
    "set_f1",
    "set_overlap",
]
