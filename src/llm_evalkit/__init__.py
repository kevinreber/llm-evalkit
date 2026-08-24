"""llm-evalkit — evaluate LLM outputs at scale.

Week 1 ships the two halves that everything else hangs off: a dataset that knows
what it is (validated and fingerprinted) and a runner that executes a task over it
concurrently without letting one bad example take the run down.
"""

from __future__ import annotations

from .dataset import (
    DRAFT_SUFFIX,
    Dataset,
    DatasetError,
    Example,
    default_adapter,
    fingerprint,
    load_dir,
    load_jsonl,
)
from .evaluators.base import Component, Evaluator, Score, check
from .evaluators.embedding import embedding_similarity
from .evaluators.exact import exact_match, normalized_match, set_f1, set_overlap
from .evaluators.llm_judge import (
    PositionBias,
    grading_judge,
    instructor_caller,
    measure_position_bias,
)
from .evaluators.regex import must_not_match, pattern_match, patterns_from_expected
from .runner import Metered, Run, RunResult, Task, Usage, run
from .scorer import (
    Aggregate,
    Interval,
    ScoreCard,
    bootstrap_ci,
    repeat_variance,
    score_run,
    summarize,
)
from .tasks import ModelPricing, anthropic_task

__version__ = "0.1.0.dev0"

__all__ = [
    "DRAFT_SUFFIX",
    "Aggregate",
    "Component",
    "Dataset",
    "DatasetError",
    "Evaluator",
    "Example",
    "Interval",
    "Metered",
    "ModelPricing",
    "PositionBias",
    "Run",
    "RunResult",
    "Score",
    "ScoreCard",
    "Task",
    "Usage",
    "__version__",
    "anthropic_task",
    "bootstrap_ci",
    "check",
    "default_adapter",
    "embedding_similarity",
    "exact_match",
    "fingerprint",
    "grading_judge",
    "instructor_caller",
    "load_dir",
    "load_jsonl",
    "measure_position_bias",
    "must_not_match",
    "normalized_match",
    "pattern_match",
    "patterns_from_expected",
    "repeat_variance",
    "run",
    "score_run",
    "set_f1",
    "set_overlap",
    "summarize",
]
