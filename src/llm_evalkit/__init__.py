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
from .runner import Metered, Run, RunResult, Task, Usage, run
from .tasks import ModelPricing, anthropic_task

__version__ = "0.1.0.dev0"

__all__ = [
    "DRAFT_SUFFIX",
    "Dataset",
    "DatasetError",
    "Example",
    "Metered",
    "ModelPricing",
    "Run",
    "RunResult",
    "Task",
    "Usage",
    "__version__",
    "anthropic_task",
    "default_adapter",
    "fingerprint",
    "load_dir",
    "load_jsonl",
    "run",
]
