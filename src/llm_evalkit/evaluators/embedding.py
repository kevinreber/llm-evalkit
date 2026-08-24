"""Cosine-similarity soft matching.

No embedding provider ships with this package, and that is deliberate. Anthropic has
no embeddings endpoint, so shipping one would mean taking a hard dependency on a
different vendor for a feature most suites never reach for — Navi's extraction eval
has scored place names with set overlap for two years and never needed vectors.

So the embedder is injected: ``(list[str]) -> list[list[float]]``. Bring Voyage,
OpenAI, sentence-transformers, or a cache in front of any of them.

Reach for this only when :func:`llm_evalkit.evaluators.exact.set_f1` visibly fails —
when correct answers are being scored as misses because they are *paraphrases*
rather than spelling variants. Embedding similarity is a fuzzier instrument than it
looks: 0.85 cosine is not a threshold with meaning, it is a number you have to
calibrate against your own labelled data before it is worth anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..dataset import Example
from .base import Score

__all__ = ["Embedder", "cosine", "embedding_similarity"]

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to 0.0-1.0.

    Negative similarity is clamped rather than preserved: a score below zero has no
    meaning in an aggregate that averages with accuracies, and letting one wildly
    opposed pair drag a mean below zero produces a scorecard nobody can read.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        # A zero vector has no direction, so similarity is undefined rather than 0.
        raise ValueError("cannot compute cosine similarity against a zero vector")
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def embedding_similarity(
    embed: Embedder,
    *,
    name: str = "embedding_similarity",
    threshold: float | None = None,
) -> Any:
    """Build an evaluator scoring cosine similarity between output and expected.

    With ``threshold`` set, the score becomes a 0/1 pass instead of a continuous
    similarity — and the raw similarity is kept in ``detail`` either way, because a
    thresholded score that discards the underlying number makes it impossible to
    tell a near miss from a total miss when you later want to move the threshold.

    Batches both strings into one ``embed`` call per example. An example with no
    expected value scores ``None``: there is nothing to be similar to, and that is
    not-applicable rather than a failure.
    """

    def evaluate(example: Example, output: Any) -> Score:
        if example.expected is None:
            return Score.of(None, name=name, detail="no expected value to compare against")

        vectors = embed([str(output), str(example.expected)])
        if len(vectors) != 2:
            raise ValueError(f"embedder returned {len(vectors)} vectors for 2 inputs")
        similarity = cosine(vectors[0], vectors[1])

        if threshold is None:
            return Score.of(similarity, name=name, detail=f"cosine {similarity:.3f}")
        passed = similarity >= threshold
        return Score.of(
            float(passed),
            name=name,
            detail=f"cosine {similarity:.3f} {'>=' if passed else '<'} {threshold}",
        )

    return evaluate
