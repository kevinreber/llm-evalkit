"""Exact and set-overlap matching — the trivial evaluators.

These exist partly to be useful and partly to prove the interface is genuinely
minimal: an evaluator is a plain function, and these are each a handful of lines
with no framework knowledge in them.

``set_f1`` is here rather than in ``embedding.py`` because it is what the real
suite actually uses. Navi's extraction eval scores lists of place names and
activities with graded set overlap, not embeddings, and it has been good enough
for two years of gold-set work. Reach for embeddings when this visibly fails.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

from ..dataset import Example
from .base import Component, Score

__all__ = ["canonical", "exact_match", "normalized_match", "set_f1", "set_overlap"]

# Word-level substitutions applied during canonicalisation, so trivial spelling
# variants of the same name do not read as misses.
_CANON_WORD = {"mount": "mt", "saint": "st", "street": "st"}


def canonical(value: Any) -> str:
    """Fold a value to a comparable form: accents stripped, punctuation dropped,
    whitespace collapsed, lowercased, common abbreviations unified.

    "Mt. Whitney" and "Mount Whitney" are the same place, and an evaluator that
    scores them as a miss is measuring typography rather than extraction.
    """
    folded = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()
    return " ".join(_CANON_WORD.get(word, word) for word in cleaned.split())


def exact_match(example: Example, output: Any) -> Score:
    """1.0 when ``output == example.expected``, byte for byte."""
    return Score.of(
        float(output == example.expected),
        name="exact_match",
        detail=None if output == example.expected else f"expected {example.expected!r}",
    )


def normalized_match(example: Example, output: Any) -> Score:
    """Exact match after :func:`canonical` folding."""
    got, want = canonical(output), canonical(example.expected)
    return Score.of(
        float(got == want),
        name="normalized_match",
        detail=None if got == want else f"expected {want!r}, got {got!r}",
    )


def set_f1(got: Iterable[Any], expected: Iterable[Any], *, min_contained_words: int = 2) -> float:
    """Graded overlap between two sets of names: ``2 * matches / (|got| + |expected|)``.

    Partial credit matters here. "Found the main location, missed a secondary one"
    is a materially different failure from "found the wrong place entirely", and
    collapsing both to 0.0 hides which one you are looking at.

    Beyond exact pairs, a pair also counts when one name's words **contain** the
    other's — "kiso valley tsumago to magome" matches "kiso valley", because
    verbose phrasing of the right place is a lesser error than the wrong place.
    The contained name needs at least ``min_contained_words`` words, so a bare city
    cannot be claimed by one of its landmarks: "tokyo tower" must not match "tokyo".

    Two empty sets score 1.0 — correctly declining to fill a field the source never
    stated is a success, and the metric that punishes it teaches the extractor to
    invent.
    """
    g = {canonical(v) for v in got}
    e = {canonical(v) for v in expected}
    if not g and not e:
        return 1.0
    if not g or not e:
        return 0.0

    matches = len(g & e)
    remaining = set(e - g)
    for candidate in sorted(g - e):
        candidate_words = set(candidate.split())
        for target in sorted(remaining):
            target_words = set(target.split())
            smaller = candidate_words if len(candidate_words) <= len(target_words) else target_words
            if len(smaller) >= min_contained_words and (
                candidate_words <= target_words or target_words <= candidate_words
            ):
                matches += 1
                remaining.discard(target)
                break
    return 2 * matches / (len(g) + len(e))


def set_overlap(*, fields: Sequence[str] | None = None, min_contained_words: int = 2) -> Any:
    """Build an evaluator scoring one or more list-valued fields by :func:`set_f1`.

    With ``fields``, both output and expected are treated as mappings and each
    named field becomes its own component — which is what makes a per-field
    scorecard possible rather than one opaque aggregate. Without it, output and
    expected are compared as flat iterables.
    """

    def evaluate(example: Example, output: Any) -> Score:
        expected = example.expected or {}
        if fields is None:
            value = set_f1(output or [], expected or [], min_contained_words=min_contained_words)
            return Score.of(value, name="set_overlap")

        components = []
        for field in fields:
            got = (output or {}).get(field) or []
            want = (expected or {}).get(field) or []
            value = set_f1(got, want, min_contained_words=min_contained_words)
            components.append(
                Component(
                    name=field,
                    value=value,
                    detail=None if value == 1.0 else f"expected {sorted(want)}, got {sorted(got)}",
                )
            )
        return Score.from_components(components)

    return evaluate
