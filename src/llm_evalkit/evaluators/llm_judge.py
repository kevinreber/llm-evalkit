"""LLM-as-judge, and the machinery to measure how much you should trust it.

A judge is the most error-prone component in any eval system, and the failure is
rarely a crash — it is a confident, well-argued, wrong score. So this module ships
the judge and the instrument for auditing it in the same file, because a judge
whose bias you have never measured is a number you cannot defend.

**Position bias** is the one to measure first. Ask a model to pick the better of two
responses and it will systematically favour one slot regardless of content. The test
is trivial and nobody runs it: present each pair as (A, B), then again as (B, A), and
count how often the verdict flips. A judge with no position bias flips only on pairs
it considers genuinely tied. :func:`measure_position_bias` reports that rate, and
:func:`judge_pairwise` averages both orders so a single comparison is not at the
mercy of it.

Judging is **async**, unlike the other evaluators here, because it is network I/O and
pretending otherwise would serialise a 500-example run into an afternoon. Use
:func:`llm_evalkit.scorer.score_run`, which handles sync and async evaluators alike.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..dataset import Example
from .base import Component, Score

__all__ = [
    "JudgeScore",
    "PairwiseVerdict",
    "PositionBias",
    "grading_judge",
    "judge_pairwise",
    "measure_position_bias",
    "measure_self_consistency",
]

# Judging needs a stronger model than the one under test; a judge that is weaker
# than the thing it grades produces scores that measure the judge.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


class JudgeScore(BaseModel):
    """A single response graded against a rubric."""

    score: int = Field(ge=1, le=5, description="1 = unusable, 3 = acceptable, 5 = excellent")
    reason: str = Field(description="One or two sentences. Cite the specific flaw or strength.")
    is_faithful: bool = Field(
        description="True only if every factual claim is supported by the provided context."
    )

    @property
    def normalized(self) -> float:
        """Map 1-5 onto 0.0-1.0 so it aggregates with every other evaluator."""
        return (self.score - 1) / 4.0


class PairwiseVerdict(BaseModel):
    """Which of two responses is better."""

    winner: Literal["A", "B", "tie"]
    reason: str = Field(description="One or two sentences naming the deciding difference.")


class StructuredCaller(Protocol):
    """Anything that can turn a prompt into a validated Pydantic model.

    Injected rather than hardcoded so the judge is testable without a network, and
    so swapping the structured-output layer is a one-line change.
    """

    async def __call__(self, *, system: str, prompt: str, model_cls: type[BaseModel]) -> Any: ...


def instructor_caller(
    client: Any, *, model: str = DEFAULT_JUDGE_MODEL, max_tokens: int = 512
) -> StructuredCaller:
    """A :class:`StructuredCaller` backed by Instructor over the Anthropic SDK.

    Instructor guarantees a Pydantic instance back and retries malformed JSON,
    which is the cheapest available fix for the judge's most common failure.

    **No temperature is sent, and none can be.** The obvious way to make a judge
    reproducible is ``temperature=0``, and that is no longer available: anthropic
    1.0.0 removed the argument from the typed signature of both ``create()`` and
    ``parse()``, and the API itself answers ``400 — `temperature` is deprecated for
    this model`` for newer models including claude-sonnet-5. Routing it through
    ``extra_body`` reaches the wire and gets the same 400. So judge determinism is
    whatever the model's default decoding gives you, which is a real limitation of
    every LLM-judge eval on current models rather than something this package can
    paper over. Measure it with repeated runs rather than assuming it away.

    Worth knowing: the Anthropic SDK now ships native structured output via
    ``messages.parse(output_format=...)``, which would remove this dependency
    entirely. Left as-is because the sprint locked Instructor as a decision; the
    ``StructuredCaller`` seam is what makes changing it later trivial.
    """

    async def call(*, system: str, prompt: str, model_cls: type[BaseModel]) -> Any:
        return await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            response_model=model_cls,
        )

    return call


GRADING_SYSTEM = """\
You grade a candidate response against a rubric. Be strict and specific.
Judge only what is in front of you: do not reward length, confident tone, or
formatting. A short correct answer outranks a long plausible one.
Set is_faithful to false if ANY factual claim is unsupported by the context given.
"""

PAIRWISE_SYSTEM = """\
You compare two candidate responses to the same request and pick the better one.
Judge only substance: correctness first, then completeness, then clarity.
Do NOT prefer a response for being longer, more confident, or better formatted.
If the two are of genuinely equal quality, answer "tie" — that is a real verdict,
not a failure to decide.
"""


def grading_judge(
    caller: StructuredCaller,
    *,
    rubric: str = "",
    name: str = "judge",
    context_of: Callable[[Example], str] | None = None,
) -> Any:
    """Build an async evaluator that grades each output 1-5 and normalizes to 0-1.

    Produces three components: the normalized score, faithfulness as a 0/1 check,
    and the raw 1-5 rating carried in ``detail`` so a report can show the reason.
    """

    async def evaluate(example: Example, output: Any) -> Score:
        context = context_of(example) if context_of else str(example.input)
        prompt = (
            f"{rubric}\n\n" if rubric else ""
        ) + f"<request>\n{context}\n</request>\n\n<response>\n{output}\n</response>"
        verdict: JudgeScore = await caller(
            system=GRADING_SYSTEM, prompt=prompt, model_cls=JudgeScore
        )
        return Score.from_components(
            [
                Component(name=name, value=verdict.normalized, detail=verdict.reason),
                Component(name=f"{name}_faithful", value=float(verdict.is_faithful)),
            ],
            detail={"raw_score": verdict.score, "reason": verdict.reason},
        )

    return evaluate


def _pair_prompt(request: str, first: str, second: str) -> str:
    return (
        f"<request>\n{request}\n</request>\n\n"
        f"<response_A>\n{first}\n</response_A>\n\n"
        f"<response_B>\n{second}\n</response_B>"
    )


async def judge_pairwise(
    caller: StructuredCaller,
    *,
    request: str,
    a: str,
    b: str,
) -> tuple[str, PairwiseVerdict, PairwiseVerdict]:
    """Compare two responses in **both** orders and reconcile.

    Returns the reconciled winner plus both raw verdicts. When the two orders
    disagree the result is a tie: the judge has demonstrated, on this specific pair,
    that its answer depends on presentation rather than content, so neither verdict
    has earned the right to decide.
    """
    forward, reverse = await asyncio.gather(
        caller(
            system=PAIRWISE_SYSTEM, prompt=_pair_prompt(request, a, b), model_cls=PairwiseVerdict
        ),
        caller(
            system=PAIRWISE_SYSTEM, prompt=_pair_prompt(request, b, a), model_cls=PairwiseVerdict
        ),
    )

    # In the reversed run, slot A held response b — translate back to a/b terms.
    flipped = {"A": "B", "B": "A", "tie": "tie"}[reverse.winner]
    winner = forward.winner if forward.winner == flipped else "tie"
    return winner, forward, reverse


@dataclass(frozen=True)
class PositionBias:
    """The measured result of an order-swap audit."""

    n_pairs: int
    disagreements: int
    prefers_first: int
    prefers_second: int
    ties: int
    # Self-disagreement when the SAME order is judged twice. Without it the swap
    # experiment is confounded and its number is not interpretable — see
    # :func:`measure_self_consistency`.
    noise_floor: float | None = None

    # Per-pair outcomes, so a caller can put a confidence interval on either rate.
    # Both are proportions estimated from a modest number of pairs, and quoting a
    # figure like "31.2%" without its interval is precisely the error this class
    # exists to prevent.
    swap_flips: tuple[bool, ...] = ()
    control_flips: tuple[bool, ...] = ()

    @property
    def disagreement_rate(self) -> float:
        """Share of pairs whose verdict flipped when the order was swapped.

        This is the headline number. 0.0 is a judge whose verdicts are about
        content; anything large means a meaningful share of your pairwise results
        were decided by slot position.
        """
        return self.disagreements / self.n_pairs if self.n_pairs else 0.0

    @property
    def first_slot_preference(self) -> float:
        """How lopsided the judge is toward one slot, from -1.0 to 1.0.

        Computed over decided verdicts only. 0.0 is balanced; positive means the
        first-presented response wins more often than the second, whichever
        response that happens to be.
        """
        decided = self.prefers_first + self.prefers_second
        if not decided:
            return 0.0
        return (self.prefers_first - self.prefers_second) / decided

    @property
    def excess_over_noise(self) -> float | None:
        """Swap-disagreement minus the same-order noise floor.

        This, not :attr:`disagreement_rate`, is the number attributable to position.
        At or below zero, the flips are resampling and the audit found no position
        bias — reporting the raw rate instead would invent one.
        """
        if self.noise_floor is None:
            return None
        return self.disagreement_rate - self.noise_floor

    def __str__(self) -> str:
        head = (
            f"position bias over {self.n_pairs} pairs: "
            f"{self.disagreement_rate:.1%} of verdicts flip on order swap, "
            f"first-slot preference {self.first_slot_preference:+.2f}"
        )
        if self.noise_floor is None:
            return head + " (UNCONTROLLED — not interpretable on its own)"
        excess = self.excess_over_noise or 0.0
        return (
            head + f"; same-order noise floor {self.noise_floor:.1%}, "
            f"excess attributable to position {excess:+.1%}"
        )


async def measure_self_consistency(
    caller: StructuredCaller,
    pairs: Sequence[tuple[str, str, str]],
    *,
    concurrency: int = 4,
) -> float:
    """Judge each pair twice in the SAME order; return the disagreement rate.

    The control that makes a position-bias number mean anything. Temperature is
    deprecated on current models, so a judge is nondeterministic and some share of
    "flipped when we swapped the order" is just resampling. Only the excess of
    swap-disagreement over this floor is attributable to position.

    Skipping it is not a theoretical risk. Measured on claude-sonnet-5 over 16
    near-tied pairs, this floor came out at **31.2%**, while the swap experiment on
    the same pairs returned 25.0% on one run and 0.0% on the next — the naive
    reading of either would have been wrong in a different direction.
    """
    flips = await _self_consistency_flips(caller, pairs, concurrency=concurrency)
    return sum(flips) / len(flips) if flips else 0.0


async def _self_consistency_flips(
    caller: StructuredCaller,
    pairs: Sequence[tuple[str, str, str]],
    *,
    concurrency: int = 4,
) -> list[bool]:
    """Per-pair self-disagreement, kept so a caller can interval it."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(request: str, a: str, b: str) -> bool:
        async with semaphore:
            prompt = _pair_prompt(request, a, b)
            first, second = await asyncio.gather(
                caller(system=PAIRWISE_SYSTEM, prompt=prompt, model_cls=PairwiseVerdict),
                caller(system=PAIRWISE_SYSTEM, prompt=prompt, model_cls=PairwiseVerdict),
            )
            return first.winner != second.winner

    return list(await asyncio.gather(*(one(r, a, b) for r, a, b in pairs)))


async def measure_position_bias(
    caller: StructuredCaller,
    pairs: Sequence[tuple[str, str, str]],
    *,
    concurrency: int = 4,
    control: bool = True,
) -> PositionBias:
    """Audit a judge by presenting every pair in both orders.

    ``pairs`` is ``(request, response_a, response_b)``. Pairs of genuinely similar
    quality are the informative ones: a judge will agree with itself about an
    obviously better answer no matter where you put it, so an audit built only from
    lopsided pairs measures nothing and reports a reassuring zero.

    ``control`` additionally runs :func:`measure_self_consistency` and records the
    noise floor, doubling the cost. It defaults to on because the uncontrolled
    number is not interpretable, and a measurement that looks rigorous while being
    confounded is worse than no measurement. Turn it off only when you already know
    the floor for this judge and these pairs.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(request: str, a: str, b: str) -> tuple[PairwiseVerdict, PairwiseVerdict]:
        async with semaphore:
            _, forward, reverse = await judge_pairwise(caller, request=request, a=a, b=b)
            return forward, reverse

    results = await asyncio.gather(*(one(r, a, b) for r, a, b in pairs))

    disagreements = prefers_first = prefers_second = ties = 0
    swap_flips: list[bool] = []
    for forward, reverse in results:
        flipped = {"A": "B", "B": "A", "tie": "tie"}[reverse.winner]
        swap_flips.append(forward.winner != flipped)
        if forward.winner != flipped:
            disagreements += 1
        # Slot preference counts every presentation, both orders, so it measures
        # the slot rather than the responses.
        for verdict in (forward, reverse):
            if verdict.winner == "A":
                prefers_first += 1
            elif verdict.winner == "B":
                prefers_second += 1
            else:
                ties += 1

    control_flips: list[bool] = []
    floor: float | None = None
    if control:
        control_flips = await _self_consistency_flips(caller, pairs, concurrency=concurrency)
        floor = sum(control_flips) / len(control_flips) if control_flips else 0.0

    return PositionBias(
        n_pairs=len(pairs),
        disagreements=disagreements,
        prefers_first=prefers_first,
        prefers_second=prefers_second,
        ties=ties,
        noise_floor=floor,
        swap_flips=tuple(swap_flips),
        control_flips=tuple(control_flips),
    )
