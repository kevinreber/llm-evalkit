"""Tests for the concrete evaluators.

The judge tests use a fake caller with a *deliberately planted* position preference,
because a bias detector that has only ever been run against an unbiased judge has
not been shown to detect anything.
"""

from __future__ import annotations

import re

import pytest

from llm_evalkit.dataset import Dataset, Example
from llm_evalkit.evaluators.embedding import cosine, embedding_similarity
from llm_evalkit.evaluators.exact import (
    canonical,
    exact_match,
    normalized_match,
    set_f1,
    set_overlap,
)
from llm_evalkit.evaluators.llm_judge import (
    JudgeScore,
    PairwiseVerdict,
    grading_judge,
    judge_pairwise,
    measure_position_bias,
    measure_self_consistency,
)
from llm_evalkit.evaluators.regex import must_not_match, pattern_match, patterns_from_expected
from llm_evalkit.runner import run
from llm_evalkit.scorer import score_run, summarize

# --- exact / normalized -----------------------------------------------------


def test_exact_match_is_byte_for_byte() -> None:
    ex = Example(id="a", input="q", expected="Paris")

    assert exact_match(ex, "Paris").value == 1.0
    assert exact_match(ex, "paris").value == 0.0


def test_normalized_match_forgives_typography_not_content() -> None:
    ex = Example(id="a", input="q", expected="Mount Whitney")

    assert normalized_match(ex, "Mt. Whitney").value == 1.0
    assert normalized_match(ex, "  MOUNT   WHITNEY  ").value == 1.0
    assert normalized_match(ex, "Mount Rainier").value == 0.0


def test_canonical_folds_accents_and_punctuation() -> None:
    assert canonical("Lake Atitlán") == "lake atitlan"
    assert canonical("St. John's") == "st john s"


# --- set overlap ------------------------------------------------------------


def test_set_f1_gives_partial_credit() -> None:
    # "found the main place, missed a secondary one" must not score the same as
    # "found the wrong place entirely".
    partial = set_f1(["tokyo"], ["tokyo", "kyoto"])
    wrong = set_f1(["osaka"], ["tokyo", "kyoto"])

    assert partial == pytest.approx(2 / 3)
    assert wrong == 0.0
    assert partial > wrong


def test_two_empty_sets_score_one() -> None:
    # Correctly declining to fill a field the source never stated is a success.
    # The metric that punishes it teaches the extractor to invent.
    assert set_f1([], []) == 1.0
    assert set_f1(["something"], []) == 0.0
    assert set_f1([], ["something"]) == 0.0


def test_containment_credits_verbose_but_correct_names() -> None:
    assert set_f1(["kiso valley tsumago to magome"], ["kiso valley"]) == 1.0


def test_a_landmark_cannot_claim_its_bare_city() -> None:
    # The contained name needs >= 2 words, or "tokyo tower" would satisfy "tokyo"
    # and every landmark would score as its city.
    assert set_f1(["tokyo tower"], ["tokyo"]) == 0.0


def test_set_overlap_scores_each_field_separately() -> None:
    evaluate = set_overlap(fields=["locations", "activities"])
    ex = Example(
        id="a",
        input="q",
        expected={"locations": ["tokyo"], "activities": ["hiking", "dining"]},
    )

    score = evaluate(ex, {"locations": ["Tokyo"], "activities": ["hiking"]})
    by_name = {c.name: c.value for c in score.components}

    assert by_name["locations"] == 1.0
    assert by_name["activities"] == pytest.approx(2 / 3)


# --- regex ------------------------------------------------------------------


def test_pattern_match_finds_an_answer_inside_prose() -> None:
    evaluate = pattern_match(r"\bParis\b")
    ex = Example(id="a", input="q")

    assert evaluate(ex, "I believe the answer is Paris, in France.").value == 1.0
    assert evaluate(ex, "Somewhere in France.").value == 0.0


def test_must_not_match_is_inverted_and_reports_what_it_caught() -> None:
    evaluate = must_not_match(r"as an AI")
    ex = Example(id="a", input="q")

    clean = evaluate(ex, "The capital is Paris.")
    tripped = evaluate(ex, "as an AI language model, I cannot")

    assert clean.value == 1.0
    assert tripped.value == 0.0
    assert "as an AI" in (tripped.components[0].detail or "")


def test_patterns_from_expected_makes_each_pattern_its_own_component() -> None:
    evaluate = patterns_from_expected()
    ex = Example(id="a", input="q", expected=["paris", "france", "eiffel"])

    score = evaluate(ex, "Paris is in France.")
    values = [c.value for c in score.components]

    assert values == [1.0, 1.0, 0.0]
    assert score.value == pytest.approx(2 / 3)


def test_asserting_nothing_is_not_applicable_not_a_pass() -> None:
    # A case that asserts nothing has been skipped, not passed. Scoring it 1.0
    # would let an empty expectation inflate the suite.
    score = patterns_from_expected()(Example(id="a", input="q", expected=None), "anything")

    assert score.value is None
    assert score.passed is None


# --- embedding --------------------------------------------------------------


def test_cosine_bounds_and_identity() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Opposed vectors clamp to 0 rather than going negative and dragging a mean
    # below zero, where no reader can interpret it.
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_cosine_rejects_undefined_inputs() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        cosine([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="zero vector"):
        cosine([0.0, 0.0], [1.0, 1.0])


def test_embedding_similarity_keeps_the_raw_number_under_a_threshold() -> None:
    def fake_embed(texts):
        table = {"cat": [1.0, 0.0], "kitten": [0.9, 0.44], "airplane": [0.0, 1.0]}
        return [table[t] for t in texts]

    ex = Example(id="a", input="q", expected="cat")
    thresholded = embedding_similarity(fake_embed, threshold=0.8)

    near = thresholded(ex, "kitten")
    far = thresholded(ex, "airplane")

    assert near.value == 1.0
    assert far.value == 0.0
    # A thresholded score that discarded the similarity would make it impossible
    # to tell a near miss from a total miss when moving the threshold later.
    assert "cosine 0.8" in (near.components[0].detail or "")
    assert "cosine 0.0" in (far.components[0].detail or "")


def test_nothing_to_compare_against_is_not_applicable() -> None:
    score = embedding_similarity(lambda t: [[1.0]] * len(t))(Example(id="a", input="q"), "x")

    assert score.value is None


# --- judge ------------------------------------------------------------------


class FakeCaller:
    """A judge with a planted, controllable position preference.

    ``first_slot_rate`` is the share of comparisons decided purely by slot rather
    than content. At 1.0 the judge always picks whatever is in slot A.
    """

    def __init__(self, *, first_slot_rate: float = 0.0, grade: int = 4) -> None:
        self.first_slot_rate = first_slot_rate
        self.grade = grade
        self.calls = 0

    async def __call__(self, *, system: str, prompt: str, model_cls):
        self.calls += 1
        if model_cls is JudgeScore:
            return JudgeScore(score=self.grade, reason="fake", is_faithful=True)

        first = prompt.split("<response_A>")[1].split("</response_A>")[0].strip()
        second = prompt.split("<response_B>")[1].split("</response_B>")[0].strip()

        # Deterministic pseudo-randomness keyed on the pair, so both orders of the
        # same pair make the same biased-or-not decision.
        biased = (hash(frozenset({first, second})) % 100) / 100.0 < self.first_slot_rate
        if biased:
            return PairwiseVerdict(winner="A", reason="position")
        # Otherwise judge on content: longer string wins, wherever it sits.
        if len(first) == len(second):
            return PairwiseVerdict(winner="tie", reason="equal")
        return PairwiseVerdict(winner="A" if len(first) > len(second) else "B", reason="content")


async def test_a_content_driven_judge_agrees_with_itself_across_orders() -> None:
    caller = FakeCaller(first_slot_rate=0.0)

    winner, forward, reverse = await judge_pairwise(
        caller, request="q", a="a much longer answer", b="short"
    )

    assert winner == "A"
    assert forward.winner == "A"
    assert reverse.winner == "B"  # same response, now in the other slot
    assert caller.calls == 2


async def test_order_disagreement_is_reconciled_to_a_tie() -> None:
    # When a judge's answer depends on presentation, neither verdict has earned
    # the right to decide the pair.
    caller = FakeCaller(first_slot_rate=1.0)

    winner, forward, reverse = await judge_pairwise(caller, request="q", a="alpha", b="beta")

    assert forward.winner == "A" and reverse.winner == "A"
    assert winner == "tie"


async def test_position_bias_is_zero_for_an_unbiased_judge() -> None:
    pairs = [(f"q{i}", f"{'long ' * (i + 2)}answer", "tiny") for i in range(12)]

    bias = await measure_position_bias(FakeCaller(first_slot_rate=0.0), pairs)

    assert bias.n_pairs == 12
    assert bias.disagreement_rate == 0.0
    # Each pair is judged in both orders, so an unbiased judge picks each slot
    # exactly as often as the other.
    assert bias.first_slot_preference == pytest.approx(0.0)


async def test_position_bias_is_detected_when_it_is_planted() -> None:
    # The test that matters: a detector only ever run against an unbiased judge
    # has not been shown to detect anything.
    pairs = [(f"q{i}", f"answer {i} alpha", f"answer {i} bravo") for i in range(20)]

    bias = await measure_position_bias(FakeCaller(first_slot_rate=1.0), pairs)

    assert bias.disagreement_rate == 1.0
    assert bias.first_slot_preference == pytest.approx(1.0)
    assert "flip" in str(bias)


async def test_grading_judge_normalizes_one_to_five_onto_zero_to_one() -> None:
    ex = Example(id="a", input="what is 2+2?")

    best = await grading_judge(FakeCaller(grade=5))(ex, "4")
    worst = await grading_judge(FakeCaller(grade=1))(ex, "purple")

    assert best.components[0].value == 1.0
    assert worst.components[0].value == 0.0
    assert best.detail["raw_score"] == 5


# --- score_run --------------------------------------------------------------


async def test_score_run_mixes_sync_and_async_evaluators() -> None:
    dataset = Dataset.from_examples(
        "qa", [Example(id=f"q{i}", input="q", expected="Paris") for i in range(3)]
    )

    async def echo(example: Example) -> str:
        return "Paris"

    result = await run(dataset, echo)
    scores = await score_run(result, [exact_match, grading_judge(FakeCaller(grade=5))])
    card = summarize(scores)

    assert len(scores) == 3
    assert card.component("exact_match").value == 1.0
    assert card.component("judge").value == 1.0
    assert card.component("judge_faithful").value == 1.0


async def test_a_failed_result_scores_not_applicable_not_zero() -> None:
    # Scoring an API timeout as a wrong answer converts an infrastructure problem
    # into a quality regression.
    dataset = Dataset.from_examples("qa", [Example(id="q0", input="q", expected="Paris")])

    async def explode(example: Example) -> str:
        raise TimeoutError("upstream")

    result = await run(dataset, explode)
    scores = await score_run(result, [exact_match])
    card = summarize(scores)

    assert len(scores) == 1
    assert scores[0].value is None
    assert card.overall is None
    assert card.component("error").n == 0


def test_regex_evaluator_accepts_a_compiled_pattern() -> None:
    evaluate = pattern_match(re.compile(r"^\d{4}-\d{2}-\d{2}$"))

    assert evaluate(Example(id="a", input="q"), "2026-08-23").value == 1.0
    assert evaluate(Example(id="a", input="q"), "August 23").value == 0.0


# --- the control that makes a bias number interpretable ---------------------


class CoinFlipCaller:
    """A judge with no position preference and no stability either.

    Alternates its verdict on every call regardless of content, which is what a
    perfectly nondeterministic judge looks like. Any swap-disagreement it produces
    is resampling, and a measurement without a control would report it as bias.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, *, system: str, prompt: str, model_cls):
        self.calls += 1
        if model_cls is JudgeScore:
            return JudgeScore(score=3, reason="fake", is_faithful=True)
        return PairwiseVerdict(winner="A" if self.calls % 2 else "B", reason="coin flip")


async def test_the_control_runs_by_default_and_records_a_floor() -> None:
    pairs = [(f"q{i}", "alpha", "bravo") for i in range(6)]

    bias = await measure_position_bias(FakeCaller(first_slot_rate=0.0), pairs)

    assert bias.noise_floor is not None
    assert bias.excess_over_noise is not None
    assert "noise floor" in str(bias)


async def test_an_uncontrolled_measurement_says_it_is_uninterpretable() -> None:
    pairs = [(f"q{i}", "alpha", "bravo") for i in range(4)]

    bias = await measure_position_bias(FakeCaller(first_slot_rate=1.0), pairs, control=False)

    assert bias.noise_floor is None
    assert bias.excess_over_noise is None
    assert "UNCONTROLLED" in str(bias)


async def test_planted_bias_survives_the_control_as_real_excess() -> None:
    pairs = [(f"q{i}", "alpha", "bravo") for i in range(8)]

    bias = await measure_position_bias(FakeCaller(first_slot_rate=1.0), pairs)

    # A judge that always picks slot A agrees with itself in the same order, so
    # the floor is zero and the whole disagreement rate is attributable to position.
    assert bias.noise_floor == 0.0
    assert bias.excess_over_noise == pytest.approx(1.0)


async def test_a_noisy_judge_is_not_reported_as_biased() -> None:
    # The finding this encodes: measured on claude-sonnet-5, the same-order noise
    # floor was 31.2% while the swap experiment returned 25.0% then 0.0% on
    # re-runs. Without the control that first number reads as position bias.
    pairs = [(f"q{i}", "alpha", "bravo") for i in range(6)]

    bias = await measure_position_bias(CoinFlipCaller(), pairs)

    assert bias.noise_floor is not None and bias.noise_floor > 0.0
    assert bias.excess_over_noise is not None
    assert bias.excess_over_noise <= 0.0


async def test_self_consistency_is_zero_for_a_deterministic_judge() -> None:
    pairs = [(f"q{i}", "a much longer answer", "tiny") for i in range(5)]

    assert await measure_self_consistency(FakeCaller(first_slot_rate=0.0), pairs) == 0.0
