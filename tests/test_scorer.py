"""Tests for aggregation and bootstrap CI.

The bootstrap is the one number the whole package's honesty rests on, so it is
checked against distributions whose true mean is known analytically rather than
against whatever the implementation happens to emit.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_evalkit.evaluators.base import Component, Score, check
from llm_evalkit.scorer import bootstrap_ci, repeat_variance, summarize

# --- bootstrap CI -----------------------------------------------------------


def test_the_interval_brackets_a_known_mean() -> None:
    # Bernoulli(0.7), n=500. True mean 0.7 and the 95% interval must contain it.
    rng = np.random.default_rng(0)
    scores = rng.binomial(1, 0.7, size=500).astype(float).tolist()

    ci = bootstrap_ci(scores)

    assert ci is not None
    assert ci.low < 0.7 < ci.high
    # Standard error at n=500 is about 0.02, so a 95% interval is roughly 0.08 wide.
    assert 0.04 < ci.width < 0.12


def test_more_examples_narrow_the_interval() -> None:
    # The whole argument for CI: 73% on 30 examples and 73% on 3000 are not the
    # same claim, and the width is what says so.
    rng = np.random.default_rng(1)
    small = rng.binomial(1, 0.73, size=30).astype(float).tolist()
    large = rng.binomial(1, 0.73, size=3000).astype(float).tolist()

    narrow = bootstrap_ci(large)
    wide = bootstrap_ci(small)

    assert narrow is not None and wide is not None
    assert narrow.width < wide.width / 3


def test_a_constant_dataset_has_a_zero_width_interval() -> None:
    ci = bootstrap_ci([1.0] * 50)

    assert ci is not None
    assert ci.low == ci.high == 1.0
    # No spread in, no spread out. A resampling scheme that manufactured one here
    # would be inventing uncertainty.
    assert ci.width == 0.0


def test_the_interval_is_reproducible_across_calls() -> None:
    scores = [0.1, 0.9, 0.4, 0.6, 0.5, 0.8, 0.2]

    assert bootstrap_ci(scores) == bootstrap_ci(scores)


def test_a_different_seed_moves_the_interval_slightly() -> None:
    scores = [0.1, 0.9, 0.4, 0.6, 0.5, 0.8, 0.2]

    assert bootstrap_ci(scores, seed=1) != bootstrap_ci(scores, seed=2)


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(0.5, 0.2, size=200).tolist()

    assert bootstrap_ci(scores, confidence=0.99).width > bootstrap_ci(scores, confidence=0.80).width


def test_no_data_means_no_interval() -> None:
    # Not [0.0, 0.0]: a caller must never be handed an interval to print when
    # there was nothing to measure.
    assert bootstrap_ci([]) is None


def test_one_observation_is_degenerate_not_confident() -> None:
    ci = bootstrap_ci([0.42])

    assert ci is not None
    assert ci.low == ci.high == pytest.approx(0.42)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([0.5], confidence=1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([0.5], n_resamples=0)


def test_overlap_is_symmetric_and_correct() -> None:
    a = bootstrap_ci([0.1, 0.2, 0.3])
    b = bootstrap_ci([0.9, 0.95, 1.0])
    assert a is not None and b is not None

    assert not a.overlaps(b)
    assert not b.overlaps(a)
    assert a.overlaps(a)


# --- repeat variance is a different question --------------------------------


def test_repeat_variance_needs_at_least_two_runs() -> None:
    # Model nondeterminism cannot be measured from a single run, and inventing a
    # spread would be exactly the dishonesty this package exists to avoid.
    assert repeat_variance([0.88]) is None
    assert repeat_variance([]) is None


def test_repeat_variance_reports_spread_across_runs() -> None:
    interval = repeat_variance([0.879, 0.881, 0.877, 0.883])

    assert interval is not None
    assert interval.low < 0.88 < interval.high
    assert interval.width < 0.02


# --- aggregation ------------------------------------------------------------


def _facets(kind: float, locations: float, inventions: int) -> Score:
    return Score.from_components(
        [
            Component("kind", kind),
            Component("locations", locations),
            Component("inventions", float(inventions), aggregation="sum"),
        ]
    )


def test_counts_sum_and_rates_average() -> None:
    card = summarize([_facets(1.0, 0.5, 1), _facets(1.0, 1.0, 0), _facets(0.0, 0.75, 2)])

    kind = card.component("kind")
    locations = card.component("locations")
    inventions = card.component("inventions")
    assert kind is not None and locations is not None and inventions is not None

    assert kind.value == pytest.approx(2 / 3)
    assert locations.value == pytest.approx(0.75)
    # The headline is "3 inventions", not "1.0 inventions per case".
    assert inventions.value == 3.0
    assert inventions.aggregation == "sum"
    # A total has no sampling interval worth printing.
    assert inventions.interval is None
    assert kind.interval is not None


def test_a_count_is_excluded_from_the_headline_number() -> None:
    # Averaging an invention count together with accuracies would produce a
    # number on no meaningful scale.
    card = summarize([_facets(1.0, 1.0, 7)])

    assert card.overall == pytest.approx(1.0)


def test_not_applicable_is_skipped_not_counted_as_zero() -> None:
    # The failure this guards: a planner that recorded no notes has no orphan
    # rate, and averaging it in as 0.0 reports a BETTER score for doing nothing.
    scores = [
        Score.from_components([Component("orphan_rate", 0.5)]),
        Score.from_components([Component("orphan_rate", None)]),
        Score.from_components([Component("orphan_rate", 0.7)]),
    ]

    card = summarize(scores)
    orphan = card.component("orphan_rate")
    assert orphan is not None

    assert orphan.value == pytest.approx(0.6)
    assert orphan.n == 2
    assert orphan.n_total == 3
    assert orphan.applicability == pytest.approx(2 / 3)


def test_a_component_that_never_applies_is_reported_not_dropped() -> None:
    card = summarize([Score.from_components([Component("never", None)])])
    never = card.component("never")

    assert never is not None
    assert never.n == 0
    # Visible at n=0 rather than absent, so a metric that stopped applying shows up.
    assert never.applicability == 0.0


def test_repeated_component_names_survive_aggregation() -> None:
    # One case asserting `ranked_above` twice is why components are a list.
    scores = [
        Score.from_components([check("ranked_above", True), check("ranked_above", False)]),
        Score.from_components([check("ranked_above", True)]),
    ]

    card = summarize(scores)
    ranked = card.component("ranked_above")

    assert ranked is not None
    assert ranked.n == 3
    assert ranked.value == pytest.approx(2 / 3)


# --- gate semantics ---------------------------------------------------------


def test_all_passed_requires_something_to_have_been_gated() -> None:
    # An evaluator that goes silent must not turn the suite green.
    empty = summarize([Score.from_components([Component("x", None)])])

    assert empty.gateable == 0
    assert empty.all_passed is False


def test_all_passed_is_true_only_when_every_check_passes() -> None:
    green = summarize([Score.from_components([check("a", True), check("b", True)])])
    red = summarize([Score.from_components([check("a", True), check("b", False)])])

    assert green.all_passed is True
    assert red.all_passed is False
    assert red.passed == 0
    assert red.gateable == 1


def test_an_empty_run_has_no_overall_and_no_interval() -> None:
    card = summarize([])

    assert card.n_examples == 0
    assert card.overall is None
    assert card.interval is None
    assert card.all_passed is False
