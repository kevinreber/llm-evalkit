from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from conftest import FakeAnthropic
from llm_evalkit.dataset import Dataset, Example, load_jsonl
from llm_evalkit.runner import Metered, Usage, run
from llm_evalkit.tasks import ModelPricing, anthropic_task, extract_text

HAIKU = "claude-haiku-4-5-20251001"


def make_dataset(n: int = 5) -> Dataset:
    return Dataset.from_examples(
        "qa", [Example(id=f"q{i}", input=f"question {i}", expected=f"a{i}") for i in range(n)]
    )


# --- the end-to-end path ----------------------------------------------------


async def test_five_examples_in_five_results_out(tmp_path: Path) -> None:
    rows = [{"id": f"q{i}", "input": f"question {i}", "expected": f"a{i}"} for i in range(5)]
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    dataset = load_jsonl(path)
    client = FakeAnthropic()
    task = anthropic_task(
        client,
        model=HAIKU,
        system="Answer in one word.",
        pricing={HAIKU: ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0)},
    )

    result = await run(dataset, task, concurrency=3, label="baseline")

    assert len(result) == 5
    assert result.failed == []
    assert [r.id for r in result] == ["q0", "q1", "q2", "q3", "q4"]
    assert result.results[0].output == "answer to question 0"

    # The run knows which dataset it measured, so a later comparison can check.
    assert result.fingerprint == dataset.fingerprint
    assert result.label == "baseline"

    # 5 calls x (10 in @ $1/Mtok + 5 out @ $5/Mtok) = 5 x $0.000035
    assert result.total_tokens == 75
    assert result.known_cost_usd == pytest.approx(5 * (10 * 1.0 + 5 * 5.0) / 1_000_000)
    assert result.unpriced_results == 0
    assert all(r.latency_ms > 0 for r in result)

    # The system prompt reached the API, once per example.
    assert len(client.requests) == 5
    assert {r["system"] for r in client.requests} == {"Answer in one word."}
    assert {r["temperature"] for r in client.requests} == {0.0}


async def test_results_come_back_in_dataset_order(tmp_path: Path) -> None:
    # Later examples finish first. Row-for-row alignment between two runs depends on
    # dataset order surviving that, not completion order.
    dataset = Dataset.from_examples("staggered", [Example(id=f"e{i}", input=i) for i in range(6)])

    async def slow_first(example: Example) -> int:
        await asyncio.sleep((6 - example.input) * 0.005)
        return example.input

    result = await run(dataset, slow_first, concurrency=6)

    assert [r.id for r in result] == [f"e{i}" for i in range(6)]
    assert [r.output for r in result] == list(range(6))


# --- concurrency ------------------------------------------------------------


async def test_concurrency_is_capped_by_the_semaphore() -> None:
    client = FakeAnthropic(delay=0.02)
    task = anthropic_task(client, model=HAIKU)

    await run(make_dataset(12), task, concurrency=4)

    assert client.max_in_flight == 4


async def test_concurrency_of_one_is_legal_and_serial() -> None:
    # Extraction runs sequentially today against a rate-limited API. Turning that
    # into "10 at once" must be a choice, not a floor the framework imposes.
    client = FakeAnthropic(delay=0.005)
    await run(make_dataset(4), anthropic_task(client, model=HAIKU), concurrency=1)

    assert client.max_in_flight == 1


async def test_zero_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        await run(make_dataset(1), anthropic_task(FakeAnthropic(), model=HAIKU), concurrency=0)


# --- failure isolation ------------------------------------------------------


async def test_one_bad_example_cannot_abort_the_run() -> None:
    client = FakeAnthropic(fail_on=frozenset({"question 2"}))
    result = await run(make_dataset(5), anthropic_task(client, model=HAIKU), concurrency=5)

    assert len(result) == 5
    assert len(result.succeeded) == 4
    (broken,) = result.failed
    assert broken.id == "q2"
    assert isinstance(broken.error, RuntimeError)
    assert broken.output is None
    # A failure still records how long it took to fail.
    assert broken.latency_ms > 0


async def test_a_failing_progress_callback_does_not_discard_paid_for_results(caplog) -> None:
    async def echo(example: Example) -> str:
        return str(example.input)

    def explode(_result: object) -> None:
        raise ValueError("the reporter is broken")

    result = await run(make_dataset(3), echo, on_result=explode)

    # Reporting is not the run. A broken printer must not turn three good results
    # into three ValueErrors.
    assert len(result) == 3
    assert result.failed == []
    assert [r.output for r in result] == ["question 0", "question 1", "question 2"]
    assert "on_result callback failed" in caplog.text


async def test_progress_callbacks_fire_once_per_example() -> None:
    async def echo(example: Example) -> str:
        return str(example.input)

    seen: list[str] = []
    await run(make_dataset(4), echo, concurrency=2, on_result=lambda r: seen.append(r.id))

    assert sorted(seen) == ["q0", "q1", "q2", "q3"]


# --- usage accounting -------------------------------------------------------


async def test_a_mechanical_task_reports_no_usage_at_all() -> None:
    # A deterministic offline scorer spends nothing. Recording a zeroed token count
    # would make it indistinguishable from a model call that returned empty.
    async def rank(example: Example) -> list[str]:
        return sorted(example.input["refs"])

    dataset = Dataset.from_examples(
        "ranking", [Example(id="c1", input={"refs": ["b", "a"]}, expected={"top": "a"})]
    )
    result = await run(dataset, rank)

    assert result.results[0].output == ["a", "b"]
    assert result.results[0].usage is None
    assert result.metered_results == 0
    assert result.known_cost_usd == 0.0
    assert result.unpriced_results == 0


async def test_an_unpriced_model_costs_unknown_not_zero() -> None:
    client = FakeAnthropic()
    # No pricing table entry for this model.
    task = anthropic_task(client, model="some-new-model")
    result = await run(make_dataset(3), task, concurrency=3)

    assert result.metered_results == 3
    assert result.unpriced_results == 3
    assert result.total_tokens == 45
    # Known cost is a floor, and the caller is told how much is missing from it.
    assert result.known_cost_usd == 0.0


async def test_usage_can_be_reported_by_any_task_via_metered() -> None:
    async def custom(example: Example) -> Metered:
        return Metered(value="x", usage=Usage(input_tokens=7, output_tokens=3, cost_usd=0.5))

    result = await run(make_dataset(2), custom)

    assert result.total_tokens == 20
    assert result.known_cost_usd == pytest.approx(1.0)


# --- response shape ---------------------------------------------------------


def test_extract_text_handles_blocks_and_flattened_strings() -> None:
    from conftest import FakeTextBlock

    assert extract_text([FakeTextBlock("hello "), FakeTextBlock("world")]) == "hello world"
    assert extract_text([{"type": "text", "text": "dict form"}]) == "dict form"
    # Some gateways flatten content to a plain string. Tolerating that here is what
    # lets the same task point at a proxy without an adapter layer.
    assert extract_text("already a string") == "already a string"
    assert extract_text(None) == ""


async def test_an_uninferable_input_says_how_to_fix_it() -> None:
    dataset = Dataset.from_examples("odd", [Example(id="x", input=42)])
    result = await run(dataset, anthropic_task(FakeAnthropic(), model=HAIKU))

    (failure,) = result.failed
    assert isinstance(failure.error, TypeError)
    assert "build_messages" in str(failure.error)
