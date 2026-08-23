"""Cases for defects found in the phase-1 self-review.

Each test names the concrete failure it prevents. They live together rather than
scattered through the other files because the shared theme is worth seeing at once:
every one of these turned a failure into a plausible-looking success, which is the
most dangerous bug class in a tool whose output is a number someone will trust.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from conftest import FakeAnthropic, FakeResponse, FakeTextBlock
from llm_evalkit.dataset import Dataset, Example, load_dir, load_jsonl
from llm_evalkit.runner import run
from llm_evalkit.tasks import ModelPricing, anthropic_task, extract_text


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# --- dataset ----------------------------------------------------------------


def test_a_zero_id_is_an_id_not_an_absence(tmp_path: Path) -> None:
    # `raw.get("id") or fallback` throws away 0, so row zero of a generated dataset
    # would be renamed to the filename while every sibling kept its number.
    path = write_jsonl(
        tmp_path / "gen.jsonl",
        [{"id": 0, "input": "first"}, {"id": 1, "input": "second"}],
    )

    assert [ex.id for ex in load_jsonl(path)] == ["0", "1"]


def test_an_empty_id_falls_through_to_the_filename(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "gen.jsonl", [{"id": "", "input": "x"}])

    assert [ex.id for ex in load_jsonl(path)] == ["gen#0"]


def test_an_array_file_indexes_ids_even_when_it_holds_one_record(tmp_path: Path) -> None:
    # Keying off len(records) meant a one-element dump got the bare stem, and every
    # id in the file changed the moment a second record was harvested into it —
    # breaking any query that addresses an example by id across runs.
    single = tmp_path / "dump.json"
    single.write_text(json.dumps([{"blocks": [1]}]), encoding="utf-8")
    assert [ex.id for ex in load_dir(tmp_path)] == ["dump#0"]

    single.write_text(json.dumps([{"blocks": [1]}, {"blocks": [2]}]), encoding="utf-8")
    assert [ex.id for ex in load_dir(tmp_path)] == ["dump#0", "dump#1"]


def test_an_object_file_still_gets_a_bare_id(tmp_path: Path) -> None:
    (tmp_path / "case.json").write_text(json.dumps({"text": "x"}), encoding="utf-8")

    assert [ex.id for ex in load_dir(tmp_path)] == ["case"]


def test_a_custom_exclude_does_not_switch_draft_filtering_off(tmp_path: Path) -> None:
    # Drafts-do-not-score is a safety property. When `exclude` REPLACED the draft
    # filter, a caller skipping a naming prefix silently began scoring the model's
    # own uncorrected output as gold.
    (tmp_path / "good.json").write_text(json.dumps({"text": "reviewed"}), encoding="utf-8")
    (tmp_path / "skipme.json").write_text(json.dumps({"text": "unwanted"}), encoding="utf-8")
    (tmp_path / "fresh.draft.json").write_text(json.dumps({"text": "unreviewed"}), encoding="utf-8")

    dataset = load_dir(tmp_path, exclude=lambda p: p.name.startswith("skipme"))

    assert [ex.id for ex in dataset] == ["good"]


def test_include_drafts_still_honours_a_custom_exclude(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps({"text": "a"}), encoding="utf-8")
    (tmp_path / "skipme.json").write_text(json.dumps({"text": "b"}), encoding="utf-8")
    (tmp_path / "fresh.draft.json").write_text(json.dumps({"text": "c"}), encoding="utf-8")

    dataset = load_dir(tmp_path, exclude=lambda p: p.name.startswith("skipme"), include_drafts=True)

    assert [ex.id for ex in dataset] == ["fresh", "good"]


def test_non_object_metadata_is_rejected(tmp_path: Path) -> None:
    from llm_evalkit.dataset import DatasetError

    row = {"id": "a", "input": "x", "metadata": ["not", "a", "map"]}
    write_jsonl(tmp_path / "bad.jsonl", [row])

    with pytest.raises(DatasetError, match="'metadata' must be an object"):
        load_jsonl(tmp_path / "bad.jsonl")


# --- runner -----------------------------------------------------------------


async def test_a_cancelled_example_voids_the_run_rather_than_scoring_as_an_error() -> None:
    # gather(return_exceptions=True) hands a child's CancelledError back like any
    # other exception. Recording it would turn cancellation into a data point — a
    # complete-looking Run carrying "errors" that could be stored as a baseline.
    dataset = Dataset.from_examples("mixed", [Example(id=f"e{i}", input=i) for i in range(3)])

    async def cancels_itself(example: Example) -> str:
        if example.id == "e1":
            raise asyncio.CancelledError
        return "fine"

    with pytest.raises(asyncio.CancelledError):
        await run(dataset, cancels_itself, concurrency=3)


async def test_interrupting_a_run_does_not_return_a_complete_looking_result() -> None:
    # The user-visible version of the same concern: Ctrl-C at example 200 of 500.
    dataset = Dataset.from_examples("slow", [Example(id=f"e{i}", input=i) for i in range(4)])

    async def never_finishes(example: Example) -> str:
        await asyncio.sleep(30)
        return "done"

    task = asyncio.create_task(run(dataset, never_finishes, concurrency=4))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# --- tasks ------------------------------------------------------------------


class _NoUsageMessages:
    """Returns content but no usage block — some proxies omit it entirely."""

    async def create(self, **request: object) -> FakeResponse:
        return FakeResponse(
            content=[FakeTextBlock(text="hi")],
            usage=None,  # type: ignore[arg-type]
            model=str(request["model"]),
        )


class _NoUsageClient:
    messages = _NoUsageMessages()


async def test_a_response_without_usage_costs_unknown_not_zero() -> None:
    task = anthropic_task(
        _NoUsageClient(),
        model="m",
        pricing={"m": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)},
    )
    dataset = Dataset.from_examples("x", [Example(id="a", input="q")])

    result = await run(dataset, task)

    # Pricing zero tokens would report a confident $0.00 for a call that certainly
    # cost something.
    assert result.failed == []
    assert result.results[0].usage is not None
    assert result.results[0].usage.cost_usd is None
    assert result.unpriced_results == 1
    assert result.known_cost_usd == 0.0


async def test_a_priced_response_with_usage_still_costs_what_it_should() -> None:
    # The guard above must not have made every call unpriced.
    task = anthropic_task(
        FakeAnthropic(),
        model="m",
        pricing={"m": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0)},
    )
    dataset = Dataset.from_examples("x", [Example(id="a", input="q")])

    result = await run(dataset, task)

    assert result.unpriced_results == 0
    assert result.known_cost_usd == pytest.approx((10 * 1.0 + 5 * 5.0) / 1_000_000)


def test_extract_text_handles_a_single_bare_block() -> None:
    # Iterating a dict walks its KEYS, every lookup misses, and the function returned
    # "" — so a real answer would have been scored as a wrong one.
    assert extract_text({"type": "text", "text": "one block"}) == "one block"
