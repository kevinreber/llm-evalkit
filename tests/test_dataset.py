from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evalkit.dataset import (
    Dataset,
    DatasetError,
    Example,
    default_adapter,
    fingerprint,
    load_dir,
    load_jsonl,
)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_jsonl_reads_every_row(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "qa.jsonl",
        [
            {"id": "a", "input": "capital of France?", "expected": "Paris"},
            {"id": "b", "input": "capital of Japan?", "expected": "Tokyo"},
        ],
    )
    dataset = load_jsonl(path)

    assert len(dataset) == 2
    assert [ex.id for ex in dataset] == ["a", "b"]
    assert dataset[0].input == "capital of France?"
    assert dataset[0].expected == "Paris"


def test_load_jsonl_names_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "a", "input": "ok"}\n{"id": "b", oops}\n', encoding="utf-8")

    with pytest.raises(DatasetError) as caught:
        load_jsonl(path)

    assert "broken.jsonl:2" in str(caught.value)


def test_blank_lines_are_skipped_without_shifting_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "gappy.jsonl"
    path.write_text('{"id": "a", "input": "ok"}\n\n{"id": "b", nope}\n', encoding="utf-8")

    with pytest.raises(DatasetError) as caught:
        load_jsonl(path)

    assert "gappy.jsonl:3" in str(caught.value)


# --- the case-file shape real golden sets actually use ----------------------


def test_load_dir_takes_id_from_the_filename_and_input_from_the_record(tmp_path: Path) -> None:
    (tmp_path / "04-lake-atitlan.json").write_text(
        json.dumps(
            {
                "source_type": "manual",
                "text": "the most beautiful view I've ever seen #guatemala",
                "expected": {"kind": "place", "locations": ["lake atitlan"]},
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dir(tmp_path, name="extraction")

    (example,) = dataset.examples
    assert example.id == "04-lake-atitlan"
    assert example.input["source_type"] == "manual"
    assert example.expected == {"kind": "place", "locations": ["lake atitlan"]}
    # The label must not ride along inside the payload handed to the task.
    assert "expected" not in example.input


def test_expect_is_read_as_a_label_and_why_is_promoted(tmp_path: Path) -> None:
    (tmp_path / "01-priority-stars.json").write_text(
        json.dumps(
            {
                "name": "priority stars break a tie",
                "why": "starring a reference is the only explicit ranking lever",
                "references": [{"id": "a"}, {"id": "b"}],
                "expect": {"top": "a"},
            }
        ),
        encoding="utf-8",
    )
    (example,) = load_dir(tmp_path).examples

    assert example.id == "priority stars break a tie"
    assert example.why == "starring a reference is the only explicit ranking lever"
    assert example.expected == {"top": "a"}
    assert "expect" not in example.input
    # The world spec survives untouched — the framework never looks inside it.
    assert example.input["references"] == [{"id": "a"}, {"id": "b"}]


def test_a_file_holding_an_array_becomes_one_example_per_element(tmp_path: Path) -> None:
    (tmp_path / "prod-plans.json").write_text(
        json.dumps([{"blocks": [1]}, {"blocks": [2]}, {"blocks": [3]}]), encoding="utf-8"
    )
    dataset = load_dir(tmp_path)

    assert [ex.id for ex in dataset] == ["prod-plans#0", "prod-plans#1", "prod-plans#2"]
    assert all(ex.expected is None for ex in dataset)


def test_drafts_are_excluded_until_a_human_renames_them(tmp_path: Path) -> None:
    (tmp_path / "corrected.json").write_text(json.dumps({"text": "reviewed"}), encoding="utf-8")
    (tmp_path / "fresh.draft.json").write_text(json.dumps({"text": "unreviewed"}), encoding="utf-8")

    assert [ex.id for ex in load_dir(tmp_path)] == ["corrected"]
    assert [ex.id for ex in load_dir(tmp_path, include_drafts=True)] == ["corrected", "fresh"]


def test_an_empty_directory_says_drafts_were_excluded(tmp_path: Path) -> None:
    (tmp_path / "only.draft.json").write_text(json.dumps({"text": "x"}), encoding="utf-8")

    with pytest.raises(DatasetError, match="drafts"):
        load_dir(tmp_path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "dupes.jsonl",
        [{"id": "same", "input": "one"}, {"id": "same", "input": "two"}],
    )

    with pytest.raises(DatasetError, match="duplicate example id"):
        load_jsonl(tmp_path / "dupes.jsonl")


# --- fingerprinting ---------------------------------------------------------


def test_fingerprint_ignores_ordering_but_not_content() -> None:
    a = Example(id="a", input="one", expected="1")
    b = Example(id="b", input="two", expected="2")

    assert fingerprint([a, b]) == fingerprint([b, a])
    assert fingerprint([a, b]) != fingerprint([a, b.model_copy(update={"expected": "3"})])


def test_fingerprint_ignores_where_the_file_lives(tmp_path: Path) -> None:
    here = Example(id="a", input="one", source_path=tmp_path / "here" / "a.json")
    there = Example(id="a", input="one", source_path=tmp_path / "elsewhere" / "a.json")

    assert fingerprint([here]) == fingerprint([there])


def test_fingerprint_covers_attachment_contents(tmp_path: Path) -> None:
    shot = tmp_path / "screenshot.png"
    shot.write_bytes(b"original pixels")
    (tmp_path / "07-alltrails.json").write_text(
        json.dumps({"source_type": "screenshot", "image": "screenshot.png"}), encoding="utf-8"
    )

    before = load_dir(tmp_path, attachment_keys=["image"])
    assert before.examples[0].attachments["image"] == shot.resolve()

    # Swap the image the case points at. The JSON is byte-identical; the dataset
    # is not the same dataset, and the fingerprint has to say so.
    shot.write_bytes(b"different pixels")
    after = load_dir(tmp_path, attachment_keys=["image"])

    assert before.fingerprint != after.fingerprint


def test_a_missing_attachment_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "case.json").write_text(json.dumps({"image": "gone.png"}), encoding="utf-8")

    with pytest.raises(DatasetError, match="missing file"):
        load_dir(tmp_path, attachment_keys=["image"])


# --- comparability ----------------------------------------------------------


def test_a_golden_set_refuses_a_mismatched_comparison() -> None:
    golden = Dataset.from_examples("gold", [Example(id="a", input="x")], stability="golden")

    assert golden.comparable_to(golden.fingerprint)
    assert not golden.comparable_to("0" * 64)


def test_a_harvested_set_is_always_comparable() -> None:
    harvested = Dataset.from_examples(
        "prod-plans", [Example(id="a", input="x")], stability="harvested"
    )

    # Re-harvesting production changes the fingerprint every time. Refusing that
    # comparison would make an organic suite permanently incomparable to itself.
    assert harvested.comparable_to("0" * 64)


def test_default_adapter_rejects_a_non_object_record(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="expected a JSON object"):
        default_adapter(["not", "an", "object"], tmp_path / "x.json", None)
