"""Datasets: load the cases an eval runs over, validate them, and fingerprint them.

The fingerprint is the point. Comparing today's 75% against last week's 73% means
nothing if the case set drifted underneath, so every run records a SHA-256 over the
canonicalised dataset and later comparisons can refuse to proceed when it changed.

Two loaders, because real suites come in both shapes:

``load_jsonl``
    One example per line. Compact, streams, and is what a generated or downloaded
    benchmark looks like.

``load_dir``
    One example per JSON file in a directory. Hand-authored golden sets want this:
    cases get edited and diffed individually, a bootstrap mode writes new ones as
    files, and workflow state can live in the filename. A 31 KB itinerary case is
    unreadable as a single JSONL line.

Both funnel through an *adapter* — ``(raw, source, index) -> Example`` — because a
real case file is not shaped like ``{"input": ..., "expected": ...}``. It carries a
task-specific world spec: a source type and an image path, or a whole simulated
calendar. :func:`default_adapter` handles that by treating the record itself as the
input, and the framework never looks inside it. Only the task does, and the task is
domain code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "DRAFT_SUFFIX",
    "Dataset",
    "DatasetError",
    "Example",
    "Stability",
    "default_adapter",
    "fingerprint",
    "load_dir",
    "load_jsonl",
]

# A case whose labels are still the model's own output, not a human's. Scoring one
# would bless today's behaviour as correct, so drafts are excluded by default and
# only count once a person renames the file. The convention is borrowed from Navi's
# extraction gold set, where reviewing a pre-filled draft is roughly ten times
# faster than authoring JSON — and rubber-stamping it is the failure mode.
DRAFT_SUFFIX = ".draft.json"

# Whether the case set is expected to hold still between runs.
#
# ``golden``     hand-authored labels. A changed fingerprint means someone edited the
#                gold set, and a cross-fingerprint comparison should be refused.
# ``harvested``  sampled from production and re-sampled on every harvest. The
#                fingerprint is recorded and reported, but a mismatch is normal and
#                must not block comparison — otherwise an organic suite can never be
#                compared to itself.
Stability = Literal["golden", "harvested"]

# Keys the adapter strips from ``input``. Only the label: leaving the expected answer
# inside the payload handed to the task is a silent, self-congratulating leak.
_LABEL_KEYS = frozenset({"expected", "expect"})


class DatasetError(ValueError):
    """A dataset failed to load. Always names the file, and the line for JSONL."""


class Example(BaseModel):
    """One case.

    ``input`` and ``expected`` are deliberately untyped. Not every suite compares a
    string response against a string answer: one returns a Pydantic model scored
    field by field, one asserts an *ordering* over a whole result set, and one has no
    labels at all and scores an artifact's internal consistency. A framework that
    insisted on ``input: str`` would exclude two of those three.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    input: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # What this case protects, printed when it fails. The single best practice in
    # the suites this framework was extracted from: a red case should tell you which
    # behaviour you broke, not just that a number moved.
    why: str | None = None

    # Where this came from, and any files it points at. Both are excluded from the
    # serialised form: an absolute path differs per machine and would make the
    # fingerprint unreproducible. Attachment *contents* are still fingerprinted —
    # see :func:`fingerprint`.
    source_path: Path | None = Field(default=None, exclude=True, repr=False)
    attachments: dict[str, Path] = Field(default_factory=dict, exclude=True, repr=False)


Adapter = Callable[[Any, Path, int | None], Example]


def default_adapter(raw: Any, source: Path, index: int | None = None) -> Example:
    """Map a raw JSON record onto an :class:`Example` without knowing its schema.

    Identity comes from ``id``, then ``name``, then the filename — so a directory of
    hand-named cases gets stable, readable ids for free.

    The input is the record itself, minus the label. That is what lets a case carry
    a source type, a URL, an image reference, or an entire simulated world: the
    framework passes it through opaquely and the task, which is domain code, reads
    the fields it knows about.
    """
    if not isinstance(raw, dict):
        raise DatasetError(f"{source}: expected a JSON object, got {type(raw).__name__}")

    stem = source.name[: -len(DRAFT_SUFFIX)] if source.name.endswith(DRAFT_SUFFIX) else source.stem
    fallback = stem if index is None else f"{stem}#{index}"
    ident = str(raw.get("id") or raw.get("name") or fallback)

    if "input" in raw:
        payload: Any = raw["input"]
    else:
        payload = {k: v for k, v in raw.items() if k not in _LABEL_KEYS}
    expected = raw.get("expected", raw.get("expect"))
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise DatasetError(f"{source}: 'metadata' must be an object, got {type(metadata).__name__}")

    return Example(
        id=ident,
        input=payload,
        expected=expected,
        metadata=metadata,
        why=raw.get("why"),
        source_path=source,
    )


@dataclass(frozen=True)
class Dataset:
    """A validated, fingerprinted set of examples."""

    name: str
    examples: tuple[Example, ...]
    fingerprint: str
    stability: Stability = "golden"
    root: Path | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    @property
    def short_fingerprint(self) -> str:
        """The first 12 hex chars — enough to distinguish, short enough to print."""
        return self.fingerprint[:12]

    def comparable_to(self, other_fingerprint: str) -> bool:
        """Whether a run over this dataset may be compared against ``other_fingerprint``.

        A ``harvested`` dataset is re-sampled from production every time it is
        collected, so its fingerprint always differs and always may be compared.
        A ``golden`` set changes only when a human edits it, so a mismatch means the
        two runs measured different things and the comparison is meaningless.
        """
        return self.stability == "harvested" or self.fingerprint == other_fingerprint

    @classmethod
    def from_examples(
        cls,
        name: str,
        examples: Iterable[Example],
        *,
        stability: Stability = "golden",
        root: Path | None = None,
    ) -> Dataset:
        items = tuple(examples)
        _reject_duplicate_ids(items, name)
        return cls(
            name=name,
            examples=items,
            fingerprint=fingerprint(items),
            stability=stability,
            root=root,
        )


def _reject_duplicate_ids(examples: Sequence[Example], name: str) -> None:
    seen: dict[str, Example] = {}
    for ex in examples:
        if ex.id in seen:
            first = seen[ex.id].source_path or "<memory>"
            here = ex.source_path or "<memory>"
            raise DatasetError(
                f"{name}: duplicate example id {ex.id!r} ({first} and {here}). "
                "Ids address results across runs, so duplicates make a scorecard ambiguous."
            )
        seen[ex.id] = ex


def fingerprint(examples: Iterable[Example]) -> str:
    """SHA-256 over the canonicalised examples.

    Canonical means: sorted by id, so file ordering does not change the digest; keys
    sorted within each record; and no whitespace slack.

    Attachments are hashed **by content**, not by path. An extraction case points at
    a screenshot beside it on disk; hashing only the JSON would leave the fingerprint
    unchanged when the image behind it was replaced, which is exactly the silent
    drift the fingerprint exists to catch.
    """
    digest = hashlib.sha256()
    for ex in sorted(examples, key=lambda e: e.id):
        record = ex.model_dump(mode="json")
        blob = json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        )
        digest.update(blob.encode("utf-8"))
        digest.update(b"\x00")
        for key in sorted(ex.attachments):
            digest.update(key.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(_file_digest(ex.attachments[key]))
            digest.update(b"\x00")
    return digest.hexdigest()


def _file_digest(path: Path) -> bytes:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.digest()


def _resolve_attachments(raw: Any, source: Path, attachment_keys: Iterable[str]) -> dict[str, Path]:
    """Turn declared path-valued keys into absolute paths, relative to the case file.

    Declared rather than sniffed: the framework cannot tell a filename from any other
    string, and guessing would make an unrelated field silently load-bearing.
    """
    resolved: dict[str, Path] = {}
    if not isinstance(raw, dict):
        return resolved
    for key in attachment_keys:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (source.parent / candidate).resolve()
        if not candidate.is_file():
            raise DatasetError(
                f"{source}: attachment {key!r} points at a missing file: {candidate}"
            )
        resolved[key] = candidate
    return resolved


def load_jsonl(
    path: str | Path,
    *,
    name: str | None = None,
    adapter: Adapter = default_adapter,
    stability: Stability = "golden",
    attachment_keys: Sequence[str] = (),
) -> Dataset:
    """Load one example per line, validating every row on the way in.

    A malformed row names its line number. A 500-line dataset that fails with
    "validation error" and no location costs more to fix than it did to write.
    """
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"No such dataset file: {source}")

    examples: list[Example] = []
    with source.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{source}:{lineno}: invalid JSON — {exc.msg}") from exc
            try:
                example = adapter(raw, source, lineno - 1)
            except ValidationError as exc:
                raise DatasetError(f"{source}:{lineno}: invalid example — {exc}") from exc
            attachments = _resolve_attachments(raw, source, attachment_keys)
            if attachments:
                example = example.model_copy(update={"attachments": attachments})
            examples.append(example)

    return Dataset.from_examples(
        name or source.stem, examples, stability=stability, root=source.parent
    )


def load_dir(
    path: str | Path,
    *,
    name: str | None = None,
    pattern: str = "*.json",
    adapter: Adapter = default_adapter,
    stability: Stability = "golden",
    attachment_keys: Sequence[str] = (),
    exclude: Callable[[Path], bool] | None = None,
    include_drafts: bool = False,
) -> Dataset:
    """Load a directory of JSON case files, one or many examples each.

    A file holding an object becomes one example; a file holding an array becomes
    one per element, with ``#index`` appended to the id. That second shape is what
    lets a harvested production dump — a single API response containing many
    records — load without any bespoke code.
    """
    root = Path(path)
    if not root.is_dir():
        raise DatasetError(f"No such dataset directory: {root}")

    skip = exclude or (lambda p: not include_drafts and p.name.endswith(DRAFT_SUFFIX))

    examples: list[Example] = []
    for file in sorted(root.glob(pattern)):
        if not file.is_file() or skip(file):
            continue
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{file}: invalid JSON — {exc.msg} (line {exc.lineno})") from exc

        records = raw if isinstance(raw, list) else [raw]
        multiple = len(records) > 1
        for index, record in enumerate(records):
            try:
                example = adapter(record, file, index if multiple else None)
            except ValidationError as exc:
                raise DatasetError(f"{file}: invalid example at index {index} — {exc}") from exc
            attachments = _resolve_attachments(record, file, attachment_keys)
            if attachments:
                example = example.model_copy(update={"attachments": attachments})
            examples.append(example)

    if not examples:
        raise DatasetError(
            f"{root}: no cases matched {pattern!r}"
            + ("" if include_drafts else f" (drafts ending {DRAFT_SUFFIX} are excluded)")
        )

    return Dataset.from_examples(name or root.name, examples, stability=stability, root=root)
