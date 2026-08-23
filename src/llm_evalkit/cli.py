"""The ``evalkit`` command.

Named ``evalkit`` and not ``eval``: ``eval`` is a builtin in both bash and zsh, so an
``eval run --dataset x.jsonl`` entry point would be swallowed by the shell and never
reach this package.

Week 1 exposes only what Week 1 built — inspecting a dataset and its fingerprint.
``run``, ``compare`` and ``report`` arrive with the scorer and the store behind them.
"""

from __future__ import annotations

from pathlib import Path

import click

from . import __version__
from .dataset import DatasetError, load_dir, load_jsonl


@click.group()
@click.version_option(__version__, prog_name="evalkit")
def main() -> None:
    """Evaluate LLM outputs at scale."""


@main.command("datasets")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--pattern", default="*.json", show_default=True, help="Glob used for a directory.")
@click.option("--drafts", is_flag=True, help="Include *.draft.json cases (excluded by default).")
def datasets(path: Path, pattern: str, drafts: bool) -> None:
    """Load PATH and print what it contains, plus its fingerprint."""
    try:
        dataset = (
            load_dir(path, pattern=pattern, include_drafts=drafts)
            if path.is_dir()
            else load_jsonl(path)
        )
    except DatasetError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"{dataset.name}: {len(dataset)} example(s)  [{dataset.stability}]")
    click.echo(f"fingerprint {dataset.fingerprint}")
    labelled = sum(1 for ex in dataset if ex.expected is not None)
    click.echo(f"{labelled} labelled, {len(dataset) - labelled} unlabelled")
    attached = sum(len(ex.attachments) for ex in dataset)
    if attached:
        click.echo(f"{attached} attachment(s) hashed into the fingerprint")


if __name__ == "__main__":  # pragma: no cover
    main()
