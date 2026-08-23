"""The ``evalkit`` command.

Named ``evalkit`` and not ``eval``: ``eval`` is a builtin in both bash and zsh, so an
``eval run --dataset x.jsonl`` entry point would be swallowed by the shell and never
reach this package.
"""

from __future__ import annotations

import click

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="evalkit")
def main() -> None:
    """Evaluate LLM outputs at scale."""


if __name__ == "__main__":  # pragma: no cover
    main()
