from __future__ import annotations

from click.testing import CliRunner

import llm_evalkit
from llm_evalkit.cli import main


def test_the_package_reports_a_version() -> None:
    assert llm_evalkit.__version__


def test_the_console_script_resolves() -> None:
    # The entry point is `evalkit`, not `eval` — `eval` is a bash and zsh builtin
    # and would swallow the command before it reached this package.
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "evalkit" in result.output
