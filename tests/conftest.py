"""Shared fakes.

The Anthropic client is faked rather than mocked with ``unittest.mock`` so the shape
of the response — content blocks, a usage object — is written down explicitly. That
shape is the contract :func:`llm_evalkit.tasks.anthropic_task` depends on, and a
``MagicMock`` would let it drift silently.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeResponse:
    content: Any
    usage: FakeUsage
    model: str


@dataclass
class FakeMessages:
    parent: FakeAnthropic

    async def create(self, **request: Any) -> FakeResponse:
        self.parent.requests.append(request)
        self.parent.in_flight += 1
        self.parent.max_in_flight = max(self.parent.max_in_flight, self.parent.in_flight)
        try:
            # Yield to the loop so overlapping calls actually overlap; without this
            # the coroutine runs to completion before the next one starts and the
            # concurrency assertions would pass on a serial runner.
            await asyncio.sleep(self.parent.delay)
            if self.parent.fail_on and request["messages"][0]["content"] in self.parent.fail_on:
                raise RuntimeError("upstream said no")
            return FakeResponse(
                content=[FakeTextBlock(text=f"answer to {request['messages'][0]['content']}")],
                usage=FakeUsage(input_tokens=10, output_tokens=5),
                model=request["model"],
            )
        finally:
            self.parent.in_flight -= 1


@dataclass
class FakeAnthropic:
    """Stands in for ``anthropic.AsyncAnthropic``."""

    delay: float = 0.01
    fail_on: frozenset[str] = frozenset()
    requests: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0

    @property
    def messages(self) -> FakeMessages:
        return FakeMessages(self)
