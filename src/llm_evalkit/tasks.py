"""Built-in tasks. Currently one: a call to the Anthropic Messages API.

This lives beside the runner rather than inside it on purpose. A task is just
``async (Example) -> output``, so calling a model is one implementation of that
interface and not a privileged one — a suite scoring a pure ranking function or a
harvested production artifact uses the same runner and never imports this module.

Pricing is supplied by the caller, not shipped as a table. List prices change, and a
stale hardcoded table does not fail loudly; it silently reports the wrong number,
which is worse than reporting none. When a model has no entry, its cost is recorded
as ``None`` (unknown) and the run reports how many results were unpriced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .dataset import Example
from .runner import Metered, Task, Usage

__all__ = ["ModelPricing", "anthropic_task", "extract_text"]


@dataclass(frozen=True)
class ModelPricing:
    """Dollars per million tokens, as published by the provider."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


def extract_text(content: Any) -> str:
    """Pull assistant text out of a Messages response body.

    Handles both shapes on purpose. The Anthropic SDK returns a list of typed content
    blocks; some gateways and proxies flatten that to a plain string. Tolerating both
    here is what lets the same task point at either without an adapter layer.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _default_messages(example: Example) -> list[dict[str, Any]]:
    payload = example.input
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        if "messages" in payload:
            return list(payload["messages"])
        for key in ("prompt", "text", "question"):
            if isinstance(payload.get(key), str):
                return [{"role": "user", "content": payload[key]}]
    raise TypeError(
        f"example {example.id!r}: cannot infer messages from input of type "
        f"{type(payload).__name__}. Pass build_messages= to say how this dataset "
        "turns a case into a request."
    )


def anthropic_task(
    client: Any,
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    pricing: Mapping[str, ModelPricing] | None = None,
    build_messages: Callable[[Example], list[dict[str, Any]]] = _default_messages,
    **create_kwargs: Any,
) -> Task:
    """Build a task that sends each example to the Messages API and returns its text.

    ``client`` is an ``anthropic.AsyncAnthropic`` (or anything with the same
    ``messages.create`` coroutine — which is what makes this testable without a key).

    ``temperature`` defaults to 0. An eval that samples at the default temperature
    measures the model *and* the sampler, and then attributes all of the resulting
    wobble to whatever changed between runs.
    """
    prices = dict(pricing or {})

    async def _task(example: Example) -> Metered:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": build_messages(example),
            **create_kwargs,
        }
        if system is not None:
            request["system"] = system

        response = await client.messages.create(**request)

        raw_usage = getattr(response, "usage", None)
        input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
        served = getattr(response, "model", None) or model
        price = prices.get(served) or prices.get(model)

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=price.cost(input_tokens, output_tokens) if price else None,
            model=served,
        )
        return Metered(value=extract_text(getattr(response, "content", None)), usage=usage)

    return _task
