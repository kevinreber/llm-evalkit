"""Measure LLM-judge position bias, with the control that makes it interpretable.

    uv run python scripts/measure_bias.py
    uv run python scripts/measure_bias.py --dataset datasets/judge_pairs.jsonl

Presents every pair in both orders and counts how often the verdict flips. That
number alone means nothing: temperature is deprecated on current models, so a judge
is nondeterministic, and some share of the flips is resampling rather than position.
So it also judges each pair TWICE IN THE SAME ORDER to establish a noise floor, and
reports the excess of one over the other.

Both figures are proportions from a modest number of pairs, so both get bootstrap
confidence intervals. Quoting "31.2%" bare would be the same error this package was
built to stop.

Costs roughly four judge calls per pair. The default dataset is 50 pairs, so about
200 calls.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from llm_evalkit import bootstrap_ci, load_jsonl  # noqa: E402
from llm_evalkit.evaluators.llm_judge import instructor_caller, measure_position_bias  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Best-effort .env loading, so the script works without an exported key."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


def rate_with_ci(flips: tuple[bool, ...]) -> str:
    scores = [float(f) for f in flips]
    rate = sum(scores) / len(scores) if scores else 0.0
    interval = bootstrap_ci(scores)
    return f"{rate:6.1%}   CI {interval}" if interval else f"{rate:6.1%}"


def paired_excess_ci(swap: tuple[bool, ...], control: tuple[bool, ...]) -> str:
    """Interval on the excess, bootstrapped over PER-PAIR differences.

    Both rates are measured on the same pairs, so they are paired samples and the
    difference must be resampled as such. Comparing the two marginal intervals and
    checking for overlap is the common shortcut and it is wrong twice over: it
    throws away the pairing, and interval overlap was never a significance test to
    begin with.
    """
    if not swap or len(swap) != len(control):
        return "n/a"
    diffs = [float(s) - float(c) for s, c in zip(swap, control, strict=True)]
    mean = sum(diffs) / len(diffs)
    interval = bootstrap_ci(diffs)
    if interval is None:
        return f"{mean:+6.1%}"
    excludes_zero = interval.low > 0.0 or interval.high < 0.0
    verdict = "excludes 0" if excludes_zero else "includes 0"
    return f"{mean:+6.1%}   CI [{interval.low:+.3f}, {interval.high:+.3f}]  ({verdict})"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets" / "judge_pairs.jsonl")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--env",
        type=Path,
        default=Path.home() / "Projects" / "navi" / ".env",
        help="dotenv file to read ANTHROPIC_API_KEY from if it is not exported",
    )
    args = parser.parse_args()

    load_dotenv(args.env)
    if os.environ.get("NAVI_ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["NAVI_ANTHROPIC_API_KEY"]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY. Export one or pass --env pointing at a dotenv file.")
        return 1

    import instructor
    from anthropic import AsyncAnthropic

    dataset = load_jsonl(args.dataset)
    pairs = [(ex.input["request"], ex.input["a"], ex.input["b"]) for ex in dataset]
    closeness = [ex.metadata.get("closeness", "unknown") for ex in dataset]

    caller = instructor_caller(
        instructor.from_anthropic(AsyncAnthropic()), model=args.model, max_tokens=300
    )

    print(f"dataset:  {dataset.name} ({len(dataset)} pairs, {dataset.short_fingerprint})")
    print(f"judge:    {args.model}")
    print(f"calls:    ~{len(pairs) * 4} (both orders, plus a same-order control)\n")

    bias = await measure_position_bias(caller, pairs, concurrency=args.concurrency)

    print("=" * 68)
    print(f"  {'swap disagreement':<26} {rate_with_ci(bias.swap_flips)}")
    print(f"  {'same-order noise floor':<26} {rate_with_ci(bias.control_flips)}")
    print("-" * 68)
    excess = bias.excess_over_noise or 0.0
    print(
        f"  {'excess (position bias)':<26} {paired_excess_ci(bias.swap_flips, bias.control_flips)}"
    )
    print(f"  {'first-slot preference':<26} {bias.first_slot_preference:+6.3f}")
    print(
        f"  {'verdicts A / B / tie':<26} {bias.prefers_first} / {bias.prefers_second} / {bias.ties}"
    )
    print("=" * 68)

    # Lopsided pairs are the sanity check on the whole experiment. If the judge
    # cannot reliably pick the obviously better answer, its ties on near-tied pairs
    # are not judgement, they are noise, and nothing else here means anything.
    for label in ("lopsided", "tied"):
        idx = [i for i, c in enumerate(closeness) if c == label]
        if not idx:
            continue
        swap = tuple(bias.swap_flips[i] for i in idx)
        ctrl = tuple(bias.control_flips[i] for i in idx)
        print(f"\n  {label} pairs (n={len(idx)})")
        print(f"    {'swap disagreement':<24} {rate_with_ci(swap)}")
        print(f"    {'noise floor':<24} {rate_with_ci(ctrl)}")
        print(f"    {'excess':<24} {paired_excess_ci(swap, ctrl)}")

    print(f"\n  {bias}\n")

    # Read the INTERVAL, not the point estimate. A positive excess whose interval
    # spans zero is not a finding, and announcing it as one would be the exact
    # failure this package exists to prevent.
    diffs = [float(s) - float(c) for s, c in zip(bias.swap_flips, bias.control_flips, strict=True)]
    interval = bootstrap_ci(diffs) if diffs else None
    if interval is None:
        print("  Reading: nothing measured.")
    elif interval.low > 0.0:
        print("  Reading: position bias is real. Even the low end of the interval")
        print(f"  ({interval.low:+.1%}) is above zero, so the excess is not noise.")
    elif interval.high < 0.0:
        print("  Reading: swap-disagreement is BELOW the noise floor, which should not")
        print("  happen and suggests a bug in the experiment rather than a finding.")
    else:
        print(f"  Reading: point estimate {excess:+.1%}, but the interval spans zero")
        print(f"  ({interval.low:+.1%} to {interval.high:+.1%}) — position bias is NOT")
        print("  detectable at this sample size. More pairs would be needed to say.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
