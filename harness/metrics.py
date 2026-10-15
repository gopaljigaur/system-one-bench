"""Accuracy, macro F1, calibration error and latency percentiles."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def accuracy(items: Sequence[dict]) -> float:
    scored = [i for i in items if i["predicted"] is not None]
    if not scored:
        return float("nan")
    return sum(1 for i in scored if i["predicted"] == i["gold"]) / len(scored)


def macro_f1(items: Sequence[dict], n_labels: int) -> float:
    scored = [i for i in items if i["predicted"] is not None]
    if not scored:
        return float("nan")
    out = []
    for label in range(n_labels):
        tp = sum(1 for i in scored if i["gold"] == label and i["predicted"] == label)
        fp = sum(1 for i in scored if i["gold"] != label and i["predicted"] == label)
        fn = sum(1 for i in scored if i["gold"] == label and i["predicted"] != label)
        denom = 2 * tp + fp + fn
        out.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(out) / len(out)


def ece(items: Sequence[dict], bins: int = 10) -> float:
    """Expected calibration error: the average gap between stated confidence and accuracy."""
    scored = [i for i in items if i["predicted"] is not None and i["confidence"] is not None]
    if not scored:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chosen = [i for i in scored if (i["confidence"] > lo or b == 0) and i["confidence"] <= hi]
        if not chosen:
            continue
        acc = sum(1 for i in chosen if i["predicted"] == i["gold"]) / len(chosen)
        conf = sum(i["confidence"] for i in chosen) / len(chosen)
        total += (len(chosen) / len(scored)) * abs(acc - conf)
    return total


def latency(items: Sequence[dict]) -> dict[str, float]:
    values = sorted(i["latency_ms"] for i in items if i["latency_ms"] is not None)
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}

    def pct(p: float) -> float:
        return values[min(len(values) - 1, int(len(values) * p))]

    return {"p50": pct(0.50), "p95": pct(0.95), "mean": statistics.fmean(values)}


def summarize(items: Sequence[dict], n_labels: int) -> dict[str, object]:
    failures = sum(1 for i in items if i["predicted"] is None)
    lat = latency(items)
    return {
        "rows": len(items),
        "answered": len(items) - failures,
        "failures": failures,
        "accuracy": accuracy(items),
        "macro_f1": macro_f1(items, n_labels),
        "ece": ece(items),
        "latency_ms_p50": lat["p50"],
        "latency_ms_p95": lat["p95"],
        "latency_ms_mean": lat["mean"],
    }


def pretty(value: float, places: int = 3) -> str:
    return "n/a" if value != value or math.isinf(value) else f"{value:.{places}f}"
