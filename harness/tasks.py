"""Labelled sets, and the typed question that asks for each one's label."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SEED = 20261015


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    split: str
    text_column: str
    classes: Sequence[str]
    question: str
    kind: str  # "choice" or "noul"
    source: str
    labels: Mapping[str, str] | None = None  # noul only: names for the two options

    def rows(self, n: int, warmup: int) -> list[tuple[str, int]]:
        from datasets import load_dataset

        data = load_dataset(self.dataset, split=self.split)
        order = list(range(len(data)))
        random.Random(SEED).shuffle(order)
        picked = []
        for i in order[: n + warmup]:
            row = data[i]
            picked.append((str(row[self.text_column]).strip(), int(row["label"])))
        return picked

    def questions(self) -> dict[str, object]:
        from decide.types import Choice, Noul

        if self.kind == "choice":
            return {"label": Choice(self.question, list(self.classes))}
        return {"label": Noul(self.question, labels=self.labels)}


TASKS: dict[str, Task] = {
    "ag_news": Task(
        name="ag_news",
        dataset="fancyzhx/ag_news",
        split="test",
        text_column="text",
        classes=["World", "Sports", "Business", "Sci/Tech"],
        question="Which section of the paper does this story belong to?",
        kind="choice",
        source="https://huggingface.co/datasets/fancyzhx/ag_news",
    ),
    "sst2": Task(
        name="sst2",
        dataset="stanfordnlp/sst2",
        split="validation",
        text_column="sentence",
        classes=["negative", "positive"],
        # Avoids the word "positive", which laya answers with a constant 0.000 at
        # confidence 1.0 unless the two options are named. sst2_named asks it that way.
        question="Is the sentiment of this review good?",
        kind="noul",
        source="https://huggingface.co/datasets/stanfordnlp/sst2",
    ),
    "sst2_named": Task(
        name="sst2_named",
        dataset="stanfordnlp/sst2",
        split="validation",
        text_column="sentence",
        classes=["negative", "positive"],
        # The same rows as sst2, asked the natural way, with the two options named
        # so the word "positive" does not collapse the answer. Backends that cannot
        # carry the names refuse the question rather than drop them.
        question="Is the sentiment of this review positive?",
        labels={"true": "positive", "false": "negative"},
        kind="noul",
        source="https://huggingface.co/datasets/stanfordnlp/sst2",
    ),
}
