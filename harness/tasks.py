"""Labelled sets, and the typed question that asks for each one's label."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

SEED = 20261015


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    split: str
    text_column: str
    labels: Sequence[str]
    question: str
    kind: str  # "choice" or "noul"
    source: str

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
            return {"label": Choice(self.question, list(self.labels))}
        return {"label": Noul(self.question)}


TASKS: dict[str, Task] = {
    "ag_news": Task(
        name="ag_news",
        dataset="fancyzhx/ag_news",
        split="test",
        text_column="text",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        question="Which section of the paper does this story belong to?",
        kind="choice",
        source="https://huggingface.co/datasets/fancyzhx/ag_news",
    ),
    "sst2": Task(
        name="sst2",
        dataset="stanfordnlp/sst2",
        split="validation",
        text_column="sentence",
        labels=["negative", "positive"],
        # Deliberately avoids the word "positive": laya returns exactly 0.000 with
        # confidence 1.0 for any yes/no question containing it, whatever the state.
        question="Is the sentiment of this review good?",
        kind="noul",
        source="https://huggingface.co/datasets/stanfordnlp/sst2",
    ),
}
