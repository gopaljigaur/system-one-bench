"""Run one or more systems over one or more tasks and write the results.

    python harness/run.py --systems laya-mlx,crossencoder --tasks ag_news --n 150

Servers have to be started first; `harness/models.py` records the command for
each one in its `note`. Results land in results/<task>.<system>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness.metrics import pretty, summarize
from harness.models import SYSTEMS
from harness.tasks import TASKS

RESULTS = Path(__file__).resolve().parent.parent / "results"
WARMUP = 3


def predict(task, answer) -> tuple[int, float]:
    from decide.types import ChoiceAnswer

    if isinstance(answer, ChoiceAnswer):
        return list(task.labels).index(answer.choice), max(answer.probabilities.values())
    positive = answer.noul >= 0.5
    return (1 if positive else 0), max(answer.noul, 1 - answer.noul)


def run(task, system, n: int) -> list[dict]:
    rows = task.rows(n, WARMUP)
    client = system.build()
    questions = task.questions()

    for text, _ in rows[:WARMUP]:
        try:
            client.decide(text, questions)
        except Exception:
            pass

    items = []
    for text, gold in rows[WARMUP:]:
        started = time.perf_counter()
        try:
            response = client.decide(text, questions)
        except Exception as exc:
            items.append(
                {
                    "gold": gold,
                    "predicted": None,
                    "confidence": None,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": type(exc).__name__,
                }
            )
            continue
        elapsed = (time.perf_counter() - started) * 1000
        guess, confidence = predict(task, response.answers["label"])
        items.append(
            {
                "gold": gold,
                "predicted": guess,
                "confidence": confidence,
                "latency_ms": elapsed,
                "error": None,
            }
        )

    close = getattr(client, "close", None)
    if callable(close):
        close()
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    table = []
    for task_name in args.tasks.split(","):
        task = TASKS[task_name]
        for key in args.systems.split(","):
            system = SYSTEMS[key]
            print(f"==> {task_name} / {key}", flush=True)
            started = time.perf_counter()
            items = run(task, system, args.n)
            summary = summarize(items, len(task.labels))
            summary |= {
                "task": task_name,
                "system": key,
                "label": system.label,
                "params": system.params,
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
            (RESULTS / f"{task_name}.{key}.json").write_text(
                json.dumps({"summary": summary, "items": items}, indent=2)
            )
            table.append(summary)
            print(
                f"    acc {pretty(summary['accuracy'])}  f1 {pretty(summary['macro_f1'])}  "
                f"ece {pretty(summary['ece'])}  p50 {pretty(summary['latency_ms_p50'], 1)}ms  "
                f"fail {summary['failures']}",
                flush=True,
            )

    print("\n| task | system | params | accuracy | macro F1 | ECE | p50 ms | failures |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table:
        print(
            f"| {r['task']} | {r['label']} | {r['params']} | {pretty(r['accuracy'])} | "
            f"{pretty(r['macro_f1'])} | {pretty(r['ece'])} | "
            f"{pretty(r['latency_ms_p50'], 1)} | {r['failures']} |"
        )


if __name__ == "__main__":
    main()
