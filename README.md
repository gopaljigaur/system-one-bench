# system-one-bench

A comparison of System One decision models on public labelled sets.

Decision models answer typed questions (pick one of these, yes or no, score
this) in one forward pass instead of generating text. A lot of them appeared
after TypeSafe named the category in September 2026, and almost every number
published about them comes from whoever trained them, measured on their own
evaluation set.

This measures them the same way, on the same rows, on one machine.

## What is measured

| | |
|---|---|
| Accuracy | fraction of rows where the answer matched the label |
| Macro F1 | so a system that favours one class cannot hide behind accuracy |
| ECE | expected calibration error: the gap between stated confidence and how often that confidence is right |
| Latency | p50 and p95 per decision, measured in the client |
| Failures | rows where no typed answer came back at all |

Calibration is in there because these models report probabilities, and every
cascade, gate or threshold built on top of one assumes those probabilities mean
something. A model that is right 92% of the time while claiming 0.99 is a
different tool from one that is right 92% of the time and says so.

## Tasks

| Task | Type | Labels | Source |
|---|---|---|---|
| ag_news | Choice | World, Sports, Business, Sci/Tech | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), test split |
| sst2 | Noul | positive or not | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2), validation split |

Both are old, public and widely reported, which is the point: nobody tuned
these models on a benchmark nobody uses. Rows are sampled with a fixed seed,
and every system sees the same rows in the same order, after three warm-up
calls that are not timed.

## Results

Not yet published. See `results/` for the raw per-row output of each run.

## Running it

```bash
pip install -r requirements.txt
python harness/run.py --systems laya-mlx,crossencoder --tasks ag_news --n 150
```

Systems that are HTTP servers have to be started first. Each one's command is
recorded in its entry in `harness/models.py`, for example:

```bash
von serve --device cpu --port 8777
LAYA_PORT=8779 LAYA_DEVICE=cpu laya-serve
```

## Adding a system

Add an entry to `SYSTEMS` in `harness/models.py`. If it serves
`POST /v1/systemone`, which most of them do, `_server("http://host:port")` is
the whole entry and no new code is needed, because the client already speaks
that protocol. Anything else needs a backend in the client instead.

## What this does not tell you

- Two text classification tasks are not a decision model's whole job. Nothing
  here measures multi-question calls, long states, structured criteria or
  non-English input.
- Latency depends on the machine, the device and what else is running. The
  numbers are comparable to each other within one run and not to anyone else's.
- A model that scores badly here may have been trained for something else
  entirely. The reranker in the table is in that position on purpose, as a
  floor rather than a competitor.
- No system was tuned, prompted or configured beyond its defaults. A model
  that needs a particular prompt to do well will look worse here than it is.

## Method notes

The harness drives every system through one client,
[pydecide](https://github.com/gopaljigaur/decide), so the request shape,
parsing and timing are identical across systems and no result depends on a
vendor SDK. That client is mine, which is a conflict worth stating: it is used
here as plumbing, it is not one of the systems being measured, and swapping it
out would not change which model answers a question correctly.

## Licence

MIT.
