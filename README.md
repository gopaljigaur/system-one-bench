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

150 rows per task, Apple M4 with 32 GB, 15 October 2026. Raw per-row output is
in `results/`.

### ag_news, a four-way Choice

| system | params | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |
|---|---|---|---|---|---|---|---|
| Laya 0.4.7 (laya-serve) | 421M | **0.953** | 0.953 | **0.028** | 160.9 | 243.4 | 0 |
| Laya multilingual (MLX) | 322M | 0.920 | 0.921 | 0.072 | **17.1** | 24.0 | 0 |
| Von 1.3 | 395M | 0.913 | 0.914 | 0.230 | 196.2 | 399.4 | 0 |
| ms-marco MiniLM reranker | 22.7M | 0.393 | 0.326 | 0.078 | 15.6 | 24.0 | 0 |
| Llama 3.2 1B via ollama | 1.2B | n/a | n/a | n/a | 1911.6 | 6180.4 | **150** |

### sst2, a yes/no Noul

| system | params | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |
|---|---|---|---|---|---|---|---|
| Von 1.3 | 395M | **0.847** | 0.846 | **0.024** | 276.2 | 387.4 | 0 |
| Llama 3.2 1B via ollama | 1.2B | 0.840 | 0.781 | 0.240 | 634.6 | 3694.0 | **125** |
| Laya 0.4.7 (laya-serve) | 421M | 0.807 | 0.805 | 0.058 | 113.5 | 149.0 | 0 |
| Laya multilingual (MLX) | 322M | 0.527 | 0.425 | 0.373 | **12.3** | 17.3 | 0 |
| ms-marco MiniLM reranker | 22.7M | 0.473 | 0.421 | 0.448 | 9.1 | 13.8 | 0 |

Latency is not comparable across rows. The MLX entry runs in process on the
GPU; the two servers run over HTTP on CPU; the ollama entries generate tokens.
Accuracy, F1, ECE and failures are comparable, since every system saw the same
rows.

### What the numbers say

**A 400M encoder beats a fine-tuned BERT on ag_news.** Published fine-tunes of
bert-base land around 0.94 to 0.95 on this test set, and Laya 0.4.7 scores
0.953 without being trained on it.

**Accuracy and calibration do not travel together.** Von and Laya 0.4.7 are
four points apart on ag_news accuracy and an order of magnitude apart on
calibration error, 0.230 against 0.028. Von then has the best calibration of
anything here on sst2, at 0.024. If you gate, cascade or threshold on these
probabilities, the accuracy column is not the column to read.

**The checkpoint matters more than the size.** Laya's 322M multilingual MLX
checkpoint scores 0.527 on yes/no, barely above a coin, while the 421M English
one scores 0.807. Both are Laya. Anything defaulting to the multilingual
checkpoint for yes/no questions is close to guessing.

**A 1B chat model cannot do this.** Llama 3.2 1B returned no parseable typed
answer in 150 attempts at a four-way choice, and failed 125 of 150 yes/no
questions, though it was right on 21 of the 25 that came back. The failures are
not wrong answers; they are replies that could not be read as a typed answer at
all, at a 256 token cap.

**The reranker is a floor, not a competitor.** ms-marco MiniLM scores 0.393 on
four classes where guessing gets 0.25. It emits one relevance score and was
never a classifier, and it is here to show what that costs.

### A defect found while measuring

Laya returns exactly `0.000` with `confidence: 1.0` for any yes/no question
containing the word "positive", whatever the state and whatever the question
means. Reproducible against `laya-serve` with no client involved:

```bash
curl -s -X POST localhost:8000/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":"this movie was absolutely wonderful, i loved every minute of it",
       "questions":{"q":{"type":"noul","instructions":"Is the sentiment of this review positive?"}}}'
# {"noul": 0.0, "confidence": 1.0}
```

| question | positive text | negative text |
|---|---|---|
| "Is the sentiment of this review good?" | 0.603 | 0.029 |
| "Is the sentiment of this review positive?" | 0.000 | 0.000 |
| "Is the weather positive?" | 0.000 | 0.000 |

The sst2 task here asks "good" for that reason, which is noted in
`harness/tasks.py`. The first version of this benchmark asked "positive" and
scored Laya at 0.500, which was measuring the defect.

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
  floor, not a competitor.
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
