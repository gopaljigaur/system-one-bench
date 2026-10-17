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
| ECE | expected calibration error: the gap between stated confidence and how often that confidence is right. Scored on the confidence the system reports, which for a `choice` is not the top probability; systems that report none are scored on the top probability |
| Latency | p50 and p95 per decision, measured in the client |
| Failures | rows where no typed answer came back at all |

Calibration is in there because these models report probabilities, and every
cascade, gate or threshold built on top of one assumes those probabilities mean
something. A model that is right 92% of the time while claiming 0.99 is a
different tool from one that is right 92% of the time and says so.

## Tasks

| Task | Type | Classes | Source |
|---|---|---|---|
| ag_news | Choice | World, Sports, Business, Sci/Tech | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), test split |
| sst2 | Noul | negative or positive | [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2), validation split |
| sst2_named | Noul | negative or positive | the same rows as sst2 |

The datasets are old, public and widely reported, which is the point: nobody
tuned these models on a benchmark nobody uses. Rows are sampled with a fixed
seed, and every system sees the same rows in the same order, after three
warm-up calls that are not timed.

`sst2` and `sst2_named` are the same 150 rows asked two ways. `sst2` asks "Is
the sentiment of this review good?" and leaves the two options with their
default names. `sst2_named` asks "Is the sentiment of this review positive?"
and names them `positive` and `negative`. The second is the question anyone
would write; the first exists because the models answer the second one badly
unless the options are named, which is the defect at the bottom of this page.

## Results

150 rows per task, Apple M4 with 32 GB, 17 October 2026, on the current release
of everything: pydecide 0.2.10, laya 0.4.9, laya-mlx 0.4.0, von-sdk 1.3.7,
sentence-transformers 6.1.0, ollama llama3.2:1b. The HTTP servers were measured
with nothing else running, then stopped before the in-process systems ran, so no
row is competing with another for the machine. Raw per-row output is in
`results/`.

### ag_news, a four-way Choice

| system | params | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |
|---|---|---|---|---|---|---|---|
| Laya 0.4.9 (laya-serve) | 421M | **0.953** | 0.953 | 0.106 | 142.0 | 222.1 | 0 |
| Laya English (MLX) | 421M | **0.953** | 0.953 | 0.106 | 39.0 | 46.8 | 0 |
| Laya typed-decisions (MLX) | 421M | **0.953** | 0.953 | 0.365 | 37.8 | 44.1 | 0 |
| Laya multilingual (MLX) | 322M | 0.920 | 0.921 | **0.047** | **15.0** | 17.1 | 0 |
| Von 1.3.7 | 395M | 0.913 | 0.914 | 0.336 | 140.8 | 202.8 | 0 |
| ms-marco MiniLM reranker | 22.7M | 0.393 | 0.326 | 0.078 | 11.6 | 16.3 | 0 |
| Llama 3.2 1B via ollama | 1.2B | n/a | n/a | n/a | 1613.4 | 5411.5 | **150** |

### sst2, a yes/no Noul with the options unnamed

| system | params | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |
|---|---|---|---|---|---|---|---|
| Laya typed-decisions (MLX) | 421M | **0.847** | 0.846 | 0.143 | 27.0 | 40.1 | 0 |
| Von 1.3.7 | 395M | **0.847** | 0.846 | **0.024** | 219.8 | 249.7 | 0 |
| Laya English (MLX) | 421M | 0.833 | 0.833 | 0.053 | 28.3 | 42.1 | 0 |
| Laya 0.4.9 (laya-serve) | 421M | 0.807 | 0.805 | 0.058 | 99.6 | 123.3 | 0 |
| Laya multilingual (MLX) | 322M | 0.527 | 0.425 | 0.373 | **12.5** | 17.0 | 0 |
| ms-marco MiniLM reranker | 22.7M | 0.473 | 0.421 | 0.448 | 5.9 | 9.1 | 0 |
| Llama 3.2 1B via ollama | 1.2B | 0.840 | 0.781 | 0.240 | 551.8 | 3066.4 | **125** |

Llama's 0.840 is 21 right out of the 25 rows that came back at all.

### sst2_named, the same rows with the options named

| system | params | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |
|---|---|---|---|---|---|---|---|
| Laya typed-decisions (MLX) | 421M | **0.907** | 0.907 | 0.091 | 24.1 | 36.5 | 0 |
| Von 1.3.7 | 395M | 0.900 | 0.900 | 0.077 | 219.4 | 239.5 | 0 |
| Laya English (MLX) | 421M | 0.887 | 0.887 | **0.049** | 22.1 | 34.3 | 0 |
| Laya 0.4.9 (laya-serve) | 421M | 0.873 | 0.873 | 0.059 | 100.1 | 123.1 | 0 |
| Laya multilingual (MLX) | 322M | 0.853 | 0.853 | 0.103 | **10.0** | 14.3 | 0 |
| ms-marco MiniLM reranker | 22.7M | n/a | n/a | n/a | n/a | n/a | **150** |
| Llama 3.2 1B via ollama | 1.2B | n/a | n/a | n/a | 483.7 | 5066.3 | **149** |

The reranker's 150 failures are refusals, not errors: a cross-encoder scores a
query against a document and has nowhere to put an option's name, so the client
refuses the question instead of answering it with the names dropped. Llama
answered one row out of 150.

Latency is not comparable across rows. The MLX entries run in process on the
GPU; the two servers run over HTTP on CPU; the ollama entries generate tokens.
Accuracy, F1, ECE and failures are comparable, since every system saw the same
rows.

### What the numbers say

**Naming the two options is worth more than changing the model.** Every system
that can carry the names scores better on `sst2_named` than on `sst2`: the
multilingual checkpoint by 32.6 points, `laya-serve` by 6.6, typed-decisions by
6.0, the English checkpoint by 5.4, Von by 5.3. The worst of them with the names
(0.853) beats the best of them without (0.847). Two things differ between those
tasks, the adjective and the names, so the table alone does not separate them;
the controlled version is in the defect section below, where the same question
word for word goes from 0.000 to 0.979 when the names are added.

**A 400M encoder matches a fine-tuned BERT on ag_news.** Published fine-tunes of
bert-base land around 0.94 to 0.95 on this test set, and three Laya checkpoints
score 0.953 without being trained on it.

**Both encoders are under-confident on a four-way choice.** Von reports 0.578 on
average while answering 0.913 of the rows correctly, and `laya-serve` reports
0.847 while answering 0.953. Von's reliability, by bin:

| von reports | rows | actually right |
|---|---|---|
| below 0.60 | 36 | 0.806 |
| 0.60 to 0.70 | 43 | 0.953 |
| 0.70 to 0.80 | 45 | 0.956 |
| 0.80 to 0.90 | 26 | 0.923 |

A gate at 0.8 escalates 79 of 150 questions it already answered correctly. The
usual worry about a decision model is confident wrong answers; here the cost
runs the other way.

**The confidence a system reports is not its top probability.** On one ag_news
story von reports 0.877 against a top probability of 0.9077, and laya reports
0.906 against 0.9779. Both numbers are in every response, and which one you
score changes the column: `laya-serve` on ag_news is 0.106 against the reported
confidence and 0.028 against the top probability. This table scores the reported
one, because that is the number a gate reads.

**Accuracy and calibration do not travel together.** The three Laya checkpoints
tie exactly on ag_news accuracy and spread 0.047 to 0.365 on calibration error,
a factor of eight for the same answers. Von is four points behind on ag_news
accuracy with 0.336, then posts the best calibration here on sst2, 0.024. Which
checkpoint is best calibrated also depends on the task: multilingual wins on the
four-way choice at 0.047, English wins on both yes/no tasks at 0.053 and 0.049.
If you gate, cascade or threshold on these probabilities, the accuracy column is
not the column to read.

**The multilingual checkpoint's yes/no weakness was mostly the option names.**
It scores 0.527 on `sst2` and 0.853 on `sst2_named`. It is still last of the
Laya checkpoints and its calibration error is twice the English one's, so the
English checkpoint is still the better default, but "barely above chance" was a
property of the question, not of the weights.

**laya-serve picks the checkpoint per request, and sometimes picks wrong.**
`laya-serve` runs Laya's `Router`, which reads the state's language and chooses,
and the client sends no explicit model. On sst2 it sent 143 of 150 rows to the
English checkpoint, which answered them at 0.832, and 7 to the multilingual one,
which got 2 of those 7 right. They are plainly English: `cool ?`, `lovely and
poignant .`, `a smart , witty follow-up .`. That is why `laya-serve` trails the
same checkpoint run in process on both yes/no tasks, by 2.6 points and 1.4. On
ag_news, whose stories are several sentences long, all 150 went to English. Name
the checkpoint if you care which one answers.

**A 1B chat model cannot do this.** Llama 3.2 1B returned no parseable typed
answer in 150 attempts at a four-way choice, failed 125 of 150 yes/no questions,
and failed 149 of 150 once the options were named. The failures are not wrong
answers; they are replies that could not be read as a typed answer at all, at a
256 token cap.

**The reranker is a floor, not a competitor.** ms-marco MiniLM scores 0.393 on
four classes where guessing gets 0.25. It emits one relevance score and was
never a classifier, and it is here to show what that costs.

### Defects found while measuring

#### A yes/no question containing "positive"

With the two options left at their default names, Laya answers exactly `0.000`
with `confidence: 1.0` to any yes/no question containing the word "positive",
whatever the state and whether or not the question means anything. Same
checkpoint, same two states, one word changed:

| question | positive text | negative text |
|---|---|---|
| Is the sentiment of this review good? | 0.603 | 0.030 |
| Is the sentiment of this review favourable? | 0.517 | 0.001 |
| Is the sentiment of this review positive? | **0.000, confidence 1.0** | 0.000, confidence 1.0 |
| Is the weather positive? | **0.000, confidence 1.0** | 0.000, confidence 1.0 |

Asked as a two-option `choice` instead, the same question degrades to a coin
flip but says so, reporting confidence 0.0003, which an abstention threshold
catches. The `noul` form is wrong at the top of the confidence range, where no
threshold can catch it.

**Naming the options fixes it**, which is why `sst2_named` exists:

```python
Noul("Is the sentiment of this review positive?",
     labels={"true": "positive", "false": "negative"})
```

0.9793 on the positive review against 0.0000 on the negative, from the wording
that was returning a constant. That is the same question word for word, so it
isolates the names from the adjective.

Reported upstream as
[laya#1244](https://github.com/NandhaKishorM/laya/issues/1244) and reproduced
there by the maintainer, who narrowed it below where this benchmark could: it
is token level, not semantic. Lowercase `positive` collapses as the predicate
of an "Is X ... positive" frame, while `Positive?` answers 0.5894 and
`POSITIVE?` answers 0.6524 on the same state. It reproduces on laya 0.3.28 and
0.4.9, on the English and typed-decisions checkpoints, in process and over
HTTP.

#### The published checkpoint ships a temperature the library rejects

Loading `convaiinnovations/laya` prints:

```
RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as
uncalibrated.
```

`rl_agent_config.json` sets `"choice:11+": 0.1006`, below the `[0.5, 5]` the
library accepts, so it is clamped to 0.5 and the confidence of any choice
question with 11 or more options is uncalibrated. `convaiinnovations/laya-typed-decisions`
ships a byte for byte identical table. Nothing here asks an 11 option question,
so no number in the tables is affected. A load that prints this still answers,
and a readiness check that only asks for an answer will not see it.

Reported upstream as
[laya#1245](https://github.com/NandhaKishorM/laya/issues/1245) and confirmed
there. The two checkpoints want different values, so deleting the entry is not
the fix.

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

The von server installs as `von-sdk`. The PyPI project called `von` is an
unrelated placeholder package, and installing it shadows the real module.

Run the servers and the in-process systems in separate passes, with the servers
stopped for the second one. Two systems competing for the machine moves the
latency columns by 40% or more, which is how the first run of this benchmark
reported numbers it should not have.

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
[pydecide](https://github.com/gopaljigaur/decide) 0.2.10, so the request shape,
parsing and timing are identical across systems and no result depends on a
vendor SDK. That client is mine, which is a conflict worth stating: it is used
here as plumbing, it is not one of the systems being measured, and swapping it
out would not change which model answers a question correctly. One place it
does decide an outcome: a backend that cannot carry an option's name refuses
the question instead of answering with the name dropped, which is why the
reranker shows 150 failures on `sst2_named` and no accuracy. It also decides
which confidence the ECE column scores, by reporting the backend's own when
there is one; 0.2.9 and earlier reported the top probability for a `Choice`,
which is why the ag_news ECE column here differs from a run on an older
release.

## Licence

MIT.
