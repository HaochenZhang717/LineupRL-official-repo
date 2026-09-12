# decodability_rl/test_reward_design — the offline pilot behind the reward

Before any RL was run, this pilot asked two questions on 10 rendered TSFragment charts:
which negative construction makes "pick the described series out of four" a caption-only
task, and which frozen reader is strong enough to be the verifier. Its answers fixed the
reward design in `decodability_rl/rl/`: real nearest-neighbour negatives, and
Qwen2.5-14B-Instruct (non-thinking) as the reader.

## Files

| file | role |
|---|---|
| `prompts.py` | Captioner prompt and the discriminator question (`DISCRIM_*`); `build_neutral_prompt` generalises the neutral question to K options and is what the reward server uses. |
| `perturb.py` | The first negative construction: three multiset-preserving rearrangements of the true series (shuffle / reverse / segment swap). |
| `hard_negatives.py` | The second construction: other real series of the same length, nearest in standardised (mean, std, min, max) space. This is the one the reward uses. |
| `oracle.py` | A programmatic description that provably identifies the true series -- the reader ceiling. |
| `run_experiment.py` | Captions the charts with the untuned VLM, builds each question as 4 rotations (true series at A/B/C/D), asks a local discriminator under five caption conditions, writes `results.json`. |
| `make_report.py` | `results.json` -> `report.md` + option figures. |
| `make_reader_questions.py` / `score_reader_answers.py` / `make_questions_md.py` | Export the same questions as standalone prompt files for a strong external reader, score its answers against the key, and render a per-question markdown. |
| `eval_local_discriminators.py` | Scores candidate local readers on the pilot questions (first-token letter distribution, or generation with thinking on/off). |

```bash
python -m decodability_rl.test_reward_design.run_experiment --n 10 --seed 0 --out-dir decodability_rl/test_reward_design/results
python -m decodability_rl.test_reward_design.run_experiment --n 10 --seed 0 --distractors real --out-dir decodability_rl/test_reward_design/results_realneg
python -m decodability_rl.test_reward_design.eval_local_discriminators --results decodability_rl/test_reward_design/results_realneg/results.json --models Qwen/Qwen2.5-14B-Instruct --out-dir decodability_rl/test_reward_design/results_realneg/discriminators
```

## What a run writes

The result directories are not shipped; the commands above recreate them. Each
`results.json` is laid out as follows.

`summary.conditions.<cond>` holds accuracy, mean p(gold) and the letter-pick distribution
over 10 questions x 4 rotations = 40 calls for each condition: `caption` (untuned
Qwen2.5-VL-3B caption of this chart), `ds_caption` (the dataset's own caption), `oracle`,
`mismatched` (caption of another chart) and `empty`. `records[i]` holds the series, the
four candidates in display order (`display_kinds`, `display_series`), the four rotations
with the discriminator's raw reply, parsed letter and letter probabilities per condition,
and the captions used. `summary.config` records the arguments.

| run (`--out-dir`) | distractors | local discriminator | oracle | ds_caption | caption | empty |
|---|---|---|---:|---:|---:|---:|
| `results/` | derived (shuffle / reverse / segment swap), length >= 48 | Qwen2.5-3B-Instruct | 42.5% | 25.0% | 22.5% | 20.0% |
| `results_len24/` | derived, length 24 | Qwen2.5-3B-Instruct | 45.0% | 27.5% | 25.0% | 32.5% |
| `results_realneg/` | real nearest neighbours | Qwen2.5-3B-Instruct | 85.0% | 47.5% | 25.0% | 25.0% |

The 3B reader is not usable as a verifier (it cannot even read the oracle), which is why
the same questions were also answered by a strong external reader and by larger local
models.

## Findings that shaped the reward

1. **Rotation is mandatory.** With a single fixed placement the discriminator answered A
   ~80% of the time and the empty-caption control scored 50%. Every question is therefore
   asked four times with the true series at each position; a pure position prior then
   scores exactly 25%.

2. **Derived distractors leak the answer.** With shuffle / reverse / segment-swap
   negatives, the strong external reader identified the true series **72.5% (29/40) of the
   time with no caption at all** (29 of 40 answers with the gold letter, from the
   derived-distractor questions of `make_reader_questions.py` scored by
   `score_reader_answers.py`). With the caption it scored 87.5%, so only 15 points of the
   accuracy came from the caption;
   every one of the 40 no-caption rationales infers the "un-corrupted original" from the
   relations among the options. Real nearest-neighbour negatives remove the shortcut: the
   same reader scores 32.5% without a caption, 90.0% with the dataset caption and 62.5%
   with the untuned VLM caption.

3. **Reader choice.** On the real-negative questions (`eval_local_discriminators.py`,
   first-token letter distribution, `summary.<cond>.accuracy`):

   | reader | oracle | ds_caption | caption | empty |
   |---|---:|---:|---:|---:|
   | Qwen2.5-7B-Instruct | 97.5% | 70.0% | 35.0% | 20.0% |
   | Qwen2.5-14B-Instruct | 100.0% | 75.0% | 55.0% | 25.0% |
   | Qwen2.5-32B-Instruct | 100.0% | 70.0% | 47.5% | 25.0% |
   | Qwen3-14B (no thinking) | 97.5% | 82.5% | 45.0% | 30.0% |
   | Qwen3-32B (no thinking) | 97.5% | 82.5% | 47.5% | 27.5% |

   Qwen2.5-14B-Instruct saturates the oracle, puts the empty caption exactly at chance and
   leaves the most room above the untuned caption (55% -> 100%), so it is the reward's
   reader. Thinking mode was rejected: it is roughly 800x slower and has no first-token
   letter to read.
