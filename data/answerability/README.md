# Training data of the answerability-reward baseline

The "RL, answerability reward" arm (CapRL-style; `rl650` in the result files) is trained
on multiple-choice questions that a VLM wrote about each chart and that were then
verified against the raw series by a text ReAct agent with code tools
(`ts_qa/agent_parse.py`). Only questions ruled KEEP survive.

Expected files in this directory:

| file | content |
|---|---|
| `qa_verified_keep.jsonl` | one line per chart: `{id, image_path, series, qa_list:[{question (A-D), answer, verify}]}`; 9,332 charts / 26,570 questions |
| `qa_extracted.jsonl` (optional) | every extracted question before verification, 5 per chart (29 MB) |

`scripts/build_data.sh` turns `qa_verified_keep.jsonl` into the OpenRLHF manifest with
`python -m ts_rl.build_rl_dataset --min-qa 2 --val-size 500 --seed 0`
(7,243 training charts / 23,436 questions; 500 validation charts / 1,545 questions).

If the bank is not present in this directory it is distributed as a separate archive
(the repository host's file-size limit); place the file here and rerun
`scripts/build_data.sh`. Regenerating it from scratch:

```bash
# 1) question writer: a VLM through an OpenAI-compatible API (qwen-vl-max, temperature 0.9)
export DASHSCOPE_API_KEY=...
python -m ts_qa.gen_qa_caprl --in out_10k_xdomain/fragments.jsonl --image-root out_10k_xdomain \
    --out-dir out_10k_xdomain/qa_gen --model qwen-vl-max --temperature 0.9
# 2) collect the raw questions (5 per chart)
python -m ts_qa.extract_qa --in-dir out_10k_xdomain/qa_gen --out out_10k_xdomain/qa_extracted.jsonl
# 3) verifier: a text ReAct agent with code tools re-derives every answer from the raw series
#    (--agent-model / --base-url / --api-key select any OpenAI-compatible text LLM, local or hosted)
python -m ts_qa.agent_parse --in out_10k_xdomain/qa_extracted.jsonl --out out_10k_xdomain/qa_agent.jsonl \
    --agent-model <text LLM> --base-url <endpoint> --margin 0.15
# 4) keep / correct / drop, and drop charts with fewer than 2 surviving questions
python -m ts_qa.verify_and_filter --in out_10k_xdomain/qa_agent.jsonl --out data/answerability/qa_verified_keep.jsonl \
    --margin 0.15 --min-per-image 2 --on-correct overwrite
```

The verification is stochastic on the LLM side, so a regenerated bank will not be
identical to the released one; the released file is what the paper's arm was trained on.
