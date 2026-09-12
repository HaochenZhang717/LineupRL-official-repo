import json
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
M = "tasksource/deberta-base-long-nli"
ROOT = "results/eval_protocol/bedtime"
TARGET_SHORT = 620
TARGET_LONG = 1535

tok = AutoTokenizer.from_pretrained(M)
mdl = AutoModelForSequenceClassification.from_pretrained(M).eval()
torch.set_num_threads(16)
IDX = {v.lower(): k for k, v in mdl.config.id2label.items()}["entailment"]


def score(pairs, bs=8):
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            c = pairs[i:i + bs]
            enc = tok([p for p, _ in c], [h for _, h in c], truncation=True,
                      max_length=1280, padding=True, return_tensors="pt")
            out += mdl(**enc).logits.softmax(-1)[:, IDX].tolist()
    print(f"  done {len(out)}", flush=True)
    return np.array(out)


def load(arm):
    caps = {}
    for line in open(f"{ROOT}/{arm}/captions.jsonl"):
        r = json.loads(line)
        caps.setdefault(r["caption_key"], r["caption"])
    recs = [json.loads(l) for l in open(f"{ROOT}/{arm}/generation_deployed.jsonl")]
    return caps, [r for r in recs if r["series_uid"] in caps]


def cut(text, n):
    if len(text) <= n:
        return text
    head = text[:n]
    stop = max(head.rfind(". "), head.rfind(".\n"))
    return head[:stop + 1] if stop > n // 2 else head


cap5, rec5 = load("decod550")
capV, recV = load("valmask400")

common = [r["series_uid"] for r in recV if r["series_uid"] in cap5]
seen, uids = set(), []
for u in common:
    if u not in seen:
        seen.add(u)
        uids.append(u)
gt = {}
for r in recV:
    gt.setdefault(r["series_uid"], r["ground_truth"])
uids = uids[:N]

G = [gt[u] for u in uids]
pool5 = [cap5[u] for u in uids]
poolV = [capV[u] for u in uids]
fill5 = [cap5[uids[(i + 7) % len(uids)]] for i in range(len(uids))]

conds = {
    "valmask400 original": list(zip(poolV, G)),
    "valmask400 cut to ~620ch": [(cut(c, TARGET_SHORT), g) for c, g in zip(poolV, G)],
    "decod550 original": list(zip(pool5, G)),
    "decod550 padded to ~1535ch": [((c + " " + f)[:TARGET_LONG], g)
                                   for c, f, g in zip(pool5, fill5, G)],
}
res = {}
for name, pairs in conds.items():
    print(f"== {name}  (mean {np.mean([len(p) for p, _ in pairs]):.0f} chars)", flush=True)
    res[name] = score(pairs)

print(f"\nn={len(uids)} series, same items in every row")
for name, a in res.items():
    print(f"{name:30} mean P={a.mean():.4f}  rate(P>=0.5)={(a >= 0.5).mean():.4f}")
print()
print(f"valmask400 shortened: {(res['valmask400 cut to ~620ch'] >= 0.5).mean() - (res['valmask400 original'] >= 0.5).mean():+.4f}")
print(f"decod550 lengthened : {(res['decod550 padded to ~1535ch'] >= 0.5).mean() - (res['decod550 original'] >= 0.5).mean():+.4f}")
print(f"original gap (5-V)  : {(res['decod550 original'] >= 0.5).mean() - (res['valmask400 original'] >= 0.5).mean():+.4f}")
