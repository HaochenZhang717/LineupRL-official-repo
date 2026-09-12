from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .datasets import load_dataset

LETTERS = [chr(ord("A") + i) for i in range(26)]

PROMPT = (
    "Below is a text description of a time series. Using only this description, decide "
    "which class the series belongs to.\n\n"
    "Description:\n{caption}\n\n"
    "Options:\n{options}\n\n"
    "Answer with a single letter."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_captions(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["id"]] = r["caption"]
    return out


def build_examples(items, captions, classes):
    idx_of = {c: i for i, c in enumerate(classes)}
    options = "\n".join(f"{LETTERS[i]}. class {c}" for i, c in enumerate(classes))
    out = []
    for it in items:
        cap = captions.get(it.id)
        if cap is None or it.label is None:
            continue
        out.append((it.split, PROMPT.format(caption=cap.strip(), options=options),
                    idx_of[it.label]))
    return out


class Batcher:
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0

    def __call__(self, texts: list[str]) -> dict:
        chats = [self.tok.apply_chat_template([{"role": "user", "content": t}],
                                              tokenize=False, add_generation_prompt=True)
                 for t in texts]
        enc = self.tok(chats, return_tensors="pt", padding=True, padding_side="left",
                       truncation=True, max_length=self.max_len, add_special_tokens=False)
        for c in chats:
            if len(self.tok(c, add_special_tokens=False).input_ids) > self.max_len:
                self.n_truncated += 1
        return enc


def letter_token_ids(tok, k: int) -> list[int]:
    ids = []
    for letter in LETTERS[:k]:
        enc = tok(letter, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise SystemExit(f"letter {letter!r} is not a single token in this tokenizer")
        ids.append(enc[0])
    return ids


@torch.no_grad()
def evaluate(model, batcher, letter_ids, examples, batch_size, device):
    model.eval()
    preds, golds = [], []
    loss_sum, n = 0.0, 0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        enc = batcher([t for _, t, _ in chunk]).to(device)
        logits = model(**enc).logits[:, -1, :][:, letter_ids]
        y = torch.tensor([g for _, _, g in chunk], device=device)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        n += len(chunk)
        preds += logits.argmax(-1).tolist()
        golds += y.tolist()
    return (accuracy_score(golds, preds), f1_score(golds, preds, average="macro"),
            loss_sum / max(n, 1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="e.g. ucr:ECG5000")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_dataset(args.dataset, "classification", seed=args.seed, limit=args.limit)
    captions = load_captions(args.captions)
    classes = sorted({it.label for it in items if it.label is not None})
    examples = build_examples(items, captions, classes)
    splits = {s: [e for e in examples if e[0] == s] for s in ("train", "val", "test")}
    print({k: len(v) for k, v in splits.items()}, f"classes={classes}", flush=True)
    if not splits["train"] or not splits["test"]:
        raise SystemExit("empty train or test split")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = letter_token_ids(tok, len(classes))
    batcher = Batcher(tok, args.max_len)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)

    zero_acc, zero_f1, zero_loss = evaluate(model, batcher, letter_ids, splits["val"],
                                            args.eval_batch_size, device)
    print(f"epoch 0 (no tuning): val_acc={zero_acc:.4f} val_f1={zero_f1:.4f} "
          f"val_loss={zero_loss:.4f}", flush=True)

    history = [{"epoch": 0, "val_accuracy": zero_acc, "val_macro_f1": zero_f1,
                "val_loss": zero_loss}]
    best = {"epoch": 0, "val_accuracy": zero_acc, "val_loss": zero_loss}
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if "lora" in k}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train = splits["train"][:]
        rng.shuffle(train)
        tot, nb = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            chunk = train[i:i + args.batch_size]
            enc = batcher([t for _, t, _ in chunk]).to(device)
            logits = model(**enc).logits[:, -1, :][:, letter_ids]
            y = torch.tensor([g for _, _, g in chunk], device=device)
            loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        va, vf1, vloss = evaluate(model, batcher, letter_ids, splits["val"],
                                  args.eval_batch_size, device)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1),
                        "val_accuracy": va, "val_macro_f1": vf1, "val_loss": vloss})
        print(f"epoch {epoch}: train_loss={tot / max(nb, 1):.4f} val_acc={va:.4f} "
              f"val_f1={vf1:.4f} val_loss={vloss:.4f}", flush=True)
        if (va, -vloss) > (best["val_accuracy"], -best["val_loss"]):
            best = {"epoch": epoch, "val_accuracy": va, "val_loss": vloss}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if "lora" in k}

    model.load_state_dict(best_state, strict=False)
    ta, tf1, tloss = evaluate(model, batcher, letter_ids, splits["test"],
                             args.eval_batch_size, device)
    rec = {
        "run": args.run_name, "readout": "mcq_llm", "model": args.model,
        "dataset": args.dataset, "captions": args.captions,
        "classes": [int(c) for c in classes],
        "best_epoch": best["epoch"],
        "val": {"accuracy": best["val_accuracy"], "loss": best["val_loss"]},
        "test": {"accuracy": ta, "macro_f1": tf1, "loss": tloss},
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_truncated_encodings": batcher.n_truncated, "max_len": args.max_len,
        "trainable_params": trainable, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "lora_r": args.lora_r, "minutes": round((time.time() - t0) / 60, 1),
        "history": history,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rec, indent=1))
    print("MCQ_RESULT " + json.dumps({k: rec[k] for k in
                                      ("run", "dataset", "best_epoch", "val", "test")}),
          flush=True)


if __name__ == "__main__":
    main()
