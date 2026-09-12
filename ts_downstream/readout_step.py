from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .captioners import MockStatCaptioner, read_captions_jsonl, write_captions_jsonl
from .config import ReadoutConfig, TrainConfig
from .datasets import DownstreamItem, load_dataset
from .metrics import classification_metrics, forecasting_metrics, naive_skill, seasonal_naive_forecast
from .readout import ReadoutModel, resolve_device


def _split(items: Sequence[DownstreamItem], name: str) -> list[DownstreamItem]:
    return [it for it in items if it.split == name]


def _targets(items: Sequence[DownstreamItem], task: str):
    if task == "classification":
        return np.asarray([it.label for it in items], dtype=np.int64)
    if task == "reconstruction":
        return np.stack([np.asarray(it.series, dtype=np.float32) for it in items], axis=0)
    return np.stack([np.asarray(it.future, dtype=np.float32) for it in items], axis=0)


def train_readout(items: Sequence[DownstreamItem], captions: dict[str, str], task: str,
                  rcfg: ReadoutConfig, tcfg: TrainConfig) -> dict:
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    tr, va, te = _split(items, "train"), _split(items, "val"), _split(items, "test")
    if not va:
        va = te
    for split in (tr, va, te):
        assert split, "empty split — check dataset/captions"

    def texts(split):
        return [captions[it.id] for it in split]

    if task == "classification":
        n_classes = int(max(it.label for it in items)) + 1
        out_dim = n_classes
        loss_fn = nn.CrossEntropyLoss()
    elif task == "reconstruction":
        out_dim = len(items[0].series)
        loss_fn = nn.MSELoss()
    else:
        out_dim = len(items[0].future)
        loss_fn = nn.MSELoss()

    model = ReadoutModel(rcfg, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)

    y_tr = _targets(tr, task)
    tr_texts = texts(tr)

    micro = tcfg.micro_batch_size or tcfg.batch_size

    def evaluate(split) -> dict:
        model.eval()
        preds = []
        with torch.no_grad():
            xs = texts(split)
            for i in range(0, len(xs), micro):
                logits = model(xs[i:i + micro])
                if task == "classification":
                    preds.append(logits.argmax(-1).cpu().numpy())
                else:
                    preds.append(logits.cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        y = _targets(split, task)
        if task == "classification":
            return classification_metrics(y, pred)
        m = forecasting_metrics(y, pred)
        if task == "reconstruction":
            naive = np.repeat(_targets(tr, task).mean(axis=0)[None, :], len(split), axis=0)
        else:
            naive = np.stack([seasonal_naive_forecast(it.series, out_dim,
                              int(it.extra.get("period", 1))) for it in split], axis=0)
        m["naive_skill"] = naive_skill(y, pred, naive)
        return m

    def score(metrics: dict) -> float:
        return metrics["macro_f1"] if task == "classification" else -metrics["mse"]

    history, best, best_state, best_epoch, bad = [], None, None, -1, 0
    idx = np.arange(len(tr))
    for epoch in range(tcfg.epochs):
        model.train()
        np.random.shuffle(idx)
        ep_loss = 0.0
        for i in range(0, len(idx), tcfg.batch_size):
            bidx = idx[i:i + tcfg.batch_size]
            opt.zero_grad()
            batch_loss = 0.0
            for j in range(0, len(bidx), micro):
                cidx = bidx[j:j + micro]
                xb = [tr_texts[k] for k in cidx]
                logits = model(xb)
                if task == "classification":
                    yb = torch.tensor(y_tr[cidx], dtype=torch.long, device=device)
                else:
                    yb = torch.tensor(y_tr[cidx], dtype=torch.float32, device=device)
                loss = loss_fn(logits, yb) * (len(cidx) / len(bidx))
                loss.backward()
                batch_loss += float(loss.item())
            opt.step()
            ep_loss += batch_loss * len(bidx)
        val_m = evaluate(va)
        history.append({"epoch": epoch, "train_loss": ep_loss / len(idx), **{f"val_{k}": v for k, v in val_m.items()}})
        if best is None or score(val_m) > score(best):
            best, best_epoch, bad = val_m, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate(te)
    return {
        "task": task,
        "best_epoch": best_epoch,
        "val": best,
        "test": test_m,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "trainable_params": model.num_trainable(),
        "history": history,
    }


def _write_report(out_dir: Path, run_name: str, result: dict, args: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"run": run_name, "args": args, **result}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    task = result["task"]
    lines = [f"# {run_name}", ""]
    lines.append(f"- task: **{task}**")
    lines.append(f"- best_epoch: {result['best_epoch']}")
    lines.append(f"- n_train/val/test: {result['n_train']}/{result['n_val']}/{result['n_test']}")
    lines.append(f"- trainable_params: {result['trainable_params']:,}")
    lines.append("")
    if task == "classification":
        t = result["test"]
        lines += ["| metric | test |", "|---|---|",
                  f"| accuracy | {t['accuracy']:.4f} |",
                  f"| macro_f1 | {t['macro_f1']:.4f} |"]
    else:
        t = result["test"]
        lines += ["| metric | test |", "|---|---|",
                  f"| mse | {t['mse']:.4f} |", f"| mae | {t['mae']:.4f} |",
                  f"| naive_skill | {t.get('naive_skill', float('nan')):.4f} |"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline 3 Stage B: train caption->task readout and report metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["classification", "forecasting", "reconstruction"],
                   required=True)
    p.add_argument("--dataset", default="synthetic",
                   help="synthetic | ucr:<Name> | ett:<csv>[:<target>]")
    p.add_argument("--captions", default="auto",
                   help="path to captions.jsonl, or 'auto' to mock-caption in-process")
    p.add_argument("--encoder-model", default="google/embeddinggemma-300m",
                   help="HF model id, or 'mock' for offline")
    p.add_argument("--run-name", default="base")
    p.add_argument("--out-dir", default=None,
                   help="default results/eval_protocol/pipeline3/<dataset>/<run-name>/")
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="forward/backward chunk size; gradients accumulate so the optimizer "
                        "step is still taken per --batch-size examples. Lower this (not "
                        "--batch-size) when a long --max-text-len OOMs: it changes memory "
                        "only, leaving the optimization -- and therefore the comparison -- "
                        "untouched. Default: no chunking.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-text-len", type=int, default=ReadoutConfig.max_text_len,
                   help="captions are truncated to this many tokens by the encoder's "
                        "tokenizer. The default (256) is BELOW the caption length of several "
                        "arms -- decod550's median is ~222 tokens but sft/teacher72b's is ~591 "
                        "-- so it silently cuts only the verbose arms. The classification "
                        "readout was raised to 1400 for exactly this reason, but the "
                        "forecasting runs still used 256. Kept at 256 by default so existing "
                        "numbers stay reproducible; pass 1400 for the corrected comparison.")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--train-seed", type=int, default=None,
                   help="seeds ONLY the training loop (torch/np init, shuffling). Defaults to "
                        "--seed. Kept separate because --seed also drives load_ett's "
                        "--max-windows subsample: changing it selects a different set of "
                        "windows, which would no longer match an on-disk captions.jsonl. Use "
                        "this to average over training seeds while holding the data fixed.")
    p.add_argument("--device", default="auto")
    p.add_argument("--caption-control",
                   choices=["none", "null", "shuffled", *TEMPLATE_CONTROLS], default="none",
                   help="'null' blanks every caption; 'shuffled' permutes them within each "
                        "split (see apply_caption_control). Both need --captions to point at "
                        "a real file, since 'shuffled' re-uses those captions. The template "
                        "controls replace every caption with the window's true values "
                        "sampled at 8 or 16 evenly spaced steps, written as text.")
    p.add_argument("--fail-on-truncation", action="store_true",
                   help="refuse to run if any caption exceeds --max-text-len. Use for "
                        "comparisons across arms, where truncation is arm-dependent and "
                        "therefore a confound rather than a nuisance.")
    p.add_argument("--shuffle-key", default=None,
                   help="permutation key for --caption-control shuffled; defaults to "
                        "--dataset; the same key reproduces the same permutation.")
    return p


TEMPLATE_CONTROLS = ("points8", "points16")


def points_caption(values, k: int) -> str:
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    parts = ", ".join(f"step {i}: {v[i]:.3g}" for i in idx)
    return (f"The series runs for {n} steps. Sampled at {len(idx)} evenly spaced steps, "
            f"its values are: {parts}.")


def template_caption(item: DownstreamItem, control: str) -> str:
    if control == "points8":
        return points_caption(item.series, 8)
    if control == "points16":
        return points_caption(item.series, 16)
    raise ValueError(control)


def apply_caption_control(items: Sequence[DownstreamItem], captions: dict[str, str],
                          control: str, key: str, seed: int = 1234) -> dict[str, str]:
    if control == "none":
        return captions
    if control == "null":
        return {it.id: "" for it in items}
    if control in TEMPLATE_CONTROLS:
        return {it.id: template_caption(it, control) for it in items}
    by_split: dict[str, list[str]] = {}
    for it in items:
        by_split.setdefault(it.split, []).append(it.id)
    out: dict[str, str] = {}
    for split, ids in by_split.items():
        rng = np.random.default_rng(zlib.crc32(f"{seed}:{key}:{split}".encode()))
        perm = rng.permutation(len(ids))
        for i, p in enumerate(perm):
            out[ids[i]] = captions[ids[p]]
    return out


def caption_truncation_stats(captions: dict[str, str], rcfg: ReadoutConfig) -> dict:
    if rcfg.encoder_model == "mock":
        return {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(rcfg.encoder_model)
    except Exception as e:
        return {"error": repr(e)}
    lens = [len(tok(c)["input_ids"]) for c in captions.values()]
    cap = rcfg.max_text_len
    return {"n": len(lens), "max_tokens": int(max(lens)) if lens else 0,
            "median_tokens": int(np.median(lens)) if lens else 0,
            "limit": cap if (cap and cap > 0) else None,
            "n_truncated": int(sum(1 for x in lens if cap and cap > 0 and x > cap))}


def main(argv=None) -> dict:
    args = build_argparser().parse_args(argv)
    rcfg = ReadoutConfig(encoder_model=args.encoder_model, lora_r=args.lora_r,
                         max_text_len=args.max_text_len)
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                       micro_batch_size=args.micro_batch_size,
                       seed=args.seed if args.train_seed is None else args.train_seed,
                       device=args.device)

    items = load_dataset(args.dataset, args.task, lookback=args.lookback, horizon=args.horizon,
                         stride=args.stride, max_windows=args.max_windows, seed=args.seed,
                         limit=args.limit)
    ds_tag = args.dataset.replace(":", "_").replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("results/eval_protocol/pipeline3") / ds_tag / args.run_name

    if args.captions == "auto":
        caps = MockStatCaptioner()(items)
        ids = [it.id for it in items]
        write_captions_jsonl(out_dir / "captions.jsonl", ids, caps)
        captions = dict(zip(ids, caps))
    else:
        captions = read_captions_jsonl(args.captions)
        missing = [it.id for it in items if it.id not in captions]
        if missing:
            raise SystemExit(f"{len(missing)} items missing captions, e.g. {missing[:3]}")

    captions = apply_caption_control(items, captions, args.caption_control,
                                     args.shuffle_key or args.dataset)
    trunc = caption_truncation_stats(captions, rcfg)
    print(json.dumps({"caption_tokens": trunc}), flush=True)
    if trunc.get("n_truncated"):
        print(f"WARNING: --max-text-len {rcfg.max_text_len} truncates "
              f"{trunc['n_truncated']}/{trunc['n']} captions "
              f"(longest {trunc['max_tokens']} tokens). Pass --max-text-len 0 to disable "
              f"truncation.", flush=True)
        if args.fail_on_truncation:
            raise SystemExit("refusing to run: captions would be truncated "
                             "(--fail-on-truncation)")

    t0 = time.time()
    result = train_readout(items, captions, args.task, rcfg, tcfg)
    result["seconds"] = round(time.time() - t0, 1)
    result["caption_tokens"] = trunc
    _write_report(out_dir, args.run_name, result, vars(args))
    print(json.dumps({"run": args.run_name, "task": args.task, "test": result["test"],
                      "out_dir": str(out_dir)}, indent=2))
    return result


if __name__ == "__main__":
    main()
