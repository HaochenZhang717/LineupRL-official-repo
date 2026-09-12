from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

SIDECAR = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True, help="mp_rank_00_model_states.pt")
    ap.add_argument("--config-src", required=True,
                    help="an existing HF export of the SAME model, for config + tokenizer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty -- refusing to overwrite")

    from transformers import AutoConfig
    import transformers

    print(f"transformers {transformers.__version__}", flush=True)
    cfg = AutoConfig.from_pretrained(args.config_src)
    cls_name = cfg.architectures[0]
    cls = getattr(transformers, cls_name)
    print(f"architecture {cls_name}", flush=True)

    print(f"loading {args.state}", flush=True)
    ck = torch.load(args.state, map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("module", ck)
    print(f"{len(sd)} tensors, dtypes {sorted({str(v.dtype) for v in sd.values()})}", flush=True)

    with torch.device("meta"):
        model = cls(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise SystemExit(
            f"state dict does not match {cls_name}: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    print("state dict matched the architecture exactly", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in SIDECAR:
        src = Path(args.config_src) / name
        if src.exists() and not (out / name).exists():
            shutil.copy2(src, out / name)

    ref = Path(args.config_src) / "model.safetensors.index.json"
    got = out / "model.safetensors.index.json"
    if ref.exists() and got.exists():
        a = set(json.load(open(ref))["weight_map"])
        b = set(json.load(open(got))["weight_map"])
        tied = set(getattr(cls, "_tied_weights_keys", None) or [])
        unexplained = {k for k in a - b if k not in tied}
        if unexplained or (b - a):
            raise SystemExit(f"tensor names differ from {args.config_src}: "
                             f"missing {sorted(unexplained)[:5]}, extra {sorted(b - a)[:5]}")
        if a - b:
            print(f"tied weights not stored (correct): {sorted(a - b)}", flush=True)
        print(f"index matches {args.config_src}: {len(b)} tensors", flush=True)
    print(f"wrote {out}  ({sum(f.stat().st_size for f in out.iterdir()) / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
