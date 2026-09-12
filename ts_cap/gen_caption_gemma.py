from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from ts_render.data import load_fragments
from ts_render.render import RenderConfig, render_fragment

from .prompts import MODES, CAPTION_SYSTEM, build_prompt
from .gemma_vl_local import DEFAULT_MODEL, GemmaVLLocal

MODE_LABEL = {
    "image": "image only",
    "text": "numbers only",
    "both": "image + numbers",
}


def _series_groundtruth(series: list[float]) -> dict:
    n = len(series)
    lo, hi = min(series), max(series)
    return {
        "len": n,
        "min": round(lo, 4),
        "max": round(hi, 4),
        "argmin": series.index(lo),
        "argmax": series.index(hi),
        "first": round(series[0], 4),
        "last": round(series[-1], 4),
    }


def _load_done_ids(jsonl_path: Path) -> set[int]:
    done: set[int] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(int(json.loads(line)["id"]))
                    except (KeyError, ValueError, json.JSONDecodeError):
                        pass
    return done


def _write_markdown(rows: list[dict], md_path: Path, model_id: str, modes: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Time-Series Caption Review (3 input settings)")
    lines.append("")
    lines.append(f"- model: `{model_id}`")
    lines.append(f"- fragments: **{len(rows)}**")
    lines.append(
        "- settings: " + ", ".join(f"**{m}** ({MODE_LABEL.get(m, m)})" for m in modes)
    )
    lines.append(
        "- Each section = one fragment. The captions below come from the SAME model "
        "given the SAME series three ways. `image` sees only the chart; `text` sees "
        "only the numbers; `both` sees both. The raw series at the bottom is ground "
        "truth for checking faithfulness (and is exactly what `text`/`both` were fed)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(rows, 1):
        gt = r["groundtruth"]
        caps = r["captions"]
        lines.append(f"## {i}. fragment id={r['id']}  (len={gt['len']})")
        lines.append("")
        lines.append(f"![fragment {r['id']}]({r['image_path']})")
        lines.append("")
        for m in modes:
            if m in caps:
                lines.append(f"**[{m}] {MODE_LABEL.get(m, m)}:** {caps[m]}")
                lines.append("")
        lines.append(
            f"*ground truth (reviewer):* range "
            f"[{gt['min']}, {gt['max']}], min@t={gt['argmin']}, max@t={gt['argmax']}, "
            f"first={gt['first']} → last={gt['last']}"
        )
        lines.append("")
        lines.append("<details><summary>raw series</summary>")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps([round(v, 4) for v in r["series"]]))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_modes(s: str) -> list[str]:
    modes = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown mode(s): {bad}; valid = {list(MODES)}")
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate VLM captions for TS fragments (3 settings).")
    ap.add_argument("--limit", type=int, default=100, help="number of unique fragments")
    ap.add_argument("--out-dir", default="out_cap", help="output directory")
    ap.add_argument("--modes", default="image,text,both", help="comma list of input settings")
    ap.add_argument("--fragments", default=None,
                    help="reuse a pre-rendered fragments.jsonl (id/image_path/series); "
                         "no re-render. When set, --source/--local-path/--limit are ignored.")
    ap.add_argument("--image-root", default=None,
                    help="directory the manifest's image_path is relative to (where the "
                         "PNGs actually live). Default: the --fragments file's directory. "
                         "Lets many models SHARE one render dir; md links are rewritten "
                         "relative to --out-dir so they still resolve.")
    ap.add_argument("--source", default="WinfredGe/TSFragment-600K", help="HF dataset id")
    ap.add_argument("--local-path", default=None, help="local CSV/parquet mirror (skips HF)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemma-3 image-text HF model id")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="fragments captioned together per forward pass (per setting). "
                         "Lower it if you hit OOM; 1 = no batching.")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--dtype", default="auto", help="torch dtype (auto/bfloat16/float16)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dpi", type=int, default=100, help="render DPI")
    ap.add_argument("--resume", action="store_true", help="skip ids already in captions.jsonl")
    args = ap.parse_args()

    modes = _parse_modes(args.modes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "captions.jsonl"
    md_path = out_dir / "captions.md"

    if args.image_root:
        image_root = Path(args.image_root)
    elif args.fragments:
        image_root = Path(args.fragments).parent
    else:
        image_root = out_dir

    cfg = RenderConfig(dpi=args.dpi)

    done = _load_done_ids(jsonl_path) if args.resume else set()
    if done:
        print(f"[resume] {len(done)} fragments already captioned, skipping them.")

    print(f"[load] loading model: {args.model} ...")
    client = GemmaVLLocal(
        model_id=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.fragments:
        with Path(args.fragments).open(encoding="utf-8") as f:
            src_rows = [json.loads(line) for line in f if line.strip()]
        items = [
            (int(r["id"]), int(r.get("len") or len(r["series"])),
             r["series"], r["image_path"], False)
            for r in src_rows
        ]
        print(f"[load] {len(items)} fragments from manifest {args.fragments}; settings = {modes}")
    else:
        fragments = list(load_fragments(
            args.source, limit=args.limit, local_path=args.local_path
        ))
        items = [
            (frag.sample_id, frag.length, frag.series,
             f"render/{frag.sample_id:08d}.png", True)
            for frag in fragments
        ]
        print(f"[load] {len(items)} unique fragments; settings = {modes}")

    todo = [it for it in items if it[0] not in done]
    if not todo:
        print("[caption] nothing to do (all fragments already captioned).")
    bs = max(1, args.batch_size)

    with jsonl_path.open("a", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(todo), bs), desc="caption(batch)"):
            chunk = todo[start:start + bs]

            img_abs = {}
            md_link = {}
            for sample_id, length, series, img_rel, need_render in chunk:
                p = image_root / img_rel
                if need_render:
                    render_fragment(series, out_path=p, config=cfg)
                img_abs[sample_id] = p
                md_link[sample_id] = os.path.relpath(p, out_dir)

            caps_by_mode: dict[str, list[str]] = {}
            for m in modes:
                prompts = [build_prompt(m, series=s) for _, _, s, _, _ in chunk]
                if m in ("image", "both"):
                    image_paths = [img_abs[sid] for sid, *_ in chunk]
                else:
                    image_paths = [None] * len(chunk)
                caps_by_mode[m] = client.generate_batch(
                    prompts, image_paths, system=CAPTION_SYSTEM
                )

            for j, (sample_id, length, series, img_rel, _) in enumerate(chunk):
                row = {
                    "id": sample_id,
                    "len": length,
                    "image_path": md_link[sample_id],
                    "captions": {m: caps_by_mode[m][j] for m in modes},
                    "groundtruth": _series_groundtruth(series),
                    "series": series,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    _write_markdown(rows, md_path, args.model, modes)

    print(f"[done] {len(rows)} fragments x {len(modes)} settings -> {jsonl_path}")
    print(f"[done] review doc -> {md_path}")


if __name__ == "__main__":
    main()
