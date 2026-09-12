from __future__ import annotations

import argparse
import json
import time
import os
import multiprocessing as mp
import pathlib
from pathlib import Path

from tqdm import tqdm

from .data import Fragment, load_fragments, synthetic_fragments
from .render import RenderConfig, render_fragment


def _render_task(task):
    sample_id, series, length, caption, img_path, rel_path, cfg = task
    render_fragment(series, out_path=img_path, config=cfg)
    return {
        "id": sample_id,
        "image_path": rel_path,
        "series": series,
        "len": length,
        "ds_caption": caption,
    }


def _load_exclude_ids(paths) -> set:
    ids = set()
    for path in paths or []:
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["id"])
    return ids


def _load_include_ids(path) -> set:
    if not path:
        return set()
    text = pathlib.Path(path).read_text(encoding="utf-8").strip()
    if path.endswith(".json"):
        obj = json.loads(text)
        return set(obj["indices"] if isinstance(obj, dict) else obj)
    ids = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)["id"] if line.startswith("{") else int(line))
    return ids


def _iter_fragments(args, exclude_ids: set, include_ids: set) -> "tuple[object, int | None]":
    if args.synthetic is not None:
        return synthetic_fragments(n=args.synthetic, seed=args.seed), args.synthetic
    limit = args.limit + len(exclude_ids) if (args.limit and exclude_ids) else args.limit
    if include_ids:
        limit = None
    frags = load_fragments(
        source=args.source,
        split=args.split,
        limit=limit,
        local_path=args.local_path,
        shuffle=args.shuffle,
        seed=args.sample_seed,
        shuffle_buffer=args.shuffle_buffer,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
    return frags, args.limit


def main() -> None:
    p = argparse.ArgumentParser(description="Render TSFragment-600K fragments to PNGs.")
    src = p.add_argument_group("data source")
    src.add_argument("--source", default="WinfredGe/TSFragment-600K", help="HF dataset id")
    src.add_argument("--split", default="train")
    src.add_argument("--local-path", default=None, help="local csv/parquet/jsonl mirror")
    src.add_argument("--limit", type=int, default=None, help="max unique fragments")
    src.add_argument("--synthetic", type=int, default=None,
                     help="render N synthetic fragments instead (offline smoke test)")
    src.add_argument("--seed", type=int, default=0)
    src.add_argument("--shuffle", action="store_true",
                     help="randomly sample fragments (TSFragment-600K is sliding-window: "
                          "the first N are near-duplicate shifts; shuffle to decorrelate)")
    src.add_argument("--sample-seed", type=int, default=0, help="RNG seed for --shuffle")
    src.add_argument("--shuffle-buffer", type=int, default=10000,
                     help="streaming-only shuffle buffer (larger = more spread, slower "
                          "first yield); ignored unless --streaming")
    src.add_argument("--streaming", action="store_true",
                     help="stream rows from the hub instead of downloading the whole split "
                          "first (default: download, then shuffle/sample in memory — no "
                          "buffer-fill wait)")
    src.add_argument("--cache-dir", default=None,
                     help="where `datasets` caches the download (default: HF default "
                          "~/.cache/huggingface)")
    src.add_argument("--include-ids", default=None,
                     help="render ONLY these ids, from an indices.json / indices.txt / "
                          "fragments.jsonl. Use this to reproduce an exact dataset on "
                          "another machine instead of relying on --sample-seed, which "
                          "depends on the datasets library's shuffle staying identical.")
    src.add_argument("--exclude-manifest", action="append", default=None,
                     help="fragments.jsonl whose ids must NOT be sampled (repeatable); "
                          "e.g. the RL training manifest, to keep eval data disjoint")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", required=True, help="output root")
    out.add_argument("--img-subdir", default="render", help="PNG subdir under out-dir")
    out.add_argument("--manifest", default="fragments.jsonl")

    style = p.add_argument_group("render style")
    style.add_argument("--dpi", type=int, default=RenderConfig.dpi)
    style.add_argument("--width", type=float, default=RenderConfig.width_in)
    style.add_argument("--height", type=float, default=RenderConfig.height_in)
    style.add_argument("--no-markers", action="store_true")
    style.add_argument("--no-grid", action="store_true")
    style.add_argument("--n-xticks", type=int, default=RenderConfig.n_xticks)
    style.add_argument("--n-yticks", type=int, default=RenderConfig.n_yticks)

    perf = p.add_argument_group("performance")
    perf.add_argument("--num-workers", type=int, default=8,
                      help="parallel render processes (0 = all cores). Rendering 10k "
                           "fragments is CPU-bound matplotlib and scales nearly linearly; "
                           "1 restores the old serial path.")
    perf.add_argument("--chunk-size", type=int, default=16,
                      help="fragments handed to a worker at a time")
    args = p.parse_args()

    cfg = RenderConfig(
        width_in=args.width,
        height_in=args.height,
        dpi=args.dpi,
        show_markers=not args.no_markers,
        grid=not args.no_grid,
        n_xticks=args.n_xticks,
        n_yticks=args.n_yticks,
    )

    out_root = Path(args.out_dir)
    img_dir = out_root / args.img_subdir
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / args.manifest

    exclude_ids = _load_exclude_ids(args.exclude_manifest)
    include_ids = _load_include_ids(args.include_ids)
    if include_ids:
        print(f"reproducing an exact id list: {len(include_ids)} ids from {args.include_ids}")
    frags, total = _iter_fragments(args, exclude_ids, include_ids)
    if include_ids:
        total = len(include_ids)

    def _tasks():
        emitted = 0
        for frag in frags:
            frag: Fragment
            if frag.sample_id in exclude_ids:
                continue
            if include_ids and frag.sample_id not in include_ids:
                continue
            if total is not None and emitted >= total:
                break
            img_path = img_dir / f"{frag.sample_id:08d}.png"
            yield (
                frag.sample_id,
                frag.series,
                frag.length,
                frag.caption,
                str(img_path),
                str(img_path.relative_to(out_root)),
                cfg,
            )
            emitted += 1

    workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, os.cpu_count() or 1))

    n = 0
    t0 = time.time()
    with manifest_path.open("w", encoding="utf-8") as mf:
        if workers > 1:
            print(f"rendering with {workers} processes")
            with mp.Pool(workers) as pool:
                for rec in tqdm(
                    pool.imap(_render_task, _tasks(), chunksize=args.chunk_size),
                    total=total,
                    desc="rendering",
                ):
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        else:
            for task in tqdm(_tasks(), total=total, desc="rendering"):
                mf.write(json.dumps(_render_task(task), ensure_ascii=False) + "\n")
                n += 1

    dt = time.time() - t0
    print(f"Rendered {n} fragments -> {img_dir}  ({dt:.0f}s, {n/max(dt,1e-9):.1f}/s, {workers} proc)")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
