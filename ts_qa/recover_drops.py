from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

_IMG_RE = re.compile(r"^##\s+(.+?)\s+\(\d+/\d+\s+survive\)\s*$")
_MARK_RE = re.compile(r"^- .*\*\*(KEEP|CORRECT|DROP)\*\*\s+\[(.*?)/(.*?)\]\s+.*?—\s*(.*)$")
_ERR_DEC_RE = re.compile(r"\(\d+/\d+\)\s+\[(KEEP|CORRECT|DROP)\].*?\|\s*(.*)$")

AGENT_NONE_REASON = "non-computable: agent->none"
STEM_KEY_LEN = 60


def _shard_of(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"shard(\d+)", base) or re.search(r"_(\d+)\.err$", base)
    return int(m.group(1)) if m else None


def _stem_key(stem: str) -> str:
    return stem[:STEM_KEY_LEN].strip()


def parse_report(path: str) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    img: Optional[str] = None
    qi = 0
    pending: Optional[bool] = None
    with open(path) as fh:
        for line in fh:
            m_img = _IMG_RE.match(line)
            if m_img:
                img, qi, pending = m_img.group(1), 0, None
                continue
            m_mark = _MARK_RE.match(line)
            if m_mark:
                dec, reason = m_mark.group(1), m_mark.group(4)
                pending = dec == "DROP" and reason.strip() == AGENT_NONE_REASON
                pending_qi = qi
                qi += 1
                if pending:
                    pending = (img, pending_qi)
                continue
            if pending and line.startswith("    - "):
                stem = line[len("    - "):].rstrip("\n")
                img_p, q = pending
                out.append((img_p, q, stem))
                pending = None
    return out


def err_timeout_stems(path: str) -> Set[str]:
    keys: Set[str] = set()
    armed = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Request timed out" in line:
                armed = True
                continue
            if armed:
                m = _ERR_DEC_RE.search(line)
                if m:
                    keys.add(_stem_key(m.group(2)))
                    armed = False
    return keys


def trace_timeouts(path: str) -> Set[Tuple[str, int]]:
    out: Set[Tuple[str, int]] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            err = (r.get("error") or "")
            if r.get("decision") == "DROP" and "timed out" in err.lower():
                out.add((r.get("image_path"), r.get("qi")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted", required=True, help="qa_extracted.jsonl (full bank)")
    p.add_argument("--reports", required=True, help="glob for report_shard*.md")
    p.add_argument("--errlogs", default=None, help="glob for verify10k-*_*.err (timeout evidence)")
    p.add_argument("--traces", default=None, help="glob for trace_shard*.jsonl (exact timeout, preferred)")
    p.add_argument("--mode", choices=["timeout-evidence", "agent-none"], default="timeout-evidence",
                   help="timeout-evidence: only agent->none DROPs with timeout evidence (default); "
                        "agent-none: the whole agent->none bucket")
    p.add_argument("--out", required=True, help="output jsonl (qa_extracted schema, filtered qa_list)")
    p.add_argument("--stats-only", action="store_true", help="print counts, do not write --out")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    by_img: Dict[str, dict] = {}
    for line in open(args.extracted):
        if line.strip():
            r = json.loads(line)
            by_img[r["image_path"]] = r
    logging.info("loaded %d images from %s", len(by_img), args.extracted)

    reports = sorted(glob.glob(args.reports))
    errlogs = {(_shard_of(x)): x for x in glob.glob(args.errlogs or "")}
    traces = {(_shard_of(x)): x for x in glob.glob(args.traces or "")}
    errlogs.pop(None, None)
    traces.pop(None, None)

    err_keys: Dict[int, Set[str]] = {}
    trc_keys: Dict[int, Set[Tuple[str, int]]] = {}

    selected: Dict[str, List[int]] = defaultdict(list)
    n_none = n_timeout = n_no_evidence = 0

    for rep in reports:
        shard = _shard_of(rep)
        nones = parse_report(rep)
        n_none += len(nones)
        if not nones:
            continue
        have_trace = shard in traces
        if have_trace and shard not in trc_keys:
            trc_keys[shard] = trace_timeouts(traces[shard])
        if not have_trace and shard in errlogs and shard not in err_keys:
            err_keys[shard] = err_timeout_stems(errlogs[shard])
        src = "trace" if have_trace else ("err" if shard in errlogs else "NONE")
        src_path = traces.get(shard) if have_trace else errlogs.get(shard)

        sel_shard = 0
        for img, qi, stem in nones:
            if args.mode == "agent-none":
                is_to = True
            elif have_trace:
                is_to = (img, qi) in trc_keys[shard]
            elif shard in err_keys:
                is_to = _stem_key(stem) in err_keys[shard]
            else:
                is_to = False
                n_no_evidence += 1
            if is_to:
                n_timeout += 1
                sel_shard += 1
                if img in by_img:
                    selected[img].append(qi)
                else:
                    logging.warning("image in report but not in extracted: %s", img)
        logging.info("shard %s: agent->none=%d  evidence=%s (%s)  -> selected=%d",
                     shard, len(nones), src, os.path.basename(src_path) if src_path else "-", sel_shard)

    n_q = sum(len(v) for v in selected.values())
    logging.info("agent->none DROPs (report): %d", n_none)
    logging.info("selected (%s): %d questions across %d images", args.mode, n_q, len(selected))
    if n_no_evidence:
        logging.warning("%d agent->none had NO evidence source (shard missing .err/trace) -> excluded",
                        n_no_evidence)
    logging.info("mode=%s  -> %d recoverable questions", args.mode, n_timeout)

    if args.stats_only:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = 0
    with open(args.out, "w") as out:
        for img in by_img:
            qis = selected.get(img)
            if not qis:
                continue
            rec = by_img[img]
            qa_list = []
            for qi in sorted(set(qis)):
                qa = dict(rec["qa_list"][qi])
                qa["recover_src_qi"] = qi
                qa_list.append(qa)
            out.write(json.dumps({**{k: rec[k] for k in ("id", "image_path", "series", "source")
                                     if k in rec}, "qa_list": qa_list}, ensure_ascii=False) + "\n")
            n_written += 1
    logging.info("wrote %d images (%d questions) -> %s", n_written, n_q, args.out)


if __name__ == "__main__":
    main()
