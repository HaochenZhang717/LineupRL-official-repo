import json, re, os, collections, statistics as st

SP = os.environ.get("CATS_AGNOSTIC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cats"))
CT = os.environ.get("CATS_TEST_DATA", "bench_data/CaTS_Datasets/test_data")
golds = {json.loads(l)["id"]: json.loads(l)["gold"] for l in open(f"{SP}/items_textonly.jsonl")}
refs = [json.loads(l) for l in open(f"{SP}/cats_agnostic_refs.jsonl")]
num = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
MONTHS = r"January|February|March|April|May|June|July|August|September|October|November|December"

ORDINALS = (r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
            r"thirteenth|fourteenth|fifteenth|twentieth|thirtieth|fortieth|fiftieth")
CHECKS = [
    ("point numbering", re.compile(rf"\bindex\s+\d+|\b\d+(?:st|nd|rd|th)\s+point|\b\d+\s+points\b(?!\s+(?:to|toward|towards)\b)|"
                                   rf"\b({ORDINALS})\s+point\b", re.I)),
    ("time-unit count", re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)[-\s]"
                                   r"(?:week|month|year|day|hour)\b", re.I)),
    ("year",            re.compile(r"\b(18|19|20)\d\d\b")),
    ("month/date",      re.compile(rf"\b({MONTHS})\b|\d{{4}}-\d{{2}}-\d{{2}}")),
    ("unit/currency",   re.compile(r"[$£€]|\bGBP\b|\bUSD\b|million metric tons|mmHg|\bppm\b|\bcases\b|\btons\b|\bvehicles\b")),
    ("external stat",   re.compile(r"historical|all-time|all time|longer-term|broader mean|reference period", re.I)),
    ("frequency word",  re.compile(r"\b(hourly|daily|weekly|monthly|quarterly|annual|annually|yearly|years?|weeks?|months?)\b", re.I)),
]
hits = collections.defaultdict(list)
for r in refs:
    t, g = r["rewritten"], golds[r["id"]]
    gn = {x.replace(",", "") for x in num.findall(g)}
    for x in num.findall(t):
        if x.replace(",", "") not in gn:
            hits["number not in gold"].append((r["id"], x)); break
    for name, pat in CHECKS:
        m = pat.search(t)
        if m:
            if name == "year" and not re.search(rf"(?:in|from|to|by|since|during|of|,)\s+{m.group(0)}(?![\d.])", g):
                continue
            hits[name].append((r["id"], m.group(0)))

print("=== forbidden content")
for name in ["point numbering", "time-unit count", "number not in gold", "year", "month/date",
             "unit/currency", "external stat", "frequency word"]:
    h = hits[name]
    ex = "; ".join(f"{i} [{v}]" for i, v in h[:3])
    print(f"  {name:20s}{len(h):5d}   {ex[:92]}")
flagged = sorted({i for v in hits.values() for i, _ in v})
print(f"  -> {len(flagged)} of {len(refs)} captions flagged ({len(flagged)/len(refs):.2%})")

EARLY = re.compile(r"\b(at the start|early on|initially|at the beginning|in the first half)\b", re.I)
LATE  = re.compile(r"\b(near the end|at the end|late in the series|in the second half|toward the end|towards the end)\b", re.I)
PEAK  = re.compile(r"(maximum|peak|peaks|peaking|highest)", re.I)
ok = wrong = 0; ex = []
for r in refs:
    t = r["rewritten"]
    m = PEAK.search(t)
    if not m: continue
    tail = t[m.end():m.end() + 90]
    where = "late" if LATE.search(tail) else ("early" if EARLY.search(tail) else None)
    if not where: continue
    v = [float(l) for l in open(f"{CT}/time series/{r['id']}.txt") if l.strip()]
    frac = v.index(max(v)) / (len(v) - 1) if len(v) > 1 else 0.0
    good = frac >= 0.5 if where == "late" else frac <= 0.5
    if good: ok += 1
    else:
        wrong += 1
        if len(ex) < 6: ex.append((r["id"], where, round(frac, 2)))
print(f"\n=== 'peak early/late' claims vs the real argmax: {ok} consistent, {wrong} not (n={ok+wrong})")
for i, w, f in ex: print(f"  {i}: says {w}, argmax at {f:.0%} of the series")

L = [len(r["rewritten"]) for r in refs]; G = [len(golds[r["id"]]) for r in refs]
print(f"\n=== length: gold mean {round(st.mean(G))} -> rewritten mean {round(st.mean(L))} "
      f"({st.mean(L)/st.mean(G):.0%}), median {int(st.median(L))}, min {min(L)}, max {max(L)}")
json.dump({"flagged": flagged}, open(f"{SP}/audit_v3_flagged.json", "w"), indent=1)
