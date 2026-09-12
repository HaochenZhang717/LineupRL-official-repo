import json, re, os

SP = os.environ.get("CATS_WORK", os.path.dirname(os.path.abspath(__file__)))
CT = os.environ.get("CATS_TEST_DATA", "bench_data/CaTS_Datasets/test_data")
golds = {json.loads(l)["id"]: json.loads(l)["gold"] for l in open(f"{SP}/items_textonly.jsonl")}
refs = {json.loads(l)["id"]: json.loads(l)["rewritten"] for l in open(f"{SP}/cats_agnostic_refs.jsonl")}

M = {m: i+1 for i, m in enumerate(["January","February","March","April","May","June","July",
                                   "August","September","October","November","December"])}
MO = "|".join(M)
SPAN = re.compile(rf"(?:from|between)\s+(?:({MO})\s+)?(\d{{4}})(?:,)?\s*(?:to|and|through|-)\s+(?:({MO})\s+)?(\d{{4}})", re.I)
DATE = re.compile(rf"\b({MO})\s+(?:\d{{1,2}},\s*)?(\d{{4}})", re.I)
YEAR = re.compile(r"\b(19|20)\d\d\b")

def months(mon, yr):
    return int(yr) * 12 + (M[mon.capitalize()] if mon else 1)

def gold_fraction(g, anchor_word):
    s = SPAN.search(g)
    if not s: return None
    a, b = months(s.group(1), s.group(2)), months(s.group(3), s.group(4))
    if b <= a: return None
    m = re.search(rf"(?:{anchor_word})[^.]{{0,80}}?(?:in|on|by|around|at)\s+(?:({MO})\s+)?(?:\d{{1,2}},\s*)?((?:19|20)\d\d)", g, re.I)
    if not m: return None
    p = months(m.group(1), m.group(2))
    return None if not (a <= p <= b) else (p - a) / (b - a)

CLAUSE = r"[^,.;]{0,60}"
EARLY = re.compile(rf"(at the start|early on|initially|at the beginning|near the beginning|first (?:third|quarter))", re.I)
MID   = re.compile(rf"(midway|midpoint|halfway|about half|middle of the series|middle)", re.I)
LATE  = re.compile(rf"(near the end|at the end|toward the end|towards the end|late in the series|final)", re.I)

def phrase_bucket(t, anchor):
    m = re.search(rf"(?:{anchor})", t, re.I)
    if not m: return None
    seg = re.split(r"[,.;]", t[m.end():], 1)[0][:70]
    for pat, b in ((EARLY, "early"), (MID, "mid"), (LATE, "late")):
        if pat.search(seg): return b
    m2 = re.search(r"(\w+)[- ](?:way|quarter)s?\s+(?:of the way\s+)?through", seg, re.I)
    if m2:
        w = m2.group(1).lower()
        return {"two-thirds": "late", "three-quarter": "late", "quarter": "early",
                "one-third": "early", "third": "early", "half": "mid"}.get(w, None)
    return None

def bucket_of(f):
    return "early" if f < 1/3 else ("late" if f > 2/3 else "mid")

agree = disagree = 0; ex = []
for i, t in refs.items():
    for anchor in ("maximum|peak\\w*|highest", "minimum|lowest|trough"):
        gf = gold_fraction(golds[i], anchor)
        pb = phrase_bucket(t, anchor)
        if gf is None or pb is None: continue
        if bucket_of(gf) == pb: agree += 1
        else:
            disagree += 1
            if len(ex) < 8: ex.append((i, anchor.split("|")[0], pb, round(gf, 2)))
print(f"=== rewriter fidelity: position phrase vs the position the GOLD's dates imply")
print(f"    agree {agree}, disagree {disagree}  ({agree/(agree+disagree):.1%} agree, n={agree+disagree})")
for i, a, pb, gf in ex: print(f"    {i}: {a} -> rewrite says {pb}, gold's dates imply {gf:.0%}")

ok = wrong = 0; ex2 = []
for i, g in golds.items():
    gf = gold_fraction(g, "maximum|peak\\w*|highest")
    if gf is None: continue
    v = [float(l) for l in open(f"{CT}/time series/{i}.txt") if l.strip()]
    tf = v.index(max(v)) / (len(v) - 1) if len(v) > 1 else 0.0
    if abs(gf - tf) <= 0.15: ok += 1
    else:
        wrong += 1
        if len(ex2) < 6: ex2.append((i, round(gf, 2), round(tf, 2)))
print(f"\n=== the GOLD itself: does its peak date land where the series actually peaks?")
print(f"    within 15% of the true position: {ok}; off by more: {wrong}  (n={ok+wrong})")
for i, gf, tf in ex2: print(f"    {i}: gold implies {gf:.0%}, true argmax at {tf:.0%}")
