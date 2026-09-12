import json, os
SP = os.environ.get("CATS_WORK", os.path.dirname(os.path.abspath(__file__)))
ids = [json.loads(l)["id"] for l in open(f"{SP}/items_textonly.jsonl")]
K, out, problems = 100, [], []
for b in range((len(ids) + K - 1) // K):
    exp = ids[b*K:(b+1)*K]
    try:
        rows = [json.loads(l) for l in open(f"{SP}/shards/batch_{b:03d}.jsonl") if l.strip()]
    except Exception as e:
        problems.append(f"batch {b}: {e}"); continue
    got = [r.get("id") for r in rows]
    if got != exp:
        i = next((j for j, (a, c) in enumerate(zip(got, exp)) if a != c), None)
        problems.append(f"batch {b}: ids differ (n={len(got)} vs {len(exp)}, first mismatch {i})")
        continue
    for r in rows:
        if not r.get("rewritten", "").strip():
            problems.append(f"{r['id']}: empty rewritten")
    out.extend(rows)
print(f"joined {len(out)} records, {len(problems)} problems")
for p in problems[:20]: print("  !", p)
assert [r["id"] for r in out] == ids, "ORDER BROKEN"
print("order check: ids match items_textonly.jsonl exactly, in order  OK")
with open(f"{SP}/cats_agnostic_refs.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({"id": r["id"], "rewritten": r["rewritten"].strip()}, ensure_ascii=False) + "\n")
print("wrote cats_agnostic_refs.jsonl")
