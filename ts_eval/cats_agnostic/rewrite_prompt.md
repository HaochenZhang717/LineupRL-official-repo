The instruction given to the rewriting LLM, verbatim. `{{START}}`, `{{END}}`, `{{K}}` and
`{{NNN}}` identify the slice of the items file that one call rewrites; everything else
was identical across the 30 slices.
---

Each line of your slice is {"id": "...", "gold": "<caption>"}. The caption is your only input: there is no chart, no values and no metadata, and you must not go looking for any.

Rewrite each caption so it describes only what an unlabelled line chart could show. Take out the domain: the phenomenon or attribute name, any place, port, store or product name, units and currency (keep the bare number), comparisons to statistics outside this series (historical mean, all-time max, "the reference period"), and any explanation that needs to know what the data is.

Dates become relative positions, never counted ones. "From 2011 to 2018" becomes "across the series"; "peaking in August 2023" becomes "peaking about two-thirds of the way through"; "a dip in late May" becomes "a dip near the end". Do not say how many points the series has and do not number them — no "the eighth point", no "40 points", no "index 6", and drop counts of time units such as "a five-week snapshot". The text does not tell you how many points there are, and a guess would put a false claim into the caption. The vocabulary is "at the start", "early on", "midway", "in the second half", "near the end", "about two-thirds of the way through".

Keep the same observations in the same order, and keep every value and statistic as written. Rephrase as freely as the removals require, so that what is left reads as fluent, natural prose rather than a sentence with holes in it. What you must not do is add an observation the caption did not make, or change a number, even if the caption looks wrong. If a sentence has nothing left after the removals, drop it.

Your slice: sed -n '{{START}},{{END}}p' <work>/items_textonly.jsonl  ({{K}} lines)
Write <work>/shards/batch_{{NNN}}.jsonl — exactly {{K}} lines, same order, one JSON object per line built with python3 + json.dumps (never by hand), "id" copied unchanged, plus "rewritten". Rewrite them yourself, one by one; a script is only for reading input and writing output. Reply with the shard path and line count, nothing else.
