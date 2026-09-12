The instruction given to the rewriting LLM for the QUALITATIVE pass, verbatim.
`{{START}}`, `{{END}}`, `{{K}}` and `{{NNN}}` identify the slice of the items file that one
call rewrites; everything else was identical across the 30 slices.

Why a second pass. The first rewrite (`rewrite_prompt.md` -> `cats_agnostic_refs.jsonl`)
stripped the domain and kept every value verbatim. That is faithful, and it made gen->gt
unusable as a caption-quality score: 99.9% of those references contain an exact number
(median 4 each) against 1.3% of BEDTime's, so a model reading an unlabelled chart cannot
entail them however well it describes the shape, and every chart-reading model landed at
or below the empty-caption floor. This pass removes the numbers so the target is the kind
of qualitative statement BEDTime uses.

Input is the FIRST rewrite, not the original CaTS caption, so the only difference between
the two reference sets is the numerals.

---

Each line of your slice is {"id": "...", "gold": "<caption>"}. The caption is your only input: there is no chart, no values and no metadata, and you must not go looking for any.

Rewrite each caption so that it states the same observations with **no numbers at all**. The target is the register of these examples, which are real captions from another benchmark:

    "line decreases near the end"
    "highest point is in middle"
    "rises in the middle then flattens"

Remove every numeral and every quantity: values ("starting at 91.29"), ranges, means, percentages, counts, and any word that only carries a magnitude ("15%", "approximately 40 units"). Positions in time stay, because they are qualitative: "at the start", "early on", "midway", "in the second half", "near the end", "about two-thirds of the way through".

Replace a quantity with the comparative fact it supports, never with a vaguer number. "Starting at 91.29 and reaching 105.27" becomes "starting low and ending higher". "A notable increase of approximately 15%" becomes "a clear increase". "Rising above 100 along the way" becomes "rising above its starting level along the way" ONLY if the caption itself says what 100 relates to; if it does not, drop that clause. "The mean of the series is 97.3" has no numberless form that says anything about THIS series against others -- drop it.

Keep the same observations in the same order, and keep every non-numeric detail: direction, shape, where features sit, whether movement is smooth or noisy, whether it is steady or abrupt. Do not add an observation the caption did not make. If a sentence is nothing but a quantity, drop the sentence. If dropping leaves the caption empty, write the single most specific shape statement the original supports.

The result must read as fluent prose, not as a sentence with holes in it, and must contain no digits.

Your slice: sed -n '{{START}},{{END}}p' work/cats_qual/items.jsonl  ({{K}} lines)
Write work/cats_qual/shards/batch_{{NNN}}.jsonl -- exactly {{K}} lines, same order, one JSON object per line built with python3 + json.dumps (never by hand), "id" copied unchanged, plus "rewritten". Rewrite them yourself, one by one; a script is only for reading input and writing output. Reply with the shard path and line count, nothing else.
