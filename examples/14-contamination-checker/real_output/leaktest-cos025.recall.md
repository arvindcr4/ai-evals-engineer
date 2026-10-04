# Leak recall test

40 eval items; leaks written by `deepseek-flash`, judged by `deepseek-flash`.

Share of eval items flagged via their own leak document (for `negative` every flag is a false positive):

| kind | items | any detector | contaminated | n-gram | MinHash | embedding | own-doc cosine min / median / max | scan + judge -> contaminated | judge YES on every pair |
|---|---|---|---|---|---|---|---|---|---|
| verbatim | 40 | 100% | 100% | 98% | 100% | 100% | 0.52 / 0.90 / 0.93 | 100% | 100% (40/40) |
| light_edit | 40 | 100% | 57% | 50% | 70% | 100% | 0.40 / 0.81 / 0.96 | 100% | 100% (40/40) |
| paraphrase | 40 | 100% | 12% | 0% | 0% | 100% | 0.39 / 0.65 / 0.85 | 100% | 100% (40/40) |
| heavy_paraphrase | 40 | 72% | 0% | 0% | 0% | 72% | 0.07 / 0.31 / 0.72 | 70% | 95% (38/40) |
| answer_only | 40 | 100% | 28% | 10% | 8% | 100% | 0.31 / 0.67 / 0.95 | 100% | 100% (40/40) |
| negative | 40 | 70% | 0% | 0% | 0% | 70% | 0.07 / 0.32 / 0.73 | 8% | 2% (1/40) |

Judge verdicts against the generator's label (YES on a non-leak, NO on a leak):

- `heavy_paraphrase` s20: NO The passage conveys the same underlying concept in abstract, generic wording without using the terms "Big O notation," "algorithm," or the specific phrasing 
- `heavy_paraphrase` s23: NO The passage conveys the same ideas in different wording, but it does not contain the eval item verbatim or in a form that would allow direct memorization of 
- `negative` s24: YES The passage directly states that judicial review was first exercised in Marbury v. Madison, which is the exact answer to the eval item.

API usage: 381 calls, 66,119 in + 11,794 out tokens, $0.0340.
