# Leak recall test

40 eval items; leaks written by `deepseek-flash`, judged by `deepseek-flash`.

Share of eval items flagged via their own leak document (for `negative` every flag is a false positive):

| kind | items | any detector | contaminated | n-gram | MinHash | embedding | own-doc cosine min / median / max | scan + judge -> contaminated | judge YES on every pair |
|---|---|---|---|---|---|---|---|---|---|
| verbatim | 40 | 100% | 100% | 98% | 100% | 100% | 0.52 / 0.90 / 0.93 | 100% | 100% (40/40) |
| light_edit | 40 | 100% | 57% | 50% | 70% | 100% | 0.40 / 0.81 / 0.96 | 100% | 100% (40/40) |
| paraphrase | 40 | 98% | 12% | 0% | 0% | 98% | 0.39 / 0.65 / 0.85 | 98% | 100% (40/40) |
| heavy_paraphrase | 40 | 28% | 0% | 0% | 0% | 28% | 0.07 / 0.31 / 0.72 | 28% | 98% (39/40) |
| answer_only | 40 | 95% | 28% | 10% | 8% | 95% | 0.31 / 0.67 / 0.95 | 95% | 98% (39/40) |
| negative | 40 | 28% | 0% | 0% | 0% | 28% | 0.07 / 0.32 / 0.73 | 2% | 2% (1/40) |

Judge verdicts against the generator's label (YES on a non-leak, NO on a leak):

- `heavy_paraphrase` s20: NO The passage conveys the same underlying concept in abstract, generic wording without using the terms "Big O notation," "algorithm," or the specific phrasing 
- `answer_only` s15: NO The passage conveys the same factual content in different wording, but it is a general textbook explanation rather than a verbatim copy of the eval item or i
- `negative` s24: YES The passage directly states that judicial review was first exercised in Marbury v. Madison, which is the exact answer to the eval item.

API usage: 341 calls, 59,092 in + 10,499 out tokens, $0.0303.
