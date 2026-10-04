# Contamination report

Scanned **15** training documents (592 tokens, 15 windows) against **12** eval items in 0.012s.

| status | items |
|---|---|
| contaminated | 8 |
| suspicious | 0 |
| clean | 4 |

Contamination rate **66.7%**, flag rate 66.7%; 8 training document(s) implicated.

Flagged by method: ngram 4, minhash 4, embedding 8, judge 4 (n=13, shingle=3, embedder=hashed).

LLM judge `deepseek-flash`: 4 call(s), 4 YES / 0 NO / 0 unparsed / 0 error(s); 620 in + 128 out tokens, $0.0003.

## Flagged items

| id | status | methods | n-gram overlap | longest span | containment | cosine | docs |
|---|---|---|---|---|---|---|---|
| q01-exact | **contaminated** | ngram, minhash, embedding | 100% | 38 | 1.00 | 0.75 | web-0001 |
| q07-chat | **contaminated** | ngram, minhash, embedding | 100% | 35 | 0.97 | 0.96 | chat-0007 |
| q02-exact | **contaminated** | ngram, minhash, embedding | 62% | 27 | 0.92 | 0.90 | web-0002 |
| q06-partial | **contaminated** | ngram, embedding, judge | 33% | 22 | 0.45 | 0.50 | web-0006 |
| q03-edited | **contaminated** | minhash, embedding | 0% | 0 | 0.67 | 0.80 | web-0003 |
| q12-short | **contaminated** | embedding, judge | 0% | 0 | 0.00 | 0.51 | web-0014 |
| q05-paraphrase | **contaminated** | embedding, judge | 0% | 0 | 0.00 | 0.48 | web-0005 |
| q04-paraphrase | **contaminated** | embedding, judge | 0% | 0 | 0.00 | 0.46 | web-0004 |

## Reasons

- **q01-exact** (contaminated)
  - [contaminated] 100% of 13-grams found in training (1 doc(s)); longest common span 38 words
  - [contaminated] MinHash containment 1.00 in web-0001
  - [suspicious] embedding cosine 0.75 with web-0001
  - evidence: "practice set for railway aptitude exams a train leaves pune at 9 40 and travels 312 kilometres at an average speed of 78 kilometres per hour at what time does …"
- **q07-chat** (contaminated)
  - [contaminated] 100% of 13-grams found in training (1 doc(s)); longest common span 35 words
  - [contaminated] MinHash containment 0.97 in chat-0007
  - [contaminated] embedding cosine 0.96 with chat-0007
  - evidence: "you are a helpful sql tutor write a sql query that returns the second highest salary from an employees table returning null when there is no second distinct sa…"
- **q02-exact** (contaminated)
  - [contaminated] 62% of 13-grams found in training (1 doc(s)); longest common span 27 words
  - [contaminated] MinHash containment 0.92 in web-0002
  - [contaminated] embedding cosine 0.90 with web-0002
  - evidence: "forum thread why does the rust borrow checker reject a program that holds a mutable reference to a vector while iterating over the same vector with an immutabl…"
- **q06-partial** (contaminated)
  - [suspicious] partial overlap: 10/30 13-grams (33%), longest common span 22 words
  - [suspicious] embedding cosine 0.50 with web-0006
  - [contaminated] LLM judge (deepseek-flash) confirmed the leak
  - judge: YES The training passage contains the same clinical answer nearly verbatim, including the CYP3A4 inhibition, INR elevation, and three-to-five-day monitoring guidance.
  - evidence: "pharmacology bulletin on macrolide interactions clarithromycin inhibits cyp3a4 and raises the inr so the inr should be checked within three to five days of sta…"
- **q03-edited** (contaminated)
  - [suspicious] MinHash containment 0.67 in web-0003
  - [contaminated] embedding cosine 0.80 with web-0003
  - evidence: "study notes in a hash table that uses linear probing what happens to lookup speed as the load factor approaches one and why do primary clusters appear probe se…"
- **q12-short** (contaminated)
  - [suspicious] embedding cosine 0.51 with web-0014
  - [note] item shorter than 8 tokens; n-gram check skipped
  - [contaminated] LLM judge (deepseek-flash) confirmed the leak
  - judge: YES The passage explicitly states "the capital of peru is lima," directly matching the eval item's question and answer.
  - evidence: "geography quiz answers the capital of peru is lima the capital of chile is santiago the capital of kenya is nairobi the capital of canada is ottawa"
- **q05-paraphrase** (contaminated)
  - [suspicious] embedding cosine 0.48 with web-0005
  - [contaminated] LLM judge (deepseek-flash) confirmed the leak
  - judge: YES The training passage conveys the same core answer—C4 plants concentrate CO2 around RuBisCO in bundle sheath cells, suppressing photorespiration when stomata close to save water—in paraphrased form.
  - evidence: "botany lecture summary under heat and drought c4 species outperform c3 species because they pump carbon dioxide into bundle sheath cells concentrating it aroun…"
- **q04-paraphrase** (contaminated)
  - [suspicious] embedding cosine 0.46 with web-0004
  - [contaminated] LLM judge (deepseek-flash) confirmed the leak
  - judge: YES The passage contains the same answer—Canberra as a purpose-built compromise between Sydney and Melbourne—in paraphrased form.
  - evidence: "travel guide many visitors assume sydney is the national capital yet the seat of government is canberra the city was designed from scratch as a compromise beca…"
