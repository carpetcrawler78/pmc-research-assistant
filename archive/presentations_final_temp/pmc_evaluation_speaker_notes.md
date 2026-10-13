# PMC Evaluation — Speaker Notes

---

## Slide 1: Two phases — benchmark to clinical test

**Why this slide:**
We need to explain why standard benchmarks aren't enough. PubMedQA is well-known but only tests yes/no/maybe — our system gives full paragraphs. That needs a different kind of evaluation.

**What you see:**
Two boxes side by side. Left: Phase I — PubMedQA, 1,000 yes/no questions, scored automatically. Right: Phase II — our custom test, 8 real clinical questions, free-text answers, scored by two LLM raters on a 1–5 scale.

**What to say:**
"Phase I is basically a sanity check — does the system work at all? We score 68% on PubMedQA, which is competitive. But yes/no answers don't tell us if a doctor would actually trust the output. That's why we designed Phase II — 8 real questions, free-text answers, much harder to evaluate."

**Conclusion:**
Phase I = works? Phase II = useful?

---

## Slide 2: 8 questions — testing different strengths

**Why this slide:**
Show that the questions aren't random. Each one was chosen to test a different aspect — some the system should handle well, others are deliberately hard.

**What you see:**
8 questions with color-coded dots. Green = core topic (well covered in corpus). Blue = moderate. Orange = borderline. Red = out-of-distribution (not in training data at all).

**What to say:**
"We picked these 8 questions based on three criteria: medical domain, complexity, and how well the topic is covered in our 10,000-article corpus. The colors tell the story — green means the system has lots of relevant articles, red means there's nothing in the training data. Q4 — LRKK2 mutations in Parkinson's — is our stress test. There's no Parkinson cluster in the corpus at all."

**Conclusion:**
We test strengths AND weaknesses on purpose.

---

## Slide 3: How we rate answers — a 5-point scale

**Why this slide:**
Before showing results, the audience needs to understand what the numbers mean and who's rating.

**What you see:**
Scale from 1 (Wrong) to 5 (Excellent) with plain-language descriptions. Below: two LLM raters — ChatGPT and Claude — rating independently.

**What to say:**
"Each answer gets a score from 1 to 5. A 1 means completely wrong — wrong topic or contradicts the evidence. A 5 means the answer is complete, correct, and clinically useful. Two different LLMs rate each answer independently — ChatGPT as Rater 1, Claude as Rater 2. Using two different models from different providers reduces bias. Ideally we'd use a panel of medical experts, but that wasn't feasible for this project."

**Conclusion:**
Two independent raters, clear scale, transparent method.

---

## Slide 4: Where do our 8 questions land in the training data?

**Why this slide:**
Before looking at scores, we need to understand what the system actually knows. The bubble map shows which medical domains are covered and where our questions fall.

**What you see:**
10 medical domain clusters as bubbles — size = share of articles. The 8 questions are placed near their best-matching cluster. Q1 and Q2 sit inside the Oncology bubble (green dots). Q4 floats between clusters (red dot — no matching domain).

**What to say:**
"This map shows our 10,000-article corpus organized into 10 medical domain clusters. The biggest clusters are COVID/Public Health, Oncology, and Epidemiology — about 20% each. Our questions are placed where they best fit. Q1 and Q2 land right inside Oncology — core topics, the system should do well. Q4, the Parkinson question, sits in no-man's land — there's literally no matching cluster. This map predicts our results before we even look at scores."

**Conclusion:**
2 of 8 questions fit well. 6 test the system at its limits.

---

## Slide 5: Which pipeline answers best?

**Why this slide:**
We tested three different pipeline configurations. This shows which combination of search method and model size works best.

**What you see:**
Bar chart with 3 variants. Each shows scores from both raters. V2 (Hybrid search + 8B model) wins with 3.61/5.

**What to say:**
"We tested three variants. V1 uses keyword search only with a small model — 3.12 average. V2 adds semantic search (meaning-based, not just keywords) — jumps to 3.61. V3 uses the same hybrid search but a much bigger model, 70 billion parameters — and it actually scores slightly lower at 3.22. The surprise: bigger model doesn't help. The bottleneck is retrieval — finding the right articles. If the search misses, even the best model can't compensate."

**Conclusion:**
Better search > bigger model. Retrieval is the bottleneck.

---

## Slide 6: Not all questions are equal

**Why this slide:**
The overall average hides big differences between questions. Some score 4+, others below 2.

**What you see:**
Grouped bar chart — 8 questions, 3 bars each (one per pipeline variant). Q1–Q3 consistently strong. Q4 consistently weak across all variants.

**What to say:**
"Look at the spread. Q1 and Q2 — oncology questions, well covered in the corpus — score around 4 out of 5 consistently. Q3, neurology, also strong. But Q4 — LRKK2 and Parkinson's — barely reaches 2 in any variant. The system doesn't fail randomly. It fails predictably on topics it hasn't seen. That's actually good news: we know exactly what to fix."

**Conclusion:**
Failures are predictable = fixable. More data in weak domains solves this.

---

## Slide 7: Closer to the training data = better answers

**Why this slide:**
This is the key insight. We can show a direct relationship between corpus coverage and answer quality.

**What you see:**
Scatter plot. X-axis = distance from question to nearest cluster center. Y-axis = score. Three colored zones: green (in corpus), blue (partial match), red (no match). Clear trend: further away = lower score.

**What to say:**
"This chart brings it all together. On the left — questions close to a training cluster, scoring well. On the right — questions far from any cluster, scoring poorly. Q4 at the bottom right: furthest from any cluster, lowest score. The correlation is clear: when the system fails, it's not because the language model is bad — it's because the retrieval can't find relevant articles. This is a data coverage problem, not a model problem. And data coverage is fixable."

**Conclusion:**
Distance to training data predicts quality. Fix the data, fix the answers.

---

## Slide 8: When answers go wrong — a concrete example

**Why this slide:**
Numbers alone don't show what "wrong" looks like. This makes it immediately understandable.

**What you see:**
A question about negative effects of nivolumab on liver function. The answer talks about positive effects on tumor regression. Same drug, completely wrong context.

**What to say:**
"Here's what a score of 1 actually looks like. The question asks about liver damage from nivolumab — a real clinical concern. The system found an article about nivolumab, so the drug name matches. But the article was about its cancer-fighting effects, not its side effects. The answer confidently talks about positive tumor regression when we asked about liver toxicity. This is the dangerous kind of error — it sounds authoritative, mentions the right drug, but answers the wrong question entirely."

**Conclusion:**
Right drug, wrong context. This is why retrieval quality matters most.

---

## Slide 9: What could come next

**Why this slide:**
End with forward-looking next steps, not a checklist of what's "done."

**What you see:**
Four numbered items: Better Evaluation, Larger Model, More Services, Bigger Corpus.

**What to say:**
"Four things that would make this system significantly better. First — better evaluation. Replace LLM raters with a panel of medical experts for ground truth. Second — try a larger model, though our results suggest retrieval matters more. Third — expand beyond single questions: drug interactions, follow-up conversations, patient-friendly summaries. And fourth, the biggest one — expand from 10,000 to 4 million articles. Most failures we saw were data coverage problems. More data means fewer blind spots. That single step would fix the majority of errors we observed."

**Conclusion:**
More data = biggest impact. The system works — it just needs a bigger knowledge base.
