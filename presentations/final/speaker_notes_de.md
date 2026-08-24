# Speaker Notes — Capstone I

---

## Slide 1 — Evaluation Design

We evaluated the system in two stages.

The first is a standard benchmark evaluation using PubMedQA — a well-known dataset with a thousand questions, each answered with yes, no, or maybe, scored automatically. It gives us a clean baseline, though it's a bit artificial for our use case, since real clinical questions are much more open-ended than that.

That's why we added a second stage: eight custom clinical questions that we designed ourselves, spread across the different difficulty levels of our corpus. Here the answers are full paragraphs, and we had two independent AI judges — ChatGPT and Claude — rate each one from one to five.

---

## Slide 2 — Coverage Map

This slide shows how well our corpus actually covers the kinds of questions we're asking.

The blue-grey bubbles are the research topic clusters in the dataset — oncology, COVID-19, epidemiology, and so on. The green dots are our eight test questions, and the colour tells you how well each question is matched by the corpus. Dark green means the topic is well represented. The lighter the dot, the less material the system has to work with. Two questions fall completely outside — there's barely anything in the corpus that matches them.

This distinction matters a lot for the results.

---

## Slide 3 — Results

Three iterations, one clear finding.

Iteration two wins with a score of 3.6 out of 5. That's our hybrid system, combining keyword and semantic search. Iteration one, the baseline with keyword search only, scores 3.1. And iteration three — the large 70-billion parameter model — comes in at 3.2, barely better than the baseline.

The heatmap on the right explains why. I3 only pulls ahead in the well-covered zones, where the corpus is dense. In sparse or outside-corpus zones, it actually drops below the baseline. So a bigger model doesn't help if the right articles aren't in the corpus to begin with. The bottleneck is the data, not the model.

---

## Demo — Live Example

Let me show you what the system actually produces.

The first question is from a well-covered area of the corpus — oncology. I'm asking: "Why do some patients with non-small cell lung cancer stop responding to checkpoint inhibitors?"

The system retrieves five relevant papers and generates this answer: it explains that resistance is driven by tumor heterogeneity, the presence of competing mutations like EGFR or ALK, and immune-related side effects that force treatment to stop. The sources are cited directly. This scored 4.5 out of 5 — the judges found it accurate and well-supported.

Now let me show the contrast. This second question is outside the corpus — ophthalmology. "What outcomes can be expected after corneal transplantation for keratoconus?"

The system still retrieves five papers and generates an answer, but notice what's happening: the sources don't really fit the question. It pulls in a global graft failure survey, an orthokeratology study, a genetics paper. The answer is vague and hedged throughout. This scored 2 out of 5. The system didn't fail gracefully — it just gave a confident-sounding answer with the wrong material. That's exactly the problem the coverage map on the previous slide was showing.

---

## Slide 4 — Outlook

Three directions from here.

First, expand the corpus. Two of our eight questions scored low not because the model is weak, but simply because those topics aren't in the dataset. Adding targeted literature for the underrepresented areas would directly raise the floor — no model changes needed.

Second, human expert evaluation. The AI judges gave us a useful signal, but what we really want to know is whether a clinician would find these answers trustworthy enough to act on. That's the next validation step.

Third, possible services. The architecture is modular and already production-ready. The retrieval pipeline, the ranking, the generation — all of it can be adapted. Potential directions include clinical decision support, structured literature review, and specialty-specific Q&A tooling.
