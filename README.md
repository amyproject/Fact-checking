# Fact-checking claims in short videos

**TL;DR.** Given the transcript of a short video, the system pulls out the checkable claims, retrieves sources, and labels each claim `supported`, `contradicted` or `unverified`, with a quoted sentence as evidence. We test it on 65 claims that professional fact-checkers already labeled, drawn from 38 YouTube clips. Four open models (DeepSeek V4.1 Flash, Qwen3.7 Plus, Kimi K3, GLM 5.3) and three frontier models (Claude Opus 5.5, GPT-6.1 Terra Pro, Gemini 4 Argon) each run off the shelf and with the final harness. GLM 5.3 also runs a full ablation of five harnesses and a LoRA fine-tune. The headline metric is **source-backed accuracy**: the label is right *and* a cited sentence really says so.

> **Status:** the pipeline, test set, trap tags, fine-tuning data, 50-item rubric and reports are built and tested end to end with a mock model (`results_mock/`, fake numbers). Real numbers need `FIREWORKS_API_KEY` and `OPENROUTER_API_KEY` in `.env`; see [Run order](#run-order). Until then, `results/` is empty on purpose.
>
> **The submission report is `results/submission.html`** (`results_mock/submission.html` for the mock run). It has seven tabs: the Forge-style write-up, 27 live charts, a page per claim (why it was selected, its traps, the fact-checker's research, every system's answer with all 50 rubric checks), every video source (all 75, kept or dropped, with each claim's fate), the rubric, the improvement ideas with the fine-tuning recipe, and a demo plan.

## Step 1: The task and why it matters

**The job.** A trust-and-safety analyst or newsroom fact-checker watches a viral clip, writes down what was claimed, searches for evidence, and files a verdict with sources. The search-and-cite part is the slow part, and it's the part this project automates. The analyst still makes the call.

**Who does it.**
- Platform trust-and-safety and misinformation teams.
- Signatories of the International Fact-Checking Network.
- Newsroom verification desks.

Short video is where a growing share of viral claims now starts. Claims there are spoken, not written, so they have to come out of a transcript before anyone can check them.

**Why a model hasn't solved it.** A model answering from memory has two problems:
1. Its knowledge stops at a cutoff, while claims refer to a specific date.
2. It can't show its work. Our closed-book baseline is allowed to cite URLs from memory, and the scorer fetches every one to check that the quoted sentence is really on the page.

**Why an open model.** Thousands of claims a day go through a cheap model, the transcripts and verdicts stay on infrastructure you control, and the same weights can be fine-tuned later (see [Next steps](#next-steps)).

**Value, with the assumptions written out.** Suppose an analyst spends ~20 minutes finding and quoting sources per claim, and the system's evidence list cuts that to ~5 minutes on claims it gets right. At 30 claims a day, that's ~7.5 hours saved per analyst per week. Swap in real numbers from a fact-checking team before quoting this.

## Step 2: The test set

**Source.** [AVeriTeC](https://fever.ai/dataset/averitec.html) (Schlichtkrull et al., NeurIPS 2023 Datasets & Benchmarks). Its real-world claims come with labels and written justifications from professional fact-checking articles. `01_build_testset.py` keeps only claims whose original post was a video, then works from transcripts, not raw video:

| Step | Claims |
|---|---|
| AVeriTeC train + dev claims whose source is a video URL | 166 (75 videos) |
| − not YouTube (TikTok, Facebook, etc.: would need speech-to-text) | −26 |
| − YouTube link without a video ID | −2 |
| − no transcript (unplayable, unavailable, captions off, not English, IP-blocked) | −61 |
| − claim not found in the transcript (<35% of its content words in any 90 s window) | −12 |
| **Kept** | **65 claims, 38 clips, 28 videos** |

**Short clips from long videos.** Only 5 of the 38 source videos are ≤3 minutes. Every longer video is cut to a ~90 s window around each claim, so each test case looks like a short video. The window comes from the URL's `?t=` timestamp (20 claims) or from the best lexical match against the transcript (39). Videos ≤3 minutes are used whole (6 claims). The median clip is 90 s.

**Labels.** The brief asks for three labels, and AVeriTeC has four: `Supported` → supported (24), `Refuted` → contradicted (25), `Not Enough Evidence` (10) and `Conflicting Evidence/Cherrypicking` (6) → unverified (16 total).

**What's in it.** Mostly political speech from 2019–2020: Kenyan county and national officials, US presidential candidates. 39 of the 65 claims are numerical. **One video holds 20 of the 65 claims** (a 52-minute speech by Wycliffe Oparanya), so the report reruns the main comparison without it.

**One row per claim, with its video.** The scorer saves rows like this, and the stats in step 6 resample by `video`:
```
{"id": "train-27", "video": "yt:WrGRkdthwKA", "gold": "contradicted", "pred": "unverified", "correct": 0, "backed": 1, "sb_correct": 0, ...}
```

## Step 3: The systems

All three share a single claim-extraction step: same prompt, temperature 0, cached. Each system labels two sets of claims per clip:
- **Gold claims** (the fact-checker's wording): isolates labeling and citation quality. This is the main comparison.
- **Extracted claims** (what the model pulled from the transcript): gives the end-to-end score.

| Guess about why the baseline fails | System | Extra cost |
|---|---|---|
| — | **Baseline: closed-book.** Same model, no tools. It may cite URLs from memory; the scorer fetches them. | 1× |
| It doesn't know the facts, or can't prove them | **Harness A: search + cite.** Write 2–3 search queries → retrieve passages → label using only the numbered evidence, quoting the sentence that backs the label | ~2× |
| Search queries built from the whole claim miss the part that's actually wrong | **Harness B: questions + search + cite.** Write 2–4 verification questions (who, what, when, how many, does context change the meaning) → search each → label | ~2.5× |

**Retrieval** (`retrieval.py`) has two backends:
- `RETRIEVER=wikipedia` (default, no key). It returns *current* article text, which can postdate the video.
- `RETRIEVER=brave`: web search capped to pages published on or before the video date. This is the better match for a real fact-checking workflow.

**Leakage guards.** Both backends drop fact-checking sites, which publish the answer key: a domain blocklist plus URL patterns like `/fact-check/`. They also drop each claim's own fact-check article and original post.

## Step 4: How a claim is scored

`03_score.py` writes one row per gold claim:

| Field | Meaning |
|---|---|
| `correct` | label matches the fact-checker |
| `backed` | label is `unverified`, or at least one citation passes all three checks: (1) the source resolves, (2) the quoted sentence is really in it (≥60% of its 4-word shingles), and (3) a judge model agrees the sentence backs the label |
| `sb_correct` | `correct` and `backed`: **source-backed accuracy, the headline metric** |
| `flip` | supported ↔ contradicted mix-up, the worst error for a fact-checker |
| `matched` / `e2e_correct` | the model extracted this claim itself (judge decides), and labeled its own version correctly |

The closed-book baseline's citations are checked by actually fetching the URL. A page that blocks our bot (401/403/429) counts as existing, but its quote can't be verified, so that citation doesn't pass.

**An abstention floor.** `unverified` needs no citation, so answering `unverified` to everything scores 16/65 = 24.6% on both metrics. The report prints this line so that a harness that mostly abstains doesn't look better than it is.

## Step 5: Results

*Pending a real run.* `04_report.py` writes `results/report.md` with the leaderboard below filled in, plus cost per system from token usage.

| System | Label acc | Source-backed acc | End-to-end acc | Macro-F1 | Flip rate | Citations that check out | Cost |
|---|---|---|---|---|---|---|---|
| Baseline: closed-book | | | | | | | |
| Harness A: search + cite | | | | | | | |
| Harness B: questions + search + cite | | | | | | | |

## Step 6: Is the gain real?

`04_report.py` runs both tests from the sample submission, on label accuracy and on source-backed accuracy:
- **McNemar.** Only claims where exactly one system was right count, and it reports a p-value.
- **Cluster bootstrap.** 5,000 resamples of **whole videos**, giving a 95% interval for the gap. Claims from one video share a speaker and topic, so resampling single claims would make the interval look tighter than it is.

**Interactive report.** `results/report.html` is one self-contained file that opens offline. It has:
- headline tiles with the verdict for each system
- the leaderboard against the "always unverified" floor
- McNemar disagreement bars, gap intervals and bootstrap histograms
- confusion matrices and the label mix
- the citation funnel (given → resolves → quote found → judge agrees)
- accuracy vs cost
- breakdowns by claim type, by gold label and per video
- a claim explorer with each system's rationale and the checks on every citation

A filter row rescopes every chart and test at once: drop the largest video, pick a claim type, or switch between source-backed and label accuracy. Every chart has a table view and a dark mode.

A gain counts as 🟢 real only when both tests agree. Anything else is ⬜️ not significant. With 65 claims, expect wide intervals: a true +10 pp gain may well come out ⬜️. Charts: `results/mcnemar.png`, `results/bootstrap.png`.

## Run order

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                      # add FIREWORKS_API_KEY (and BRAVE_API_KEY for RETRIEVER=brave)

python 01_build_testset.py   # already run: data/ is checked in and locked
python 06_traps.py           # trap tags + selection trail per claim -> data/claim_details.json (no model)
python 07_sft_data.py        # fine-tuning set from held-out AVeriTeC claims -> data/sft/ (no model)
python 08_sft_train.py --go  # LoRA fine-tune on Fireworks; then set SFT_MODEL_ID in .env
python run_all.py            # every (model, system) in models.json: 02_run -> 03_score -> 09_rubric, then 10_submission
```

| File | Does |
|---|---|
| `models.json` | Model registry (Fireworks / OpenRouter IDs, prices), harness list, which pairs run, the pre-registered final system |
| `02_run.py --model M --system S` | One run: `closed_book`, `rag` (A), `rag_qa` (B), `rag_qa_verify` (C, final), `agent` (D), `vote` (E) |
| `03_score.py --run M/S` | Label, citation and extraction scoring per claim |
| `09_rubric.py --run M/S` | 50-item rubric (20 deterministic + 30 blind-judge checks) and a failure-mode name per claim |
| `rubric.py` | The rubric, its area weights, and the 12-trap taxonomy |
| `10_submission.py` | `results/submission.html` |
| `04_report.py`, `05_dashboard.py` | The earlier single-model reports; pass run paths, e.g. `--baseline glm-5.3/closed_book --systems glm-5.3/rag` |

- Add `--limit 3` to `run_all.py` or `02_run.py` for a smoke test.
- **Model IDs and prices in `models.json` haven't been called yet.** Check them against the Fireworks model library and OpenRouter first.
- `LLM_BACKEND=mock` runs everything offline with fake replies and writes to `results_mock/`, never to `results/`.
- Every model call and web fetch is cached under `.cache/`, so reruns are free and reproducible.

## v2: harder task, bigger test set, dev-split iteration

v2 is a separate pipeline (scripts `11`–`18`, report `results_v2/report.html`; `results_v2_mock/` in mock mode). v1 is untouched.

**What changes:**
- **More claims.** v1's 65 AVeriTeC claims, plus English ClaimReview fact-checks (Google / Data Commons feed) of YouTube videos, located in the transcript the same way as v1. A separate **dev split** is used for every design decision. The test split is run once, at the end.
- **A harder task.** Four labels instead of three (adds `conflicting` / cherry-picked), a stated confidence, atomic sub-claims, and 2–5 verification questions. Each question must be answered from a cited, verbatim sentence.
- **Ground truth.** The fact-checker's label stays the gold label. `v2.ground_truth_model` (Claude Opus 5.5 at the highest reasoning effort the provider exposes) writes the rest of the answer key: gold questions and answers, sub-claims, the true values of numbers, traps and difficulty. It also flags label disagreements for human review instead of overwriting them. That model also judges, so it is **not evaluated**.
- **A stricter rubric** (`rubric_v2.py`): 57 checks in 6 weighted dimensions, plus 5 gates that cap the score at 25, a harm-weighted cost matrix, a Brier score, robustness checks (paraphrase, shuffled evidence) and efficiency budgets. A second judge (Qwen3.7 Plus) re-scores 25% of answers. Judge checks with Cohen's κ < 0.4 count half.
- **Iteration** (`15_iterate_v2.py`):
  1. Baselines on dev.
  2. Every candidate harness feature on its own.
  3. All positive features combined.
  4. The fine-tune (SFT + DPO) on the best harness.
  5. An open model (Kimi K3) reads the dev failures and proposes new ideas; arXiv references are checked against the arXiv API.
  6. Pre-registered stop rule, freeze, then one test run for every evaluated model.

**Status:** the code runs end to end in mock mode, so every model number in `results_v2_mock/` is fake. The test set currently has **103 test + 19 dev claims (54 test videos)**, because YouTube rate-limited transcript downloads partway through. Run `11_build_v2.py` again later to fill it toward 195 + 78. Note that the feed has only 59 supported and 28 not-enough-evidence YouTube claims in total, so those two labels can't reach their quotas. The v2 SFT data is real: 2,548 train / 283 val, with every v2 claim, video and fact-check article removed.

```bash
python 11_build_v2.py            # add --offline to use only cached transcripts
python 17_sft_v2.py data         # SFT data (no model needed)
python 12_ground_truth_v2.py     # answer key from the ground-truth model
python 17_sft_v2.py launch       # prints the Fireworks SFT -> DPO commands; set SFT2_MODEL_ID after
python 15_iterate_v2.py          # dev iteration, freeze, one test run, then 18_report_v2.py
python 17_sft_v2.py dpo          # after dev runs: preference pairs for the DPO stage
python 19_deck_v2.py             # 18-slide deck: results_v2/deck.html (arrow keys; print for PDF)
```

## Where this would lose points

| Criterion | What it shows | Where a judge would push |
|---|---|---|
| Benchmark evaluation | Same model across all systems, gold labels from professional fact-checkers, McNemar + video-level bootstrap, a sensitivity check without the biggest video, an abstention floor | 65 claims is small; one video holds 20 of them; Wikipedia retrieval is current, not date-capped |
| Business value | A named user (fact-checkers, trust and safety), value math with the assumptions written out | No interview with a fact-checker; time-saved numbers are assumptions |
| Technical execution | Full pipeline from transcripts to cited verdicts; citations verified by fetching the source, checking the quote, then judging it; results cached per call | Only YouTube clips with captions (no speech-to-text for TikTok/Reels); the judge model is the same family as the model under test |
| Documentation | Run order, mock mode, the funnel table above | Results pending a real run |

## Caveats

- **"Short video" is approximated.** Most clips are 90 s windows cut from longer speeches, because AVeriTeC's video claims are mostly speeches and interviews. Real short-form video (TikTok, Reels) would need speech-to-text and has more visual-only claims.
- **Wikipedia is a weak source for local statistics.** Claims like Kenyan county budget shares often have no Wikipedia evidence, so the retrieval harnesses will answer `unverified` there. The Brave backend fixes this but needs a key.
- **The judge is also a model.** It decides whether a quote backs a label and whether an extracted claim matches a gold claim. Set `JUDGE_MODEL` to a stronger fixed model for the final run.

## Next steps

- **Train and run the fine-tune** (`08_sft_train.py`). The data is built: 2,171 train / 241 validation examples, with every test claim, test video, test fact-check article and near-duplicate wording removed.
- **Build the planned harness ideas** (numbers checker, claim decomposition, country-aware sources); the full list of 25 is in the report's Improvements tab.
- **Add speech-to-text** (e.g. Whisper) to bring in the 26 TikTok, Facebook and Instagram claims.

## Disclosure

Built with Claude Code as an AI coding assistant. Data: AVeriTeC, under its dataset license (non-commercial; check the dataset card before reuse).
