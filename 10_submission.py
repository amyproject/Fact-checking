"""The submission report: a Forge-style write-up plus every chart, claim and rubric detail, in one HTML file.

    python 10_submission.py

Reads every results/<model>/<system>/ that has rubric.jsonl, plus the test set, trap tags
and fine-tuning stats, and writes results/submission.html (self-contained, opens offline).
All statistics (McNemar, video-level bootstrap, Holm correction, permutation tests, power)
are computed in the page so its filters rescope every number at once.
"""
import importlib
import json
from datetime import datetime

from common import DATA, MOCK, MODELS, REGISTRY, RESULTS, ROOT, SYSTEMS, read_jsonl
from rubric import AREAS, RUBRIC, TRAPS

TEMPLATE = ROOT / "submission_template.html"
ROW_KEYS = ("pred", "correct", "backed", "sb_correct", "flip", "matched", "e2e_correct", "e2e_pred", "rationale",
            "parsed", "repaired", "n_citations", "n_citations_ok", "search_rounds", "votes", "queries", "e2e_claim")

# status: built = in the code and run by run_all.py; data = built and run, no model needed;
# untested = written, needs a key; planned = not built yet; rejected = considered and dropped
IDEAS = [
    ("Search + cite (harness A)", "harness", "built", "rag",
     "The model doesn't know local or dated facts.", "2-3 queries, label only from numbered passages, quote the backing sentence."),
    ("Verification questions (harness B)", "harness", "built", "rag_qa",
     "Queries built from the whole claim miss the part that is wrong.", "Ask 2-4 who/what/when/how-many questions, search each."),
    ("Verbatim quote check + repair (harness C)", "harness", "built", "rag_qa_verify",
     "Models paraphrase or invent quotes, so citations fail the scorer.", "Deterministic shingle check on every quote; one repair turn with the failures listed."),
    ("Iterative search agent (harness D)", "harness", "built", "agent",
     "One round of search is often not enough for numbers.", "Model reads what it has and asks for up to 2 more rounds of queries."),
    ("Self-consistency vote (harness E)", "harness", "built", "vote",
     "Single samples flip on borderline claims.", "5 samples at T=0.7, majority label, citations from a majority sample."),
    ("Fact-check-site blocklist", "harness", "built", None,
     "Search results include the answer key.", "27 fact-check domains + URL patterns + each claim's own article and post are removed from evidence."),
    ("Date-capped web search", "harness", "untested", None,
     "Wikipedia is current, not as of the video date, and thin on local facts.", "Brave backend with freshness capped at the video date (RETRIEVER=brave, needs a key)."),
    ("Fine-tune on fact-checkers' research", "post-training", "data", "glm-5.3-ft",
     "The base model can't map evidence to the right label and quote reliably.", "LoRA SFT on AVeriTeC questions + answers -> label + justification + quoted sentence, in the exact harness-C prompt."),
    ("Evidence-dropout augmentation", "post-training", "data", "glm-5.3-ft",
     "When retrieval misses, models guess instead of abstaining.", "10% of decisive training examples lose every gold passage and become unverified."),
    ("Same-country hard negatives", "post-training", "data", "glm-5.3-ft",
     "Models cite passages that are on-topic but about the wrong place or number.", "2-4 distractor answers from other claims in the same country are mixed into each example."),
    ("Leakage-safe training split", "post-training", "data", "glm-5.3-ft",
     "63 of 65 test claims come from AVeriTeC train.", "Drop test claims, same video, same fact-check article, and Jaccard >= 0.5 near-duplicates."),
    ("Citation-quality filter on targets", "post-training", "data", "glm-5.3-ft",
     "AVeriTeC answers are sometimes fragments that can't back a label.", "Only cite gold passages with >= 5 words sharing >= 2 content words with the claim."),
    ("Label rebalancing", "post-training", "data", "glm-5.3-ft",
     "AVeriTeC is mostly Refuted, so a fine-tune would over-predict contradicted.", "Cap the majority class at 1.5x the next largest."),
    ("Atomic claim decomposition", "harness", "planned", None,
     "Compound claims (21 of 65) are judged by their easiest part.", "Split into atomic sub-claims, verify each, label by the weakest."),
    ("Deterministic number checker", "harness", "planned", None,
     "53 of 65 claims hinge on an exact figure; models accept near-misses.", "Parse numbers and units from claim and quote; compare with a tolerance rule before labeling."),
    ("Country-aware source routing", "harness", "planned", None,
     "37 of 65 claims need local knowledge (mostly Kenya).", "Add site: queries for national statistics offices, parliament records and national papers by location code."),
    ("Passage reranking", "harness", "planned", None,
     "Top Wikipedia passages are often off-topic.", "Cross-encoder rerank of 30 candidates down to 10 before labeling."),
    ("Wider transcript window", "harness", "planned", None,
     "53 claims are cut from long videos; context may sit outside 90 s.", "Give the labeler +/-60 s of transcript when the source video is long."),
    ("Vote-agreement abstention", "harness", "planned", None,
     "Low agreement means the evidence is weak.", "In harness E, answer unverified when fewer than 3 of 5 samples agree."),
    ("Open-model critic", "harness", "planned", None,
     "A second pass catches rationale/label mismatches.", "A second open model audits the verdict against the evidence; disagreements go to unverified."),
    ("Few-shot label definitions", "harness", "planned", None,
     "\"Misleading\" vs \"unverified\" is the commonest confusion.", "Add 6 worked AVeriTeC train examples (2 per label) to the labeling prompt."),
    ("Train the question writer", "post-training", "planned", None,
     "Better questions mean better retrieval.", "SFT on AVeriTeC claim -> fact-checker questions (data already in 07_sft_data.py's source)."),
    ("Preference tuning on citations (DPO)", "post-training", "planned", None,
     "SFT teaches format; it doesn't punish invented quotes.", "Pairs from harness runs: verified-quote answers preferred over failed-quote answers for the same claim."),
    ("RL with a verifiable reward", "post-training", "planned", None,
     "The scorer's checks are a ready-made reward.", "GRPO with reward = label correct x quote verbatim x judge-backed, on the SFT checkpoint."),
    ("Distill frontier rationales", "post-training", "rejected", None,
     "Frontier explanations might teach better reasoning.", "Dropped: frontier providers' terms restrict training competing models on outputs; fact-checker justifications are used instead."),
]

DEMOS = [
    ("Headline tiles", "r-hero", "The one sentence: improved open model vs best frontier, with the verdict."),
    ("Open vs frontier vs improved", "r-tiers", "Every model before and after the harness on one axis."),
    ("Leaderboard with intervals", "r-board", "All runs ranked, each with its 95% video-level interval."),
    ("Paired win rate", "r-winrate", "How often each system beats its reference on the same claim, ties shown."),
    ("Significance table (Holm)", "r-sig", "Every planned comparison, corrected for testing 13 at once."),
    ("McNemar disagreement bars", "r-mcnemar", "Only the claims where exactly one system was right."),
    ("Gap forest", "r-gaps", "Size of each gap with its interval; crossing zero means luck."),
    ("Bootstrap histograms", "r-boot", "Would the headline gap survive a different set of videos?"),
    ("Power curve", "r-power", "Which gains 65 claims can detect, and how many claims a smaller gain would need."),
    ("Rubric heatmap (5 areas)", "r-areas", "Where each system earns and loses points."),
    ("Rubric drill-down (50 items)", "r-items", "Every one of the 50 checks for every run."),
    ("Confusion matrices", "r-confusion", "What each system answers for each true label."),
    ("Label mix", "r-mix", "Who abstains, who over-claims."),
    ("Failure modes", "r-failures", "Flip, wrong abstention, over-claim, citation failures, by run."),
    ("Citation funnel", "r-funnel", "Given -> resolves -> verbatim -> judge agrees."),
    ("Accuracy vs cost", "r-cost", "Bubble = seconds per claim. Is the gain worth it?"),
    ("Coverage vs precision", "r-coverage", "How often a system commits, and how often it's right when it does."),
    ("By claim type", "r-type", "Numerical, event, causal, with n beside every bar."),
    ("By trap", "r-trap", "Which traps the improvement fixes."),
    ("By country", "r-country", "Local-knowledge claims vs the rest, with n."),
    ("By true label", "r-gold", "Per-class accuracy."),
    ("Per video", "r-video", "One dot per video; shows the 20-claim speech's weight."),
    ("Ablation waterfall", "r-waterfall", "What each harness step and the fine-tune add."),
    ("Before / after example", "r-example", "One claim, baseline vs improved, citations checked."),
    ("Extraction and end-to-end", "r-extract", "Can the model find the claims in the transcript itself?"),
    ("Test-set funnel", "s-funnel", "166 video claims -> 65, every drop explained."),
    ("Fine-tuning data", "s-sft", "What the fine-tune learns from, after leakage control."),
    ("Every video source", "v-status", "All 75 videos behind the claims, why 47 were dropped, and each video's claims."),
]


PLATFORMS = (("youtu", "YouTube"), ("tiktok", "TikTok"), ("instagram", "Instagram"), ("facebook", "Facebook"),
             ("fb.watch", "Facebook"), ("rumble", "Rumble"), ("bitchute", "BitChute"))


def sources(kept_claims, clips):
    """Every video behind an AVeriTeC video claim, kept or not, with each claim's fate."""
    tb = importlib.import_module("01_build_testset")
    df = tb.load_averitec()
    df = df[df.original_claim_url.fillna("").str.contains(tb.VIDEO_URL)].copy()
    df["video"] = df.original_claim_url.map(tb.video_key)
    kept = {c["id"]: c for c in kept_claims}
    out = []
    for vid, g in df.groupby("video"):
        url = g.original_claim_url.iloc[0]
        bare = url.split("web.archive.org/web/")[-1]
        platform = next((n for k, n in PLATFORMS if k in bare.lower()), "other")
        duration, reason = None, None
        if not vid.startswith("yt:") and platform == "YouTube":
            reason = "YouTube link without a video ID"
        elif vid.startswith("yt:"):
            f = DATA / "transcripts" / f"{vid[3:]}.json"
            if f.exists():
                t = json.loads(f.read_text())
                reason = t.get("reason")
                if t.get("snippets"):
                    duration = round(t["snippets"][-1]["end"])
            else:
                reason = "request blocked by YouTube (transient; not cached)"
        else:
            reason = "not YouTube: would need speech-to-text"
        cl = []
        for _, r in g.iterrows():
            cid = f"{r.split}-{r.row}"
            fate = ("kept" if cid in kept else "claim not found in the transcript" if not reason
                    else reason if platform != "YouTube" or not vid.startswith("yt:") else f"no transcript: {reason}")
            cl.append({"id": cid, "claim": r.claim, "gold_raw": r.label, "date": r.claim_date if isinstance(r.claim_date, str) else "",
                       "speaker": (r.speaker if isinstance(r.speaker, str) else "").strip(), "fate": fate,
                       "fact_checking_article": r.fact_checking_article if isinstance(r.fact_checking_article, str) else ""})
        n_kept = sum(c["fate"] == "kept" for c in cl)
        status = ("kept" if n_kept else "not YouTube" if platform != "YouTube"
                  else "no transcript" if reason else "claims not found")
        out.append({
            "video": vid, "url": f"https://www.youtube.com/watch?v={vid[3:]}" if vid.startswith("yt:") else url,
            "platform": platform, "duration": duration, "status": status, "reason": reason,
            "speaker": next((c["speaker"] for c in cl if c["speaker"]), ""),
            "date": min((c["date"] for c in cl if c["date"]), key=lambda d: d.split("-")[::-1], default=""),
            "clips": [{"clip_id": c["clip_id"], "start": c["start"], "end": c["end"], "url": c["url"]}
                      for c in clips if c["video"] == vid],
            "claims": cl, "n_kept": n_kept,
        })
    return sorted(out, key=lambda v: (-v["n_kept"], -len(v["claims"]), v["video"]))


def main():
    claims = read_jsonl(DATA / "claims.jsonl")
    details = json.loads((DATA / "claim_details.json").read_text())
    runs, rows = [], {}
    for d in sorted(RESULTS.glob("*/*/rubric.jsonl")):
        model, system = d.parent.parent.name, d.parent.name
        run = f"{model}/{system}"
        scored = {r["id"]: r for r in read_jsonl(d.parent / "rows.jsonl")}
        rub = {r["id"]: r for r in read_jsonl(d)}
        usage = json.loads((d.parent / "usage.json").read_text())
        n_verdicts = sum(len(p["oracle"]) + len(p["extracted"]) for p in read_jsonl(d.parent / "preds.jsonl"))
        m = MODELS.get(model, {"name": model, "tier": "open"})
        runs.append({"run": run, "model": model, "system": system, "name": m["name"], "tier": m["tier"],
                     "sys_name": SYSTEMS[system]["name"], "usage": usage, "n_verdicts": n_verdicts})
        rows[run] = {i: {**{k: r.get(k) for k in ROW_KEYS},
                         "cites": [{k: c.get(k) for k in ("url", "quote", "resolves", "quote_found", "backs", "ok")}
                                   for c in r["citations"]],
                         "score": rub[i]["score"], "areas": rub[i]["areas"], "failure": rub[i]["failure"],
                         "items": [rub[i]["items"][it["id"]] for it in RUBRIC]}
                     for i, r in scored.items() if i in rub}
    if not runs:
        raise SystemExit("no scored runs: run run_all.py first")
    ids = set.intersection(*(set(r) for r in rows.values()))
    sft = DATA / "sft" / "stats.json"
    payload = {
        "meta": {"mock": MOCK, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "base_open": REGISTRY["base_open"], "final_system": REGISTRY["final_system"],
                 "improved_model": "glm-5.3-ft", "judge": MODELS[REGISTRY["judge"]]["name"],
                 "retriever": next((r["usage"].get("retriever") for r in runs if r["usage"].get("retriever")), None)},
        "models": REGISTRY["models"], "systems": SYSTEMS,
        "runs": runs,
        "rows": {k: {i: v[i] for i in ids} for k, v in rows.items()},
        "claims": [{**{k: c[k] for k in ("id", "video", "claim", "gold", "gold_raw", "claim_types", "speaker", "claim_date")},
                    **details[c["id"]]} for c in claims if c["id"] in ids],
        "rubric": {"areas": AREAS, "items": RUBRIC, "traps": TRAPS},
        "testset": json.loads((DATA / "testset_stats.json").read_text()),
        "sft": json.loads(sft.read_text()) if sft.exists() else None,
        "sources": sources(claims, read_jsonl(DATA / "clips.jsonl")),
        "ideas": [dict(zip(("name", "type", "status", "link", "hypothesis", "how"), x)) for x in IDEAS],
        "demos": [dict(zip(("name", "target", "shows"), x)) for x in DEMOS],
    }
    blob = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    out = RESULTS / "submission.html"
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/{}", blob))
    print(f"wrote {out}: {len(runs)} runs, {len(ids)} claims{' (MOCK DATA)' if MOCK else ''}")


if __name__ == "__main__":
    main()
