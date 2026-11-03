"""The v2 report: 3x claims, harder task, rubric v2, the iteration log, improvements with diagrams and references.

    python 18_report_v2.py

Reads every results_v2/<split>/<model>/<system>/rows.jsonl, recomputes claim scores with
judge-reliability weights (Cohen's kappa per judge check, primary vs second judge), and
writes results_v2/report.html (self-contained, opens offline).
"""
import json
from datetime import datetime

import common
from common import DATA, MODELS, REGISTRY, ROOT, read_jsonl
from rubric import TRAPS
from rubric_v2 import COST, DIMENSIONS, GATES, LABELS4, RUBRIC, claim_score, kappa

V2 = REGISTRY["v2"]
RESULTS = ROOT / ("results_v2_mock" if common.MOCK else "results_v2")
TEMPLATE = ROOT / "report_v2_template.html"

A = "https://arxiv.org/abs/"
# name, type, status, feature flag, diagram steps, reference (label, url), build hours (estimate), note
IDEAS = [
    ("Search + cite", "harness", "built", "retrieve", ["Claim", "2-4 searches", "Numbered passages", "Label + quote"],
     ("Lewis et al. 2020, Retrieval-Augmented Generation", A + "2005.11401"), 3, "Base of every harness"),
    ("Verification questions", "harness", "built", "questions", ["Claim", "Who / what / when / how many?", "Search each", "Answer each from evidence", "Label"],
     ("Schlichtkrull et al. 2023, AVeriTeC; Dhuliawala et al. 2023, Chain-of-Verification", A + "2305.13117"), 2, ""),
    ("Verbatim quote check + repair", "harness", "built", "quote_check", ["Answer", "Shingle-match every quote", "Failures listed", "One repair turn", "Final answer"],
     ("Gao et al. 2022, RARR; Bohnet et al. 2022, Attributed QA", A + "2210.08726"), 2, "Deterministic, no extra model unless a quote fails"),
    ("Atomic sub-claims", "harness", "built", "decompose", ["Claim", "Split into atomic sub-claims", "Questions per sub-claim", "Label each", "Weakest-part aggregation"],
     ("Min et al. 2023, FActScore; Pan et al. 2023, ProgramFC", A + "2305.14251"), 4, "Targets the compound-claim trap"),
    ("Deterministic number check", "harness", "built", "numcheck", ["Claim figures", "Evidence figures (same unit)", "Nearest match + gap", "NUMBER CHECK block", "Labeler"],
     ("V et al. 2024, QuanTemp (numerical claims)", A + "2403.17169"), 3, "No model call; zero added time"),
    ("BM25 passage rerank", "harness", "built", "rerank", ["2x candidate passages", "BM25 vs claim + questions", "Top 10", "Labeler"],
     ("Nogueira & Cho 2019, Passage Re-ranking with BERT (reranking baseline)", A + "1901.04085"), 2, "Local, no model call"),
    ("Checkability router", "harness", "built", "router", ["Claim", "Cheap triage call", "Opinion / prediction?", "Skip retrieval", "Closed-book answer"],
     ("Chen et al. 2023, FrugalGPT; Mallen et al. 2022, adaptive retrieval", A + "2305.05176"), 2, "Saves time on claims search can't settle"),
    ("Adaptive voting", "harness", "built", "adaptive_vote", ["First answer", "Confidence < 0.7?", "4 more samples", "Majority label", "Agreement = confidence"],
     ("Aggarwal et al. 2023, Adaptive-Consistency; Wang et al. 2022, Self-Consistency", A + "2305.11860"), 2, "Cost only where uncertain, unlike always-5x voting"),
    ("Date guard", "harness", "built", "date_guard", ["Passages with dates", "Drop if after video date", "Labeler"],
     ("Glockner et al. 2022, Missing Counter-Evidence (temporal leakage)", A + "2210.13865"), 1, "Needs a dated backend (Brave)"),
    ("Open-model critic", "harness", "built", "critic", ["Answer + evidence", "Critic audits label vs evidence", "Inconsistent?", "Relabel, lower confidence"],
     ("Madaan et al. 2023, Self-Refine; McAleese et al. 2024, LLM Critics", A + "2303.17651"), 2, ""),
    ("Fact-check-site blocklist", "harness", "built", None, ["Search results", "Drop 27 fact-check domains + own article", "Clean evidence"],
     ("Glockner et al. 2022", A + "2210.13865"), 1, "Always on; prevents answer-key leakage"),
    ("Iterative search agent", "harness", "built (v1)", None, ["Evidence so far", "Need more?", "New queries", "Up to 2 more rounds", "Label"],
     ("Yao et al. 2022, ReAct; Press et al. 2022, Self-Ask", A + "2210.03629"), 3, "v1 harness D"),
    ("Selective abstention by confidence", "harness", "planned", None, ["Answer + confidence", "Below tuned threshold?", "Answer not_enough_evidence"],
     ("Kamath et al. 2020, Selective QA; Kadavath et al. 2022", A + "2006.09462"), 2, "Tune the threshold on dev only"),
    ("Few-shot label definitions", "harness", "planned", None, ["2 worked examples per label", "Prepend to labeler", "Label"],
     ("Brown et al. 2020, few-shot prompting", A + "2005.14165"), 1, "Targets conflicting vs refuted confusion"),
    ("Country-aware sources", "harness", "planned", None, ["Location code", "site: national statistics / parliament", "Extra passages"],
     ("This test set: local-knowledge claims are the largest trap group", ""), 4, ""),
    ("Wider transcript window", "harness", "planned", None, ["Long source video", "+/-60 s context", "Labeler"],
     ("This test set: most clips are cut from long videos", ""), 1, ""),
    ("SFT v2 on fact-checkers' research", "post-training", "data", "ft", ["AVeriTeC claims", "Leakage filter", "v2 prompt + gold QA target", "LoRA SFT", "SFT checkpoint"],
     ("Hu et al. 2021, LoRA; Schlichtkrull et al. 2023, AVeriTeC", A + "2106.09685"), 6, "~1-1.5 h training on Fireworks serverless"),
    ("Evidence dropout", "post-training", "data", "ft", ["Decisive example", "Remove gold passages (10%)", "Target: not_enough_evidence"],
     ("Rajpurkar et al. 2018, Know What You Don't Know", A + "1806.03822"), 1, ""),
    ("Same-country hard negatives", "post-training", "data", "ft", ["Gold passages", "+2-4 distractors, same country", "Shuffle"],
     ("Karpukhin et al. 2020, Dense Passage Retrieval (hard negatives)", A + "2004.04906"), 1, ""),
    ("Leakage-safe training split", "post-training", "data", "ft", ["All AVeriTeC", "Drop held-out claims, videos, articles, near-duplicates", "Training pool"],
     ("Lewis et al. 2020, Test-Train Overlap in Open-Domain QA", A + "2008.02637"), 1, ""),
    ("Citation-quality filter", "post-training", "data", "ft", ["Gold answer", "Best sentence vs claim", ">= 5 words, >= 2 shared words?", "Cite or skip"],
     ("Gao et al. 2023, ALCE (citation quality)", A + "2305.14627"), 1, ""),
    ("Confidence targets", "post-training", "data", "ft", ["Label", "Target confidence 0.7-0.9", "Calibrated outputs"],
     ("Guo et al. 2017, On Calibration of Modern Neural Networks", A + "1706.04599"), 1, ""),
    ("DPO on citation quality", "post-training", "untested", "ft", ["Dev runs", "Verified vs failed answer, same prompt", "Preference pairs", "DPO on SFT model"],
     ("Rafailov et al. 2023, Direct Preference Optimization", A + "2305.18290"), 4, "~0.5 h training; needs dev runs first"),
    ("RL with a verifiable reward", "post-training", "planned", None, ["DPO checkpoint", "Sample answers", "Reward = label x gates x verified quote", "GRPO update"],
     ("Shao et al. 2024, DeepSeekMath (GRPO)", A + "2402.03300"), 16, ""),
    ("Train the question writer", "post-training", "planned", None, ["Claim", "Fact-checker's questions as target", "SFT", "Better searches"],
     ("Schlichtkrull et al. 2023, AVeriTeC", A + "2305.13117"), 4, ""),
    ("Distill frontier rationales", "post-training", "rejected", None, ["Frontier answers", "Train open model"],
     ("Provider terms restrict training competing models on outputs", ""), 0, "Fact-checker justifications are used instead"),
]


def failure(r):
    if not r["gates"].get("G4", True):
        return "unparseable output"
    if not r["gates"].get("G1", True):
        return "flip (supported <-> refuted)"
    if not r["gates"].get("G2", True):
        return "fabricated citation"
    if r["pred"] == "not_enough_evidence" and r["gold"] != "not_enough_evidence":
        return "wrong abstention"
    if r["pred"] != "not_enough_evidence" and r["gold"] == "not_enough_evidence":
        return "over-claim"
    if r["gold"] == "conflicting" and r["pred"] != "conflicting":
        return "missed the misleading part"
    if not r["correct"]:
        return "wrong label"
    if not r["verified"]:
        return "right label, not verified"
    return "verified"


def build_payload():
    """Everything the report (and 19_deck_v2.py) needs, from results_v2[_mock]/ and data/v2/."""
    runs, rows = [], {}
    for f in sorted(RESULTS.glob("*/*/*/rows.jsonl")):
        split, model, system = f.parts[-4], f.parts[-3], f.parts[-2]
        usage = json.loads((f.parent / "usage.json").read_text())
        key = f"{split}/{model}/{system}"
        m = MODELS.get(model, {"name": model, "tier": "open"})
        runs.append({"key": key, "split": split, "model": model, "system": system, "flags": usage.get("flags", []),
                     "name": m["name"], "tier": m["tier"], "usage": usage})
        rows[key] = {r["id"]: r for r in read_jsonl(f)}
    if not runs:
        raise SystemExit("no v2 runs: run 15_iterate_v2.py")
    jids = [it["id"] for it in RUBRIC if it["method"] == "judge"]
    pairs = {i: [] for i in jids}
    for rs in rows.values():
        for r in rs.values():
            if r.get("judge2"):
                for i in jids:
                    pairs[i].append((r["items"].get(i), r["judge2"].get(i)))
    kap = {i: kappa(p) for i, p in pairs.items()}
    compact = {}
    for key, rs in rows.items():
        compact[key] = {}
        for i, r in rs.items():
            sc, dims = claim_score(r["items"], r["gates"], kap)
            compact[key][i] = {
                **{k: r[k] for k in ("pred", "confidence", "correct", "correct3", "verified", "cost", "brier", "gates", "matched",
                                     "e2e_correct", "route", "repaired", "votes", "critic", "questions", "subclaims", "rationale",
                                     "claim_understood", "tiers", "difficulty", "per_claim")},
                "score": round(sc, 2), "dims": dims, "items": [r["items"].get(it["id"]) for it in RUBRIC],
                "cites": [{k: c.get(k) for k in ("url", "quote", "resolves", "quote_found", "backs", "ok")} for c in r["citations"]],
                "failure": failure(r)}
    claims = read_jsonl(DATA / "v2" / "claims.jsonl")
    clips = {c["clip_id"]: c for c in read_jsonl(DATA / "v2" / "clips.jsonl")}
    gt_path = DATA / "v2" / "ground_truth.jsonl"
    gt = {x["id"]: x for x in read_jsonl(gt_path)} if gt_path.exists() else {}
    v1 = json.loads((DATA / "claim_details.json").read_text())
    adj = DATA / "v2" / "adjudication.jsonl"

    def load(p, default=None):
        return json.loads(p.read_text()) if p.exists() else default
    payload = {
        "meta": {"mock": common.MOCK, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "gt_model": MODELS[V2["ground_truth_model"]]["name"], "gt_effort": V2["ground_truth_reasoning"]["effort"],
                 "judge": MODELS[V2["judge"]]["name"], "judge2": MODELS[V2["second_judge"]]["name"],
                 "ideation": MODELS[V2["ideation_model"]]["name"], "base_open": V2["base_open"], "evaluated": V2["evaluated"],
                 "excluded": [V2["ground_truth_model"]]},
        "models": REGISTRY["models"], "features": V2["features"], "runs": runs, "rows": compact,
        "claims": [{**{k: c.get(k) for k in ("id", "split", "source", "video", "claim", "label4", "gold_raw", "speaker", "setting",
                                             "claim_date", "fact_checker", "justification", "fact_checking_article", "located_by",
                                             "overlap", "clip_id")},
                    "traps": sorted(set(v1.get(c["id"], {}).get("traps", [])) | set(gt.get(c["id"], {}).get("traps", []))),
                    "qa": v1.get(c["id"], {}).get("qa", []), "gt": gt.get(c["id"]),
                    "clip": {k: clips[c["clip_id"]][k] for k in ("start", "end", "url", "transcript", "video_seconds")}}
                   for c in claims],
        "rubric": {"dims": DIMENSIONS, "gates": GATES, "cost": COST, "items": RUBRIC, "labels": LABELS4, "traps": TRAPS, "kappa": kap},
        "iteration": load(RESULTS / "iteration_log.json", {}), "ideas_open": load(RESULTS / "ideas_open.json", {}),
        "stats": load(DATA / "v2" / "stats.json", {}), "sft": load(DATA / "sft_v2" / "stats.json"),
        "dpo_pairs": sum(1 for _ in open(DATA / "sft_v2" / "dpo_pairs.jsonl")) if (DATA / "sft_v2" / "dpo_pairs.jsonl").exists() else 0,
        "adjudication": read_jsonl(adj) if adj.exists() else [],
        "ideas": [dict(zip(("name", "type", "status", "feature", "steps", "ref", "build_hours", "note"), (*x[:5], {"label": x[5][0], "url": x[5][1]}, *x[6:])))
                  for x in IDEAS],
    }
    return payload


def main():
    payload = build_payload()
    runs, claims = payload["runs"], payload["claims"]
    blob = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    out = RESULTS / "report.html"
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/{}", blob))
    print(f"wrote {out}: {len(runs)} runs, {len(claims)} claims{' (MOCK DATA)' if common.MOCK else ''}")


if __name__ == "__main__":
    main()
