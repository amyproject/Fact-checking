"""Score a v2 run: citations, gates, the 64-check rubric (rubric_v2.py), verified accuracy, cost, calibration.

    python 14_score_v2.py --run dev/glm-5.3/rag

Needs 12_ground_truth_v2.py (the answer key the judge compares against). Judge checks go
to the ground-truth model in one blind call per answer; a 25% hash-chosen sample also goes
to a second judge so the report can show Cohen's kappa per check.

Writes results_v2/<split>/<model>/<system>/rows.jsonl.
"""
import argparse
import asyncio
import hashlib
import json

import common
from common import DATA, MODELS, REGISTRY, ROOT, chat, parse_json, quote_in, read_jsonl, write_jsonl
from retrieval import blocked, parse_claim_date
from rubric_v2 import COST, GATES, RUBRIC, applies, claim_score, source_tier
from rubric import TRAPS

s1 = __import__("importlib").import_module("03_score")
V2 = REGISTRY["v2"]
RESULTS = ROOT / ("results_v2_mock" if common.MOCK else "results_v2")
JUDGE, JUDGE2 = V2["judge"], V2["second_judge"]
TO3 = {"supported": "supported", "refuted": "contradicted", "not_enough_evidence": "unverified", "conflicting": "unverified"}
BUDGET = {"calls": 8, "tokens": 12000, "seconds": 20}

JUDGE_PROMPT = """You are auditing an automated fact-check against an answer key. Score each criterion: 1 if the answer meets it, 0 if not, null if it does not apply. Be strict: pass only if a careful editor would agree without caveats. You do not know which system wrote the answer.

CLAIM: {claim}
VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}

ANSWER KEY (fact-checker verdict + ground-truth annotations)
Verdict: {gold} ({rating})
Explanation: {justification}
Key questions and answers:
{gold_qa}
Sub-claims: {gold_sub}
Quantities: {gold_q}
Traps: {traps}

SYSTEM ANSWER
Label: {label} (confidence {conf})
Claim as understood: {understood}
Sub-claims: {sub}
Questions and answers:
{qa}
Rationale: {rationale}
Citations:
{cites}

CRITERIA
{criteria}

Return JSON: {{"scores": {{"<id>": 1, ...}}}}"""


def domain(u):
    u = (u or "").lower().split("web.archive.org/web/")[-1]
    u = u.split("://", 1)[-1].split("/", 1)[0]
    return u.removeprefix("www.")


async def judge_items(model, ctx, items):
    if not items:
        return {}
    text = await chat([{"role": "user", "content": JUDGE_PROMPT.format(**ctx, criteria="\n".join(
        f"- {i['id']}: {i['criterion']}" for i in items))}], tag="rubric", model=model, judge=True, max_tokens=1500)
    got = parse_json(text).get("scores", {})
    return {i["id"]: (None if got.get(i["id"]) is None else int(got.get(i["id"]) in (1, True, "1", "true"))) for i in items}


async def score(g, pred, gt, det, usage, base_usage):
    v = pred["oracle"][g["id"]]
    checked = list(await asyncio.gather(*[s1.check_citation(g["claim"], {**v, "label": v["label"]}, c) for c in v["citations"]]))
    qa_cites = [c for q in v["questions"] for c in q["citations"]]
    qa_verbatim = [quote_in(c["quote"], v["evidence"][c["id"] - 1]["text"]) if "id" in c and v["evidence"] else True for c in qa_cites]
    m = await s1.match(g["claim"], pred["extracted"])
    e2e = pred["extracted"][m - 1]["verdict"] if m else None
    gold, lab = g["label4"], v["label"]
    correct = int(lab == gold)
    decisive = lab != "not_enough_evidence"
    tiers = [source_tier(c.get("url")) for c in checked]
    doms = {domain(c.get("url")) for c in checked if c["ok"]}
    gold_doms = {domain(x.get("url")) for x in det.get("qa", []) if x.get("url")}
    own = (g.get("fact_checking_article"), g.get("original_claim_url"))
    dates = [str(v["evidence"][c["id"] - 1].get("date") or "")[:10] for c in v["citations"] if "id" in c and v["evidence"]]
    vdate = parse_claim_date(g.get("claim_date"))  # yyyy-mm-dd or None
    n = max(usage["n_verdicts"], 1)
    per = {"calls": usage["calls"] / n, "tokens": (usage["prompt_tokens"] + usage["completion_tokens"]) / n,
           "seconds": usage["latency_s"] / n}
    rob = v.get("robust") or {}
    gates = {
        "G1": not ({gold, lab} == {"supported", "refuted"}),
        "G2": all(c["quote_found"] and c["resolves"] for c in checked) and all(qa_verbatim),
        "G3": not any(vdate and len(d) == 10 and d > vdate for d in dates),
        "G4": bool(v.get("parsed", True)),
    }
    a = {"traps": set(det.get("traps", [])) | set(gt.get("traps", [])), "gold_decisive": gold != "not_enough_evidence",
         "gold_nee": gold == "not_enough_evidence", "gold_conflicting": gold == "conflicting", "matched": bool(m),
         "pred_decisive": decisive, "has_citations": bool(checked), "has_gold_sources": bool(gold_doms),
         "has_gold_qa": bool(gt.get("questions")), "has_questions": bool(v["questions"]),
         "has_paraphrase": "paraphrase_label" in rob, "has_shuffle": "shuffle_label" in rob, "has_votes": bool(v.get("votes"))}
    auto = {
        "V1": correct, "V2": int(TO3[lab] == TO3[gold]), "V3": int(COST[gold][lab] <= 0.3),
        "V4": int((v["confidence"] - correct) ** 2 <= 0.25), "V5": int(decisive), "V6": int(not decisive),
        "V7": int(lab == "conflicting"), "V10": int(bool(m)), "V11": int(bool(e2e) and e2e["label"] == gold),
        "E1": int(any(c["ok"] for c in checked)), "E2": int(all(c["quote_found"] for c in checked)),
        "E3": int(all(c["resolves"] for c in checked)),
        "E4": int(not any(blocked(c.get("url", ""), own) for c in checked)),
        "E5": int(bool(tiers) and min(tiers) <= 2), "E6": int(bool(tiers) and sum(tiers) / len(tiers) <= 2.5),
        "E7": int(len(doms) >= 2), "E8": int(bool(doms & gold_doms)),
        "E11": int(bool(checked) and sum(c["ok"] for c in checked) >= len(checked) / 2),
        "Q1": int(2 <= len(v["questions"]) <= 5),
        "Q4": int(all(q["citations"] for q in v["questions"] if q["answer"] and "no answer" not in q["answer"].lower())),
        "Q5": int(all(qa_verbatim)), "Q10": int(len({q["q"].lower() for q in v["questions"]}) == len(v["questions"])),
        "R2": int(not ("refuted" in [s["label"] for s in v["subclaims"]] and lab == "supported")),
        "R3": int(bool(v.get("numbers_checked")) or any(ch.isdigit() for ch in " ".join(q["answer"] for q in v["questions"]))),
        "B1": int(rob.get("paraphrase_label") == lab), "B2": int(rob.get("shuffle_label") == lab),
        "B3": int(abs(rob.get("paraphrase_conf", v["confidence"]) - v["confidence"]) < 0.2),
        "B4": int(max((v.get("votes") or {"x": 5}).values()) >= 4),
        "U1": int(per["calls"] <= BUDGET["calls"]), "U2": int(per["tokens"] <= BUDGET["tokens"] or not usage["prompt_tokens"]),
        "U3": int(per["seconds"] <= BUDGET["seconds"]), "U8": int(15 <= len(v["rationale"].split()) <= 150),
        "U9": int(len(v["citations"]) <= 4),
    }
    items, jitems = {}, []
    for it in RUBRIC:
        if not applies(it["applies"], a):
            items[it["id"]] = None
        elif it["method"] == "auto":
            items[it["id"]] = auto[it["id"]]
        else:
            jitems.append(it)
    jitems.append({"id": "G5", "criterion": GATES[4][1], "method": "judge"})
    ctx = {"claim": g["claim"], "date": g.get("claim_date") or "unknown", "transcript": det["transcript"][:1500],
           "gold": gold, "rating": g.get("gold_raw", ""), "justification": (g.get("justification") or "")[:900],
           "gold_qa": "\n".join(f"- {q['q']} -> {q.get('answer', '')}" for q in gt.get("questions", [])) or "(none)",
           "gold_sub": json.dumps(gt.get("subclaims", []))[:600], "gold_q": json.dumps(gt.get("quantities", []))[:400],
           "traps": ", ".join(TRAPS[t]["name"] for t in a["traps"] if t in TRAPS) or "none",
           "label": lab, "conf": v["confidence"], "understood": v["claim_understood"], "sub": json.dumps(v["subclaims"])[:600],
           "qa": "\n".join(f"- {q['q']} -> {q['answer']}" for q in v["questions"]) or "(none)", "rationale": v["rationale"],
           "cites": "\n".join(f"- \"{c['quote']}\" ({c.get('url', '')}; verbatim={c['quote_found']})" for c in checked) or "(none)"}
    primary = await judge_items(JUDGE, ctx, jitems)
    second = None
    if int(hashlib.md5((g["id"] + usage["system"]).encode()).hexdigest(), 16) % 100 < 100 * V2["second_judge_share"]:
        second = await judge_items(JUDGE2, ctx, jitems)
    gates["G5"] = primary.get("G5") != 0
    for it in jitems[:-1]:
        items[it["id"]] = primary.get(it["id"])
    sc, shares = claim_score(items, gates)
    verified = int(correct and all(gates.values()) and (not decisive or auto["E1"]))
    return {
        "id": g["id"], "split": g["split"], "source": g["source"], "video": g["video"], "gold": gold, "pred": lab,
        "confidence": v["confidence"], "correct": correct, "correct3": auto["V2"], "verified": verified,
        "cost": COST[gold][lab], "brier": round((v["confidence"] - correct) ** 2, 4), "gates": gates, "items": items,
        "judge2": second, "score": round(sc, 2), "dims": shares, "matched": int(bool(m)), "e2e_correct": auto["V11"],
        "route": v.get("route"), "repaired": bool(v.get("repaired")), "votes": v.get("votes"), "critic": v.get("critic"),
        "citations": checked, "questions": v["questions"], "subclaims": v["subclaims"], "rationale": v["rationale"],
        "claim_understood": v["claim_understood"], "queries": v.get("queries", []), "tiers": tiers,
        "per_claim": per, "difficulty": gt.get("difficulty"),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="<split>/<model>/<system> under results_v2/")
    args = ap.parse_args()
    run = RESULTS / args.run
    preds = {p["clip_id"]: p for p in read_jsonl(run / "preds.jsonl")}
    usage = json.loads((run / "usage.json").read_text())
    claims = [c for c in read_jsonl(DATA / "v2" / "claims.jsonl") if c["clip_id"] in preds]
    gt = {x["id"]: x for x in read_jsonl(DATA / "v2" / "ground_truth.jsonl")}
    v1 = json.loads((DATA / "claim_details.json").read_text())
    clips = {c["clip_id"]: c for c in read_jsonl(DATA / "v2" / "clips.jsonl")}
    det = {c["id"]: {**v1.get(c["id"], {}), "transcript": clips[c["clip_id"]]["transcript"],
                     "traps": v1.get(c["id"], {}).get("traps", [])} for c in claims}
    rows = await asyncio.gather(*[score(c, preds[c["clip_id"]], gt.get(c["id"], {}), det[c["id"]], usage, None) for c in claims])
    write_jsonl(run / "rows.jsonl", rows)
    n = len(rows)
    print(f"{args.run}: {n} claims | verified {sum(r['verified'] for r in rows) / n:.1%} | exact 4-way "
          f"{sum(r['correct'] for r in rows) / n:.1%} | rubric {sum(r['score'] for r in rows) / n:.1f} | "
          f"harm {sum(r['cost'] for r in rows) / n:.3f} | brier {sum(r['brier'] for r in rows) / n:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
