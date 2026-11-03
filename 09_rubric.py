"""Score every claim of a run on the 50-item rubric (rubric.py) and name its failure mode.

    python 09_rubric.py --run glm-5.3/rag_qa_verify

Needs 03_score.py (rows.jsonl) and 06_traps.py (data/claim_details.json) first. Auto items
come from the scored row. Judge items go to JUDGE_MODEL in one call per claim, blind: the
judge sees the answer, the evidence and the fact-checker's reference, never the model or
system name. Writes results/<model>/<system>/rubric.jsonl.
"""
import argparse
import asyncio
import json

from common import DATA, JUDGE_MODEL, RESULTS, chat, parse_json, read_jsonl, write_jsonl
from retrieval import blocked
from rubric import AREAS, RUBRIC, TRAPS

JUDGE = """You are auditing an automated fact-check of one claim from a short video. You will see the claim, the video transcript excerpt, the system's answer with its citations, and the professional fact-checker's reference verdict. Score each criterion below: 1 if the answer meets it, 0 if not, null if it does not apply to this answer.

Be strict: a criterion passes only if a careful editor would agree without caveats. Judge the answer, not the fact-checker; the reference is there so you know what the evidence showed.

CLAIM: {claim}
VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}
THIS CLAIM'S TRAPS: {traps}

SYSTEM ANSWER
Label: {label}
Rationale: {rationale}
Citations:
{citations}
System's own extracted version of the claim: {extracted}

FACT-CHECKER REFERENCE
Label: {gold} ({gold_raw})
Justification: {justification}
Fact-checker's research:
{qa}

CRITERIA
{criteria}

Return JSON: {{"scores": {{"<criterion id>": 1, ...}}}}"""


def applies(rule, row, det):
    if rule == "always":
        return True
    if rule.startswith("trap:"):
        return rule[5:] in det["traps"]
    return {
        "gold_decisive": row["gold"] != "unverified",
        "gold_unverified": row["gold"] == "unverified",
        "pred_decisive": row["pred"] != "unverified",
        "pred_unverified": row["pred"] == "unverified",
        "has_citations": row["n_citations"] > 0,
        "has_traps": bool(det["traps"]),
        "matched": bool(row["matched"]),
    }[rule]


def auto_score(item, row, det, ctx):
    c = row["citations"]
    own = (det["fact_checking_article"], det["selection"]["original_post"])
    return int({
        "A1": row["correct"], "A2": not row["flip"], "A3": row["pred"] != "unverified",
        "A4": row["pred"] == "unverified", "A9": row["matched"], "A10": row["e2e_correct"],
        "B1": row["n_citations"] > 0, "B2": all(x["resolves"] for x in c), "B3": all(x["quote_found"] for x in c),
        "B4": any(x["ok"] for x in c), "B5": not any(blocked(x.get("url", "")) for x in c),
        "B6": not any(blocked(x.get("url", ""), own) and not blocked(x.get("url", "")) for x in c),
        "B9": sum(x["ok"] for x in c) >= len(c) / 2,
        "C6": 1 <= row.get("n_extracted", 0) <= 15,
        "D9": 8 <= len(str(row["rationale"]).split()) <= 120,
        "E4": row.get("parsed", 1), "E5": ctx["calls_per_claim"] <= 3 * ctx["base_calls_per_claim"],
        "E8": row["n_citations"] <= 3,
    }[item["id"]])


def failure_mode(row):
    """One name per answer, most severe first."""
    if not row.get("parsed", 1):
        return "unparseable output"
    if row["flip"]:
        return "flip (supported <-> contradicted)"
    if row["pred"] == "unverified" and row["gold"] != "unverified":
        return "wrong abstention"
    if row["pred"] != "unverified" and row["gold"] == "unverified":
        return "over-claim"
    if not row["correct"]:
        return "wrong label"
    if row["backed"]:
        return "correct and backed"
    c = row["citations"]
    if not c:
        return "correct, no citation"
    if not any(x["resolves"] for x in c):
        return "correct, cited source doesn't resolve"
    if not any(x["quote_found"] for x in c):
        return "correct, quote not in source"
    return "correct, judge rejects quote"


async def score(row, det, pred, ctx):
    items, judge_items = {}, []
    for it in RUBRIC:
        if not applies(it["applies"], row, det):
            items[it["id"]] = None
        elif it["method"] == "auto":
            items[it["id"]] = auto_score(it, row, det, ctx)
        else:
            judge_items.append(it)
    if judge_items:
        v = pred["oracle"][row["id"]]
        cites = "\n".join(
            f"- \"{c['quote']}\" ({c.get('url', '')}; resolves={c['resolves']}, verbatim={c['quote_found']})"
            + (f"\n  passage: {v['evidence'][c['id'] - 1]['text'][:400]}" if 'id' in c and v.get('evidence') else "")
            for c in row["citations"]) or "(none)"
        qa = "\n".join(f"- Q: {x['q']}\n  A: {x['a'][:300]}" for x in det["qa"][:4]) or "(none recorded)"
        prompt = JUDGE.format(
            claim=row["claim"], date=row["claim_date"], transcript=det["transcript"][:1500],
            traps="; ".join(f"{TRAPS[t]['name']}: {TRAPS[t]['why']}" for t in det["traps"]) or "none",
            label=row["pred"], rationale=row["rationale"] or "(none)", citations=cites,
            extracted=row.get("e2e_claim") or "(the system did not extract this claim)",
            gold=row["gold"], gold_raw=row["gold_raw"], justification=det["justification"][:800], qa=qa,
            criteria="\n".join(f"- {it['id']}: {it['criterion']}" for it in judge_items))
        got = parse_json(await chat([{"role": "user", "content": prompt}], tag="rubric", model=JUDGE_MODEL,
                                    judge=True, max_tokens=1024)).get("scores", {})
        for it in judge_items:
            s = got.get(it["id"])
            items[it["id"]] = None if s is None else int(s in (1, True, "1", "true"))
    areas = {}
    for a in AREAS:
        vals = [items[it["id"]] for it in RUBRIC if it["area"] == a and items[it["id"]] is not None]
        areas[a] = sum(vals) / len(vals) if vals else None
    live = {a: s for a, s in areas.items() if s is not None}
    total = 100 * sum(AREAS[a]["weight"] * s for a, s in live.items()) / sum(AREAS[a]["weight"] for a in live)
    return {"id": row["id"], "items": items, "areas": areas, "score": round(total, 2), "failure": failure_mode(row)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="<model>/<system> under results/")
    args = ap.parse_args()
    model = args.run.split("/")[0]
    details = json.loads((DATA / "claim_details.json").read_text())
    claims = {c["id"]: c for c in read_jsonl(DATA / "claims.jsonl")}
    rows = read_jsonl(RESULTS / args.run / "rows.jsonl")
    preds = {p["clip_id"]: p for p in read_jsonl(RESULTS / args.run / "preds.jsonl")}

    def calls_per_claim(run):
        u = json.loads((RESULTS / run / "usage.json").read_text())
        n = sum(len(p["oracle"]) + len(p["extracted"]) for p in read_jsonl(RESULTS / run / "preds.jsonl"))
        return u["calls"] / max(n, 1)
    base = f"{model}/closed_book" if (RESULTS / model / "closed_book" / "usage.json").exists() else args.run
    ctx = {"calls_per_claim": calls_per_claim(args.run), "base_calls_per_claim": calls_per_claim(base)}

    out = await asyncio.gather(*[
        score({**r, "claim": claims[r["id"]]["claim"], "claim_date": claims[r["id"]]["claim_date"],
               "gold_raw": claims[r["id"]]["gold_raw"]}, details[r["id"]], preds[r["clip_id"]], ctx)
        for r in rows])
    write_jsonl(RESULTS / args.run / "rubric.jsonl", out)
    mean = sum(o["score"] for o in out) / len(out)
    print(f"{args.run}: rubric {mean:.1f}/100 over {len(out)} claims; "
          + ", ".join(f"{AREAS[a]['name']} {100 * sum(o['areas'][a] or 0 for o in out) / len(out):.0f}" for a in AREAS))


if __name__ == "__main__":
    asyncio.run(main())
