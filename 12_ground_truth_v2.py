"""Ground-truth annotations for every v2 claim, from the strongest frontier model at max reasoning effort.

    python 12_ground_truth_v2.py

The fact-checker's label stays the gold label. The ground-truth model (models.json
v2.ground_truth_model, default Claude Opus 5.5, reasoning effort "high", the maximum
OpenRouter exposes) reads the claim, the transcript clip and the fact-checker's verdict
and explanation (plus AVeriTeC's question-answer evidence when it exists) and writes the
structured answer key the v2 rubric scores against:
  questions   3-5 verification questions a fact-checker needs answered, with gold answers
  subclaims   the atomic sub-claims, each with its own 4-way label
  quantities  every number/date in the claim and the true value as of the video date
  traps       which trap types apply (rubric.TRAPS ids), and a 1-5 difficulty rating
  label_check whether it agrees with the fact-checker; disagreements go to a review queue
This model also judges, and is excluded from every evaluation (models.json v2.evaluated).

Writes data/v2/ground_truth.jsonl and data/v2/adjudication.jsonl (label disagreements).
"""
import asyncio
import json
import os

from common import DATA, MODELS, REGISTRY, chat, parse_json, read_jsonl, write_jsonl
from rubric import TRAPS
from rubric_v2 import LABELS4

V2 = REGISTRY["v2"]
GT_MODEL = os.environ.get("GT_MODEL", V2["ground_truth_model"])
GT_REASONING = {"effort": os.environ.get("GT_EFFORT", V2["ground_truth_reasoning"]["effort"])}
assert GT_MODEL not in V2["evaluated"], "the ground-truth model must not be evaluated"

PROMPT = """You are building the answer key for a fact-checking benchmark. A professional fact-checker has already verified the claim below; their verdict is authoritative. Your job is to make the answer key explicit and checkable, so that automated fact-checkers can be scored on how they reached a verdict, not only on the verdict.

CLAIM: {claim}
SPEAKER / SETTING: {speaker} / {setting}
VIDEO DATE: {date}
TRANSCRIPT EXCERPT (the claim is spoken somewhere in here): {transcript}

FACT-CHECKER VERDICT: {label} (original rating: {rating})
FACT-CHECKER EXPLANATION: {justification}
FACT-CHECKER RESEARCH (may be empty):
{qa}

Write, as of the video date:
1. "questions": 3-5 verification questions a fact-checker must answer to reach this verdict, each with the "answer" supported by the fact-checker's explanation or research, and "kind" (who/what/when/how_many/context/attribution). Make them specific and searchable; at least one must probe context if the claim could be technically true but misleading.
2. "subclaims": the claim split into atomic, independently checkable sub-claims, each with its label from {labels}.
3. "quantities": every number, percentage, amount or date in the claim, with "stated" (as the speaker said it), "actual" (the correct value as of the video date, or null if unknown) and "unit".
4. "traps": the ids from this list that apply: {traps}.
5. "difficulty": 1 (a single lookup) to 5 (multi-step reasoning over scarce or conflicting evidence).
6. "label_check": {{"agrees": true/false, "label": your own 4-way verdict, "note": one sentence if you disagree}}.

Return only JSON with those six keys."""


async def annotate(c, clip, qa):
    text = await chat([{"role": "user", "content": PROMPT.format(
        claim=c["claim"], speaker=c.get("speaker") or "unknown", setting=c.get("setting") or "video",
        date=c.get("claim_date") or "unknown", transcript=clip["transcript"][:3000], label=c["label4"],
        rating=c.get("gold_raw") or c.get("rating_raw", ""), justification=(c.get("justification") or "")[:2000],
        qa="\n".join(f"- Q: {x['q']}\n  A: {x['a'][:400]}" for x in qa[:6]) or "(none)",
        labels=", ".join(LABELS4), traps=", ".join(TRAPS))}],
        tag="ground_truth", model=GT_MODEL, judge=True, max_tokens=6000, extra={"reasoning": GT_REASONING})
    g = parse_json(text)
    lc = g.get("label_check") if isinstance(g.get("label_check"), dict) else {}
    return {
        "id": c["id"],
        "questions": [q for q in g.get("questions", []) if isinstance(q, dict) and q.get("q")][:5],
        "subclaims": [s for s in g.get("subclaims", []) if isinstance(s, dict) and s.get("text")][:6],
        "quantities": [q for q in g.get("quantities", []) if isinstance(q, dict)][:8],
        "traps": [t for t in g.get("traps", []) if t in TRAPS],
        "difficulty": int(g.get("difficulty", 3)) if str(g.get("difficulty", "")).isdigit() else 3,
        "label_check": {"agrees": lc.get("agrees") is not False, "label": lc.get("label"), "note": lc.get("note", "")},
        "model": MODELS.get(GT_MODEL, {}).get("name", GT_MODEL), "effort": GT_REASONING["effort"],
    }


async def main():
    claims = read_jsonl(DATA / "v2" / "claims.jsonl")
    clips = {c["clip_id"]: c for c in read_jsonl(DATA / "v2" / "clips.jsonl")}
    v1 = json.loads((DATA / "claim_details.json").read_text())
    out = await asyncio.gather(*[annotate(c, clips[c["clip_id"]], v1.get(c["id"], {}).get("qa", [])) for c in claims])
    write_jsonl(DATA / "v2" / "ground_truth.jsonl", out)
    queue = [{"id": o["id"], "fact_checker": c["label4"], "model": o["label_check"]["label"], "note": o["label_check"]["note"]}
             for o, c in zip(out, claims) if not o["label_check"]["agrees"]]
    write_jsonl(DATA / "v2" / "adjudication.jsonl", queue)
    print(f"ground truth for {len(out)} claims by {GT_MODEL} (effort {GT_REASONING['effort']}); "
          f"{len(queue)} label disagreements queued for review")


if __name__ == "__main__":
    asyncio.run(main())
