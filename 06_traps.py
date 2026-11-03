"""Tag every test claim with its traps and record why it was selected. No model calls.

    python 06_traps.py

Joins each claim back to its AVeriTeC row for the fact-checker's strategies, location and
question-answer evidence, then applies the rule-based trap detectors in rubric.TRAPS.
Writes data/claim_details.json (one entry per claim id) and prints trap counts.
"""
import ast
import json
import re
from collections import Counter

import pandas as pd

from common import DATA, read_jsonl
from rubric import TRAPS

NUM = re.compile(r"\d|%|\b(?:hundred|thousand|million|billion|trillion|percent|half|double|twice|triple)\b", re.I)
WORD = {
    "missing_context": re.compile(r"mislead|out of context|cherry|partly|partially|lacks context|missing context|exaggerat", re.I),
    "causal": re.compile(r"\b(?:because|due to|caused?|causing|led to|leads to|result(?:ed|s)? in|responsible for|thanks to)\b", re.I),
    "attribution": re.compile(r"\b(?:said|says|claimed|stated|according to|quoted|told)\b|[\"“”]", re.I),
    "temporal": re.compile(r"\b(?:now|currently|today|this year|last year|since|record|first time|ever|recently|so far|(?:19|20)\d\d)\b", re.I),
    "superlative": re.compile(r"\b(?:most|largest|biggest|first|highest|lowest|least|never|always|only|every|all-time|best|worst)\b", re.I),
}


def txt(x):
    """AVeriTeC leaves some text fields as NaN."""
    return x if isinstance(x, str) else ""


def answers(q):
    """AVeriTeC stores each question's answers as a Python-literal string."""
    a = q.get("answers")
    if isinstance(a, str):
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            a = [{"answer": a}]
    return list(a) if a is not None else []


def main():
    raw = {}
    for split in ("train", "dev"):
        df = pd.read_parquet(DATA / "raw" / f"averitec_{split}.parquet")
        for i, r in enumerate(df.itertuples(index=False)):
            raw[f"{split}-{i}"] = r
    clips = {c["clip_id"]: c for c in read_jsonl(DATA / "clips.jsonl")}
    claims = read_jsonl(DATA / "claims.jsonl")

    out, counts = {}, Counter()
    for c in claims:
        r, clip = raw[c["id"]], clips[c["clip_id"]]
        strategies = list(r.fact_checking_strategies) if r.fact_checking_strategies is not None else []
        types = c["claim_types"]
        text, just = c["claim"], c["justification"] or ""
        loc = txt(r.location_ISO_code).upper()
        n_numbers = len(re.findall(r"\d[\d,.]*", text))
        traps = {
            "exact_number": bool(NUM.search(text)) or "Numerical Claim" in types,
            "missing_context": c["gold_raw"].startswith("Conflicting") or bool(WORD["missing_context"].search(just)),
            "no_evidence": c["gold_raw"] == "Not Enough Evidence",
            "local_knowledge": bool(loc) and loc not in ("US", "GB"),
            "causal": "Causal Claim" in types or bool(WORD["causal"].search(text)),
            "compound": n_numbers >= 2 or len(re.findall(r"\band\b|;|, (?:and |but )?(?:the |it |they |he |she )", text)) >= 2,
            "attribution": bool({"Quote Verification", "Position Statement"} & set(types)) or bool(WORD["attribution"].search(text)),
            "temporal": bool(WORD["temporal"].search(text)),
            "superlative": bool(WORD["superlative"].search(text)),
            "offline_source": "Consultation" in strategies,
            "paraphrase_gap": c["located_by"] == "lexical match" and c["overlap"] < 0.5,
            "long_source": clip["video_seconds"] > 600,
        }
        tags = [k for k in TRAPS if traps[k]]
        counts.update(tags)
        qa = []
        for q in (r.questions if r.questions is not None else []):
            for a in answers(q)[:1]:
                qa.append({"q": q.get("question", ""), "a": str(a.get("answer", ""))[:600],
                           "type": a.get("answer_type", ""), "url": a.get("source_url", ""),
                           "medium": a.get("source_medium", "")})
        out[c["id"]] = {
            "traps": tags,
            "strategies": strategies,
            "location": loc,
            "reporting_source": txt(r.reporting_source),
            "justification": just,
            "fact_checking_article": c["fact_checking_article"],
            "qa": qa,
            "selection": {
                "averitec_row": c["id"],
                "original_post": c["original_claim_url"],
                "video_seconds": clip["video_seconds"],
                "clip": [clip["start"], clip["end"]],
                "located_by": c["located_by"],
                "overlap": c["overlap"],
                "claims_in_clip": sum(1 for x in claims if x["clip_id"] == c["clip_id"]),
            },
            "transcript": clip["transcript"],
            "clip_url": clip["url"],
        }
    (DATA / "claim_details.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote data/claim_details.json for {len(out)} claims")
    for k in TRAPS:
        print(f"  {TRAPS[k]['name']:<24} {counts[k]:>3}")
    print(f"  traps per claim: mean {sum(len(v['traps']) for v in out.values()) / len(out):.1f}")


if __name__ == "__main__":
    main()
