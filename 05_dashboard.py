"""Interactive HTML report: the 04_report.py tables and charts plus breakdowns, in one file.

    python 05_dashboard.py --baseline closed_book --systems rag rag_qa

Writes results/report.html, self-contained (opens offline, no CDN). The page recomputes
every metric, McNemar and the video bootstrap from the per-claim rows, so its filters
(drop the largest video, claim type, metric) rescope every chart at once. Bootstrap
intervals use the page's own seeded RNG and can differ from report.md in the last digit.
"""
import argparse
import json
from datetime import datetime

from common import DATA, MOCK, RESULTS, read_jsonl

TEMPLATE = DATA.parent / "report_template.html"
NAMES = {"closed_book": "Baseline: closed-book", "rag": "Harness A: search + cite",
         "rag_qa": "Harness B: questions + search + cite", "sft": "Fine-tuned labeler + search"}
ROW_KEYS = ("pred", "correct", "backed", "sb_correct", "flip", "matched", "e2e_pred", "e2e_correct", "rationale")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="closed_book")
    ap.add_argument("--systems", nargs="+", default=["rag", "rag_qa"])
    args = ap.parse_args()
    systems = [args.baseline] + args.systems

    rows, usage = {}, {}
    for s in systems:
        rows[s] = {r["id"]: {**{k: r[k] for k in ROW_KEYS},
                             "cites": [{k: c.get(k) for k in ("url", "quote", "resolves", "quote_found", "backs", "ok")}
                                       for c in r["citations"]]}
                   for r in read_jsonl(RESULTS / s / "rows.jsonl")}
        u = RESULTS / s / "usage.json"
        usage[s] = json.loads(u.read_text()) if u.exists() else {}
    # only claims every system was scored on, so all charts compare like with like
    ids = set.intersection(*(set(r) for r in rows.values()))
    clips = {c["clip_id"]: c for c in read_jsonl(DATA / "clips.jsonl")}
    claims = [{"id": c["id"], "video": c["video"], "claim": c["claim"], "gold": c["gold"],
               "gold_raw": c["gold_raw"], "types": c["claim_types"], "speaker": (c["speaker"] or "").strip(),
               "date": c["claim_date"], "url": clips[c["clip_id"]]["url"]}
              for c in read_jsonl(DATA / "claims.jsonl") if c["id"] in ids]

    payload = {
        "meta": {"baseline": args.baseline, "systems": systems, "names": {s: NAMES.get(s, s) for s in systems},
                 "model": usage[args.baseline].get("model", "?"),
                 "retriever": next((usage[s].get("retriever") for s in args.systems if usage[s].get("retriever")), None),
                 "mock": MOCK, "generated": datetime.now().strftime("%Y-%m-%d %H:%M")},
        "claims": claims,
        "rows": {s: {i: rows[s][i] for i in ids} for s in systems},
        "usage": usage,
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out = RESULTS / "report.html"
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/{}", blob))
    print(f"wrote {out} ({len(claims)} claims, {len(systems)} systems{', MOCK DATA' if MOCK else ''})")


if __name__ == "__main__":
    main()
