"""Score one system: one row per gold claim, saved for the stats in 04_report.py.

    python 03_score.py --run glm-5.3/rag      # <model>/<system> under results/

Per gold claim:
  correct     oracle label == fact-checker label (gold claim given to the model)
  backed      label is unverified, or at least one citation checks out:
                the URL resolves, the quoted sentence is really in the source,
                and the judge agrees that sentence backs the label
  sb_correct  correct AND backed  ("source-backed accuracy", the headline metric)
  flip        supported <-> contradicted mix-up, the worst error for a fact-checker
  matched     the system extracted this claim from the transcript (judge decides)
  e2e_correct matched AND the extracted claim's label is correct

Writes results/<model>/<system>/rows.jsonl.
"""
import argparse
import asyncio
import hashlib
import json
import re
import urllib.request

from common import CACHE, DATA, JUDGE_MODEL, RESULTS, chat, parse_json, quote_in, read_jsonl, write_jsonl
from retrieval import UA, blocked

MATCH = """Does any extracted claim assert the same checkable fact as the reference claim (same subject and the same number, event or quote)? Wording can differ; a broader or vaguer claim does not count.

REFERENCE: {gold}
EXTRACTED:
{extracted}

Return JSON: {{"match": <number of the matching extracted claim, or 0 if none>}}"""

BACKS = """A fact-checker labeled a claim and cited a source sentence. Taken on its own, does the sentence back the label? For "supported" it must show the claim is true; for "contradicted" it must show the claim is false or misleading.

CLAIM: {claim}
LABEL: {label}
SOURCE SENTENCE: {quote}

Return JSON: {{"backs": true or false}}"""


def fetch(url):
    """(status, page_text). Cached; status 0 means unreachable."""
    f = CACHE / "fetch" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if f.exists():
        d = json.loads(f.read_text())
        return d["status"], d["text"]
    status, text = 0, ""
    if url.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as r:
                status = r.status
                raw = r.read(3_000_000).decode("utf-8", "ignore")
                text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
                text = re.sub(r"<[^>]+>", " ", text)[:300_000]
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception:
            status = 0
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"status": status, "text": text}))
    return status, text


async def check_citation(claim, verdict, cite):
    """Annotate one citation with resolves / quote_found / backs / ok."""
    if "id" in cite:  # retrieved passage: we already hold the text the model saw
        source, resolves = verdict["evidence"][cite["id"] - 1]["text"], True
    else:             # closed-book: the model named a URL from memory; go get it
        status, source = await asyncio.to_thread(fetch, cite["url"])
        resolves = 200 <= status < 400 or status in (401, 403, 429)  # bot-blocked pages still exist
    found = bool(source) and quote_in(cite["quote"], source)
    backs = False
    if found:
        text = await chat([{"role": "user", "content": BACKS.format(
            claim=claim, label=verdict["label"], quote=cite["quote"])}], tag="backs", model=JUDGE_MODEL, judge=True)
        backs = parse_json(text).get("backs") is True
    return {**cite, "resolves": resolves, "quote_found": found, "backs": backs, "ok": found and backs}


async def score_verdict(claim, verdict):
    checked = await asyncio.gather(*[check_citation(claim, verdict, c) for c in verdict["citations"]])
    backed = verdict["label"] == "unverified" or any(c["ok"] for c in checked)
    return checked, backed


async def match(gold_claim, extracted):
    if not extracted:
        return 0
    listing = "\n".join(f"{i}. {e['claim']}" for i, e in enumerate(extracted, 1))
    text = await chat([{"role": "user", "content": MATCH.format(gold=gold_claim, extracted=listing)}],
                      tag="match", model=JUDGE_MODEL, judge=True)
    try:
        m = int(parse_json(text).get("match", 0))
    except (TypeError, ValueError):
        return 0
    return m if 1 <= m <= len(extracted) else 0


async def score_claim(g, pred, run):
    v = pred["oracle"][g["id"]]
    checked, backed = await score_verdict(g["claim"], v)
    m = await match(g["claim"], pred["extracted"])
    e2e_label = pred["extracted"][m - 1]["verdict"]["label"] if m else None
    correct = v["label"] == g["gold"]
    return {
        "id": g["id"], "run": run, "video": g["video"], "clip_id": g["clip_id"], "gold": g["gold"], "pred": v["label"],
        "correct": int(correct), "backed": int(backed), "sb_correct": int(correct and backed),
        "flip": int({g["gold"], v["label"]} == {"supported", "contradicted"}),
        "matched": int(m > 0), "e2e_pred": e2e_label, "e2e_correct": int(e2e_label == g["gold"]),
        "n_citations": len(checked), "n_citations_ok": sum(c["ok"] for c in checked),
        "n_citations_resolve": sum(c["resolves"] for c in checked),
        "citations": checked, "rationale": v.get("rationale", ""),
        "parsed": int(v.get("parsed", True)), "repaired": int(bool(v.get("repaired"))),
        "search_rounds": v.get("search_rounds", 0), "votes": v.get("votes"),
        "leak_cite": int(any(blocked(c.get("url", ""), (g["fact_checking_article"], g["original_claim_url"]))
                             for c in checked)),
        "e2e_claim": pred["extracted"][m - 1]["claim"] if m else None, "n_extracted": len(pred["extracted"]),
        "queries": v.get("queries", []),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="<model>/<system> under results/")
    args = ap.parse_args()

    preds = {p["clip_id"]: p for p in read_jsonl(RESULTS / args.run / "preds.jsonl")}
    gold = [g for g in read_jsonl(DATA / "claims.jsonl") if g["clip_id"] in preds]
    rows = await asyncio.gather(*[score_claim(g, preds[g["clip_id"]], args.run) for g in gold])
    write_jsonl(RESULTS / args.run / "rows.jsonl", rows)

    n = len(rows)
    cites = sum(r["n_citations"] for r in rows)
    print(f"{args.run}: {n} gold claims | label acc {sum(r['correct'] for r in rows) / n:.1%} | "
          f"source-backed acc {sum(r['sb_correct'] for r in rows) / n:.1%} | "
          f"end-to-end {sum(r['e2e_correct'] for r in rows) / n:.1%} | "
          f"extraction recall {sum(r['matched'] for r in rows) / n:.1%} | "
          f"citations ok {sum(r['n_citations_ok'] for r in rows)}/{cites}")


if __name__ == "__main__":
    asyncio.run(main())
