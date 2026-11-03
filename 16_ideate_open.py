"""Ask an open model for new harness and fine-tuning ideas, from the dev split's failures.

    python 16_ideate_open.py --run dev/glm-5.3/rag+numcheck

The ideation model (models.json v2.ideation_model, an open model) sees the failure analysis
of a dev run: label confusions, failed gates and rubric checks, trap-level accuracy and
five failed answers. It proposes ideas with a hypothesis, mechanism, expected gain, added
seconds per claim, build hours, a step diagram and a reference. Every arXiv reference is
looked up on the arXiv API and marked verified or unverified (models invent citations).

Writes results_v2/ideas_open.json.
"""
import argparse
import asyncio
import json
import re
import urllib.request
from collections import Counter

import common
from common import MODELS, REGISTRY, ROOT, chat, parse_json, read_jsonl
from rubric_v2 import RUBRIC

V2 = REGISTRY["v2"]
RESULTS = ROOT / ("results_v2_mock" if common.MOCK else "results_v2")

PROMPT = """You are improving an open-weight fact-checking system for claims in short-video transcripts. Below is how the current system fails on a development split. Propose 8 new improvements that are NOT already in the system, aimed at its largest failure buckets. Mix harness changes (prompting, retrieval, tools, verification, routing) and post-training (SFT data, preference tuning, RL). Prefer ideas that also keep it fast.

CURRENT SYSTEM: model {model}; harness features {flags}
DEV RESULTS ({n} claims): verified accuracy {verified:.0%}, exact 4-way accuracy {exact:.0%}, mean rubric {score:.1f}/100, {sec:.1f} model-seconds per claim
LABEL CONFUSIONS (fact-checker -> system): {conf}
MOST-FAILED RUBRIC CHECKS: {checks}
FAILED GATES: {gates}
FIVE FAILED ANSWERS:
{examples}

For each idea return: name, type ("harness" or "post-training"), hypothesis (which failure it fixes, citing the numbers above), mechanism (what to build), expected_gain_pp, added_seconds (per claim; negative if it saves time), build_hours, steps (3-6 short diagram steps from input to output), reference (an arXiv id like arXiv:2305.14251, or a named method; write "none" rather than guessing).

Return JSON: {{"ideas": [{{...}}]}}"""


def arxiv_titles(ids):
    if not ids:
        return {}
    try:
        with urllib.request.urlopen(f"http://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results={len(ids)}", timeout=30) as r:
            x = r.read().decode()
    except Exception:
        return {}
    out = {}
    for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
        m = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e)
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        if m and t and "Error" not in t.group(1):
            out[m.group(1)] = " ".join(t.group(1).split())
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    d = RESULTS / args.run
    rows = read_jsonl(d / "rows.jsonl")
    usage = json.loads((d / "usage.json").read_text())
    n = len(rows)
    fails = Counter(it["id"] for r in rows for it in RUBRIC if r["items"].get(it["id"]) == 0)
    crit = {it["id"]: it["criterion"] for it in RUBRIC}
    bad = [r for r in rows if not r["verified"]][:5]
    model = V2["ideation_model"]
    text = await chat([{"role": "user", "content": PROMPT.format(
        model=usage["model"], flags=", ".join(usage["flags"]) or "none", n=n,
        verified=sum(r["verified"] for r in rows) / n, exact=sum(r["correct"] for r in rows) / n,
        score=sum(r["score"] for r in rows) / n, sec=usage["latency_s"] / max(usage["n_verdicts"], 1),
        conf=dict(Counter(f"{r['gold']}->{r['pred']}" for r in rows if not r["correct"]).most_common(6)),
        checks="; ".join(f"{k} {crit[k]} (failed {v}x)" for k, v in fails.most_common(8)),
        gates=dict(Counter(g for r in rows for g, ok in r["gates"].items() if not ok)),
        examples="\n".join(f"- claim: {r.get('claim_understood', '')[:140]} | gold {r['gold']} | said {r['pred']} | rationale: {r['rationale'][:160]}"
                           for r in bad))}], tag="ideas", model=model, max_tokens=4000, judge=True)
    ideas = [i for i in parse_json(text).get("ideas", []) if isinstance(i, dict) and i.get("name")]
    ids = sorted({m for i in ideas for m in re.findall(r"(\d{4}\.\d{4,5})", str(i.get("reference", "")))})
    found = arxiv_titles(ids)
    for i in ideas:
        m = re.search(r"(\d{4}\.\d{4,5})", str(i.get("reference", "")))
        i["reference_check"] = ("verified: " + found[m.group(1)]) if m and m.group(1) in found else \
            ("unverified arXiv id" if m else "no arXiv id")
        i["proposed_by"] = MODELS[model]["name"]
    out = {"model": MODELS[model]["name"], "from_run": args.run, "ideas": ideas}
    (RESULTS / "ideas_open.json").write_text(json.dumps(out, indent=2))
    print(f"{len(ideas)} ideas from {MODELS[model]['name']}; references verified: "
          f"{sum(i['reference_check'].startswith('verified') for i in ideas)}/{len(ideas)}")


if __name__ == "__main__":
    asyncio.run(main())
