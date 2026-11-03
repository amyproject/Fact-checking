"""Run one (model, harness config) on a v2 split. The harness is a set of feature flags.

    python 13_run_v2.py --model glm-5.3 --features retrieve,questions,quote_check --split dev
    python 13_run_v2.py --model glm-5.3 --system rag --split test --robust

Every answer is the v2 task, harder than v1: a 4-way label (supported / refuted /
conflicting / not_enough_evidence), a confidence, the claim as understood, its atomic
sub-claims with labels, 2-5 verification questions each answered from cited evidence,
a rationale and citations.

Features (models.json v2.features), each independently switchable:
  retrieve       search and label from numbered passages (otherwise closed-book)
  questions      verification questions drive the searches and must be answered
  quote_check    deterministic verbatim check on every quote; one repair turn
  decompose      atomic sub-claims first; each gets its own questions; label by the weakest
  numcheck       deterministic number extraction: claim figures vs evidence figures
  rerank         fetch 2x passages, keep the 10 best by BM25 against claim + questions
  router         a cheap triage call skips retrieval for claims search can't settle
  adaptive_vote  4 extra samples at T=0.7 only when confidence < 0.7
  date_guard     drop passages dated after the video (when the backend knows dates)
  critic         a second pass audits the verdict against the evidence
--robust adds a paraphrased-claim run and a shuffled-evidence run (rubric B1-B3).

Writes results_v2/<model>/<system>/preds.jsonl and usage.json.
"""
import argparse
import asyncio
import json
import math
import random
import re
import time
from collections import Counter

import common
from common import DATA, MODELS, REGISTRY, ROOT, USAGE, chat, parse_json, quote_in, read_jsonl, resolve, \
    usage_cost, write_jsonl
from retrieval import RETRIEVER, parse_claim_date, search, tokens
from rubric_v2 import LABELS4

run1 = __import__("importlib").import_module("02_run")
V2 = REGISTRY["v2"]
RESULTS = ROOT / ("results_v2_mock" if common.MOCK else "results_v2")
FEATURES = tuple(V2["features"])
MAX_PASSAGES = 10

RULES = """Labels (choose exactly one):
- supported: reliable evidence shows the claim is true as stated
- refuted: reliable evidence shows the claim is false
- conflicting: parts are true and parts false, or it is technically true but misleading (cherry-picked, missing context)
- not_enough_evidence: the evidence does not settle it
Judge as of the video date, not today. Confidence is your probability (0-1) that your label is right."""

SCHEMA = """Return JSON: {{"label": "supported|refuted|conflicting|not_enough_evidence", "confidence": 0.0-1.0,
 "claim_understood": "who claims what, when, with which number",
 "subclaims": [{{"text": "...", "label": "..."}}],
 "questions": [{{"q": "...", "answer": "... or 'no answer found'", "citations": [{cite}]}}],
 "rationale": "2-4 sentences", "citations": [{cite}]}}"""

CLOSED = """Fact-check one claim from a short video. You have no search tool; use only what you know.

VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}
CLAIM: {claim}

{rules}
Ask 2-5 verification questions and answer each from memory. For any decisive label, cite 1-3 sources you are confident exist (URL + the sentence that backs it); never invent sources.
""" + SCHEMA.replace("{cite}", '{{"url": "...", "quote": "..."}}')

TRIAGE = """Classify this claim from a short video before fact-checking it.

CLAIM: {claim}
VIDEO DATE: {date}

Return JSON: {{"checkable": true/false, "needs_retrieval": true/false, "kind": "number|event|quote|causal|opinion|prediction|other"}}
needs_retrieval is false only for opinions, predictions, or claims about the speaker's own feelings."""

DECOMPOSE = """Split this claim from a short video into atomic sub-claims a fact-checker can verify independently, and write 1-2 search-ready verification questions for each (names, places, numbers, dates).

VIDEO DATE: {date}
CLAIM: {claim}

Return JSON: {{"subclaims": [{{"text": "...", "questions": ["..."]}}]}}"""

LABEL = """Fact-check one claim from a short video using ONLY the numbered evidence.

VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}
CLAIM: {claim}
{plan}{numbers}
EVIDENCE:
{evidence}

{rules}
Answer every verification question from the evidence, citing passage numbers and copying the exact sentence; write "no answer found" when the evidence is silent. For conflicting, cite both sides.
""" + SCHEMA.replace("{cite}", '{{"id": 1, "quote": "..."}}')

CRITIC = """Audit this fact-check. Is the label consistent with the cited evidence and the answers to the questions? Check numbers and dates.

CLAIM: {claim}
ANSWER: {answer}
EVIDENCE:
{evidence}

Return JSON: {{"consistent": true/false, "label": "the label the evidence supports", "reason": "one sentence"}}"""

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent|per cent|million|billion|trillion|thousand)?", re.I)
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def numbers(text):
    out = []
    for m in NUM.finditer(text or ""):
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        out.append((v * SCALE.get(unit, 1), "%" if unit in ("%", "percent", "per cent") else "", m.group(0).strip()))
    return out


def number_check(claim, evidence):
    """Deterministic: for each figure in the claim, the closest figure (same unit) in the evidence."""
    lines = []
    for v, unit, raw in numbers(claim):
        if 1900 <= v <= 2100 and not unit:
            continue  # a year, handled as time
        best = None
        for i, e in enumerate(evidence, 1):
            for w, u2, r2 in numbers(e["text"]):
                if u2 == unit and w > 0:
                    gap = abs(w - v) / max(abs(v), 1e-9)
                    if best is None or gap < best[0]:
                        best = (gap, i, r2)
        if best:
            verdict = "exact match" if best[0] < 0.005 else "close (within 10%)" if best[0] < 0.1 else "different"
            lines.append(f"- claim says {raw}; nearest evidence figure {best[2]} in [{best[1]}] ({verdict})")
        else:
            lines.append(f"- claim says {raw}; no comparable figure in the evidence")
    return ("NUMBER CHECK (deterministic):\n" + "\n".join(lines) + "\n") if lines else ""


def bm25(query, docs, k1=1.5, b=0.75):
    q = tokens(query)
    toks = [tokens(d) for d in docs]
    n, avg = len(docs), sum(map(len, toks)) / max(len(docs), 1)
    df = Counter(t for d in toks for t in set(d))
    scores = []
    for d in toks:
        tf = Counter(d)
        scores.append(sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5)) * tf[t] * (k1 + 1) /
                          (tf[t] + k1 * (1 - b + b * len(d) / max(avg, 1))) for t in q if t in tf))
    return scores


def clean4(x):
    x = str(x or "").strip().lower().replace(" ", "_")
    return x if x in LABELS4 else "not_enough_evidence"


def read(text, evidence=None):
    v = parse_json(text)
    def cites(lst):
        out = []
        for c in lst or []:
            if not isinstance(c, dict):
                continue
            if evidence is None:
                if c.get("url"):
                    out.append({"url": str(c["url"]), "quote": str(c.get("quote", ""))})
                continue
            try:
                i = int(c.get("id"))
            except (TypeError, ValueError):
                continue
            if 1 <= i <= len(evidence):
                out.append({"id": i, "url": evidence[i - 1]["url"], "quote": str(c.get("quote", ""))})
        return out
    try:
        conf = min(1.0, max(0.0, float(v.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"label": clean4(v.get("label")), "confidence": conf, "claim_understood": str(v.get("claim_understood", "")),
            "subclaims": [{"text": str(s.get("text", "")), "label": clean4(s.get("label"))}
                          for s in v.get("subclaims", []) or [] if isinstance(s, dict)][:6],
            "questions": [{"q": str(q.get("q", "")), "answer": str(q.get("answer", "")), "citations": cites(q.get("citations"))}
                          for q in v.get("questions", []) or [] if isinstance(q, dict) and q.get("q")][:5],
            "rationale": str(v.get("rationale", "")), "citations": cites(v.get("citations")),
            "parsed": str(v.get("label", "")).strip().lower().replace(" ", "_") in LABELS4}


async def closed_book(claim, clip, date, temperature=0.0, sample=0):
    text = await chat([{"role": "user", "content": CLOSED.format(date=date or "unknown", transcript=clip["transcript"],
                                                                  claim=claim, rules=RULES)}],
                      tag="v2_closed", temperature=temperature, sample=sample, max_tokens=2500)
    return {**read(text), "evidence": [], "route": "closed"}


async def verify(claim, clip, exclude, flags, shuffle_seed=None):
    date = parse_claim_date(clip["claim_date"])
    t0 = time.perf_counter()
    if "router" in flags:
        tri = parse_json(await chat([{"role": "user", "content": TRIAGE.format(claim=claim, date=date or "unknown")}],
                                    tag="triage", max_tokens=200))
        if tri.get("needs_retrieval") is False:
            return {**await closed_book(claim, clip, date), "route": "router: no retrieval"}
    if "retrieve" not in flags:
        return await closed_book(claim, clip, date)

    plan, queries = [], []
    if "decompose" in flags:
        d = parse_json(await chat([{"role": "user", "content": DECOMPOSE.format(date=date or "unknown", claim=claim)}],
                                  tag="decompose", max_tokens=800))
        for s in d.get("subclaims", []) or []:
            if isinstance(s, dict) and s.get("text"):
                plan.append({"text": str(s["text"]), "questions": [str(q) for q in s.get("questions", [])][:2]})
        queries = [q for p in plan for q in p["questions"]][:6]
    if not queries:
        tag = "questions" if "questions" in flags else "queries"
        text = await chat([{"role": "user", "content": (run1.QUESTIONS if "questions" in flags else run1.QUERIES).format(
            date=date or "unknown", claim=claim)}], tag=tag)
        queries = [str(q) for q in parse_json(text).get(tag, []) if str(q).strip()][:4] or [claim]

    k = 6 if "rerank" in flags else 3
    hits = await asyncio.gather(*[asyncio.to_thread(search, q, k, date, exclude) for q in queries])
    pool, seen = [], set()
    for h in (h for hs in hits for h in hs):
        if (h["url"], h["text"][:80]) in seen:
            continue
        seen.add((h["url"], h["text"][:80]))
        if "date_guard" in flags and date and h.get("date") and str(h["date"])[:10] > date:
            continue
        pool.append(h)
    if "rerank" in flags and len(pool) > MAX_PASSAGES:
        sc = bm25(claim + " " + " ".join(queries), [p["text"] for p in pool])
        pool = [p for _, p in sorted(zip(sc, pool), key=lambda x: -x[0])]
    evidence = pool[:MAX_PASSAGES]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(evidence)
    base = {"evidence": evidence, "queries": queries, "plan": plan, "route": "retrieve"}
    if not evidence:
        return {**base, "label": "not_enough_evidence", "confidence": 0.5, "claim_understood": claim, "subclaims": [],
                "questions": [{"q": q, "answer": "no answer found", "citations": []} for q in queries[:5]],
                "rationale": "No evidence was retrieved.", "citations": [], "parsed": True}

    plan_txt = ""
    if plan:
        plan_txt = "SUB-CLAIMS AND QUESTIONS:\n" + "\n".join(f"- {p['text']}\n" + "".join(f"  - {q}\n" for q in p["questions"]) for p in plan)
    elif "questions" in flags:
        plan_txt = "VERIFICATION QUESTIONS:\n" + "\n".join(f"- {q}" for q in queries) + "\n"
    nums = number_check(claim, evidence) if "numcheck" in flags else ""
    prompt = [{"role": "user", "content": LABEL.format(date=date or "unknown", transcript=clip["transcript"], claim=claim,
                                                       plan=plan_txt, numbers=nums, evidence=run1.numbered(evidence), rules=RULES)}]
    text = await chat(prompt, tag="v2_label", max_tokens=2500)
    v = read(text, evidence)
    v["numbers_checked"] = bool(nums)
    if plan and "decompose" in flags:  # label by the weakest sub-claim
        subs = [s["label"] for s in v["subclaims"]]
        if "refuted" in subs and v["label"] == "supported":
            v["label"], v["aggregated"] = "conflicting" if "supported" in subs else "refuted", True
    if "quote_check" in flags:
        bad = [c["id"] for c in v["citations"] + [c for q in v["questions"] for c in q["citations"]]
               if not quote_in(c["quote"], evidence[c["id"] - 1]["text"])]
        if (v["label"] != "not_enough_evidence" and not v["citations"]) or bad:
            problem = (f"These quotes are not in their passages: {sorted(set(bad))}." if bad
                       else f"You answered {v['label']} with no citation.")
            text = await chat(prompt + [{"role": "assistant", "content": text},
                                        {"role": "user", "content": problem + " Copy quotes exactly from the evidence, or answer not_enough_evidence. Same JSON."}],
                              tag="repair_v2", max_tokens=2500)
            v = {**read(text, evidence), "repaired": True, "numbers_checked": bool(nums)}
    if "adaptive_vote" in flags and v["confidence"] < 0.7:
        more = await asyncio.gather(*[chat(prompt, tag="v2_label", temperature=0.7, sample=i + 1, max_tokens=2500) for i in range(4)])
        labels = [v["label"]] + [read(t, evidence)["label"] for t in more]
        top, n = Counter(labels).most_common(1)[0]
        v = {**v, "label": top, "confidence": n / len(labels), "votes": dict(Counter(labels))}
    if "critic" in flags:
        cr = parse_json(await chat([{"role": "user", "content": CRITIC.format(
            claim=claim, answer=json.dumps({k: v[k] for k in ("label", "rationale", "questions")})[:3000],
            evidence=run1.numbered(evidence))}], tag="critic", max_tokens=400))
        if cr.get("consistent") is False and clean4(cr.get("label")) != v["label"]:
            v = {**v, "label": clean4(cr.get("label")), "confidence": min(v["confidence"], 0.5), "critic": cr.get("reason", "")}
    v["seconds"] = round(time.perf_counter() - t0, 2)
    return {**base, **v}


def paraphrase(claim):
    """Deterministic reframing (no model): tests whether wording, not content, drives the label."""
    c = claim.strip().rstrip(".")
    first = c.split()[0] if c.split() else ""
    keep = first.isupper() or first == "I"  # acronyms and "I" keep their case
    return f"According to what is said in the video, {c if keep else c[:1].lower() + c[1:]}."


async def run_clip(clip, gold, flags, robust):
    exclude = tuple(u for g in gold for u in (g.get("fact_checking_article"), g.get("original_claim_url")) if u)
    extracted = await run1.extract(clip)
    ex = await asyncio.gather(*[verify(c["claim"], clip, exclude, flags) for c in extracted])
    orc = await asyncio.gather(*[verify(g["claim"], clip, exclude, flags) for g in gold])
    if robust:
        para = await asyncio.gather(*[verify(paraphrase(g["claim"]), clip, exclude, flags) for g in gold])
        shuf = await asyncio.gather(*[verify(g["claim"], clip, exclude, flags, shuffle_seed=7) for g in gold])
        for v, p, s in zip(orc, para, shuf):
            v["robust"] = {"paraphrase_label": p["label"], "paraphrase_conf": p["confidence"], "shuffle_label": s["label"]}
    return {"clip_id": clip["clip_id"], "extracted": [{**c, "verdict": v} for c, v in zip(extracted, ex)],
            "oracle": {g["id"]: v for g, v in zip(gold, orc)}}


def system_name(flags):
    named = {tuple(sorted(v)): k for k, v in V2["systems"].items()}
    return named.get(tuple(sorted(flags))) or "+".join(f for f in FEATURES if f in flags)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.MODEL)
    ap.add_argument("--system", help="a named system in models.json v2.systems")
    ap.add_argument("--features", default="", help="comma-separated feature flags")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    common.MODEL = resolve(args.model)[0]
    assert common.MODEL not in (V2["ground_truth_model"],), "the ground-truth model is never evaluated"
    flags = set(V2["systems"][args.system]) if args.system else {f for f in args.features.split(",") if f}
    unknown = flags - set(FEATURES)
    assert not unknown, f"unknown features {unknown}"
    name = args.system or system_name(flags)

    claims = [c for c in read_jsonl(DATA / "v2" / "claims.jsonl") if c["split"] == args.split]
    ids = {c["clip_id"] for c in claims}
    clips = [c for c in read_jsonl(DATA / "v2" / "clips.jsonl") if c["clip_id"] in ids][: args.limit]
    by_clip = {}
    for c in claims:
        by_clip.setdefault(c["clip_id"], []).append(c)
    print(f"{common.MODEL} / {name} on {args.split}: {len(clips)} clips, flags={sorted(flags)}, retriever={RETRIEVER}")
    t0 = time.perf_counter()
    preds = await asyncio.gather(*[run_clip(cl, by_clip[cl["clip_id"]], flags, args.robust) for cl in clips])
    out = RESULTS / args.split / common.MODEL / name
    write_jsonl(out / "preds.jsonl", preds)
    m = MODELS.get(common.MODEL, {})
    cost = usage_cost(common.MODEL)
    usage = {**USAGE, "latency_s": round(USAGE["latency_s"], 2), "wall_s": round(time.perf_counter() - t0, 1),
             "cost_usd": None if cost is None else round(cost, 4), "model": common.MODEL, "model_name": m.get("name"),
             "tier": m.get("tier"), "system": name, "flags": sorted(flags), "split": args.split, "robust": args.robust,
             "clips": len(clips), "n_verdicts": sum(len(p["oracle"]) + len(p["extracted"]) for p in preds)}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    print(f"wrote {out}: {usage['calls']} calls, {usage['n_verdicts']} verdicts")


if __name__ == "__main__":
    asyncio.run(main())
