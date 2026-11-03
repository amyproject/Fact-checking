"""Run one (model, system) pair over the locked test set.

    python 02_run.py --model glm-5.3 --system closed_book    # baseline: off-the-shelf, no retrieval
    python 02_run.py --model glm-5.3 --system rag            # A: search -> label from evidence, cite passages
    python 02_run.py --model glm-5.3 --system rag_qa         # B: verification questions -> search each -> label
    python 02_run.py --model glm-5.3 --system rag_qa_verify  # C: B + verbatim-quote check and one repair turn
    python 02_run.py --model glm-5.3 --system agent          # D: C + up to 2 extra search rounds
    python 02_run.py --model glm-5.3 --system vote           # E: C sampled 5x at T=0.7, majority label

Models are keys in models.json (open models on Fireworks, frontier models on OpenRouter).
For one model, every system shares the claim-extraction step (same prompt, temperature 0,
cached), so differences between systems come from verification. Each system labels two
claim sets per clip:
  extracted: claims the model pulled out of the transcript (end-to-end score)
  oracle:    the fact-checker's gold claims (isolates labeling + citation quality)

Writes results/<model>/<system>/preds.jsonl and usage.json.
"""
import argparse
import asyncio
import json
from collections import Counter

import common
from common import DATA, LABELS, MODELS, RESULTS, SYSTEMS, USAGE, chat, parse_json, quote_in, read_jsonl, \
    resolve, usage_cost, write_jsonl
from retrieval import RETRIEVER, parse_claim_date, search

LABEL_RULES = """Labels:
- supported: reliable evidence shows the claim is true
- contradicted: reliable evidence shows the claim is false or misleading
- unverified: the evidence does not settle it either way
Judge the claim as of the video date, not today."""

EXTRACT = """Below is the transcript of a short video posted around {date}. List every checkable factual claim in it: specific statements about the world that a fact-checker could verify (numbers, events, quotes, causes, records). Skip opinions, predictions, jokes and questions.

Rewrite each claim so it stands on its own: resolve pronouns and add the speaker, place or time from the transcript where needed. Keep the speaker's numbers exactly as said.

Return JSON: {{"claims": [{{"claim": "...", "quote": "the words in the transcript it comes from"}}]}}

TRANSCRIPT: {transcript}"""

VERIFY_CLOSED = """Fact-check one claim from a short video.

VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}
CLAIM: {claim}

Use only what you already know; you have no search tool. {rules}
For supported or contradicted, cite 1-3 specific sources you are confident exist: the URL and the sentence in it that backs your label. Do not invent sources; if you can't name one, answer unverified.

Return JSON: {{"label": "supported|contradicted|unverified", "rationale": "one or two sentences", "citations": [{{"url": "...", "quote": "..."}}]}}"""

QUERIES = """Write 2-3 short web search queries that would find evidence for or against this claim. Include names, places and numbers from the claim.

VIDEO DATE: {date}
CLAIM: {claim}

Return JSON: {{"queries": ["...", "..."]}}"""

QUESTIONS = """A fact-checker is verifying this claim from a short video. Write 2-4 questions whose answers would settle it (who/what/when/how many, and whether the context changes the meaning). Each question should work as a web search.

VIDEO DATE: {date}
CLAIM: {claim}

Return JSON: {{"questions": ["...", "..."]}}"""

MORE = """You are gathering evidence to fact-check a claim from a short video. Here is what the searches found so far.

VIDEO DATE: {date}
CLAIM: {claim}
EVIDENCE SO FAR:
{evidence}

If this evidence already settles the claim, or more searching is unlikely to help, answer done. Otherwise write 1-3 new search queries for what is still missing (don't repeat earlier queries: {asked}).

Return JSON: {{"done": true or false, "queries": ["..."]}}"""

VERIFY_EVIDENCE = """Fact-check one claim from a short video using ONLY the numbered evidence below.

VIDEO DATE: {date}
TRANSCRIPT EXCERPT: {transcript}
CLAIM: {claim}
{questions}
EVIDENCE:
{evidence}

{rules}
For supported or contradicted, cite the evidence numbers and copy the exact sentence from each that backs your label. If the evidence doesn't settle it, answer unverified with no citations.

Return JSON: {{"label": "supported|contradicted|unverified", "rationale": "one or two sentences", "citations": [{{"id": 1, "quote": "..."}}]}}"""

REPAIR = """{problem} Copy each quote exactly from the numbered evidence, or answer unverified if no passage backs your label. Return the same JSON format."""

VOTES = 5
MAX_PASSAGES = 14


def clean_label(x):
    x = str(x or "").strip().lower()
    return x if x in LABELS else "unverified"


async def extract(clip):
    text = await chat([{"role": "user", "content": EXTRACT.format(
        date=parse_claim_date(clip["claim_date"]) or "unknown", transcript=clip["transcript"])}], tag="extract")
    out = []
    for c in parse_json(text).get("claims", []):
        if isinstance(c, dict) and str(c.get("claim", "")).strip():
            out.append({"claim": str(c["claim"]).strip(), "quote": str(c.get("quote", ""))})
    return out


async def verify_closed(claim, clip, exclude):
    text = await chat([{"role": "user", "content": VERIFY_CLOSED.format(
        date=parse_claim_date(clip["claim_date"]) or "unknown", transcript=clip["transcript"],
        claim=claim, rules=LABEL_RULES)}], tag="verify_closed")
    v = parse_json(text)
    cites = [{"url": str(c.get("url", "")), "quote": str(c.get("quote", ""))}
             for c in v.get("citations", []) or [] if isinstance(c, dict) and c.get("url")]
    return {"label": clean_label(v.get("label")), "rationale": v.get("rationale", ""), "citations": cites,
            "evidence": [], "parsed": str(v.get("label", "")).lower() in LABELS}


async def gather(queries, date, exclude, evidence, seen):
    hits = await asyncio.gather(*[asyncio.to_thread(search, q, 3, date, exclude) for q in queries])
    for h in (h for hs in hits for h in hs):
        if (h["url"], h["text"][:80]) not in seen and len(evidence) < MAX_PASSAGES:
            seen.add((h["url"], h["text"][:80]))
            evidence.append(h)


def numbered(evidence):
    return "\n".join(f"[{i}] {e['url']} {e['text']}" for i, e in enumerate(evidence, 1))


def read_verdict(text, evidence):
    v = parse_json(text)
    cites = []
    for c in v.get("citations", []) or []:
        try:
            i = int(c.get("id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 1 <= i <= len(evidence):
            cites.append({"id": i, "url": evidence[i - 1]["url"], "quote": str(c.get("quote", ""))})
    return {"label": clean_label(v.get("label")), "rationale": v.get("rationale", ""), "citations": cites,
            "parsed": str(v.get("label", "")).lower() in LABELS}


def quote_problem(v, evidence):
    """Why the verdict's citations fail the verbatim check, or None if they pass."""
    if v["label"] == "unverified":
        return None
    bad = [c["id"] for c in v["citations"] if not quote_in(c["quote"], evidence[c["id"] - 1]["text"])]
    if not v["citations"]:
        return f"You answered {v['label']} but gave no citation."
    if bad:
        return f"The quotes you gave for evidence {', '.join(map(str, bad))} are not in those passages."
    return None


async def verify_evidence(claim, clip, exclude, decompose, check=False, rounds=0, temperature=0.0, sample=0):
    date = parse_claim_date(clip["claim_date"])
    tag = "questions" if decompose else "queries"
    text = await chat([{"role": "user", "content": (QUESTIONS if decompose else QUERIES).format(
        date=date or "unknown", claim=claim)}], tag=tag)
    queries = [str(q) for q in parse_json(text).get(tag, []) if str(q).strip()][:4] or [claim]
    evidence, seen = [], set()
    await gather(queries, date, exclude, evidence, seen)
    used = 0
    for _ in range(rounds):  # agent: the model decides whether to keep searching
        text = await chat([{"role": "user", "content": MORE.format(
            date=date or "unknown", claim=claim, evidence=numbered(evidence) or "(none)",
            asked="; ".join(queries))}], tag="more")
        more = parse_json(text)
        new = [str(q) for q in more.get("queries", []) if str(q).strip() and str(q) not in queries][:3]
        if more.get("done") is True or not new or len(evidence) >= MAX_PASSAGES:
            break
        queries += new
        used += 1
        await gather(new, date, exclude, evidence, seen)
    base = {"evidence": evidence, "queries": queries, "search_rounds": 1 + used}
    if not evidence:
        return {**base, "label": "unverified", "rationale": "no evidence retrieved", "citations": [], "parsed": True}
    qblock = "QUESTIONS TO ANSWER FIRST:\n" + "\n".join(f"- {q}" for q in queries) + "\n" if decompose else ""
    prompt = [{"role": "user", "content": VERIFY_EVIDENCE.format(
        date=date or "unknown", transcript=clip["transcript"], claim=claim, questions=qblock,
        evidence=numbered(evidence), rules=LABEL_RULES)}]
    text = await chat(prompt, tag="verify_evidence", temperature=temperature, sample=sample)
    v = read_verdict(text, evidence)
    problem = quote_problem(v, evidence) if check else None
    if problem:  # one repair turn with the deterministic check's feedback
        text = await chat(prompt + [{"role": "assistant", "content": text},
                                    {"role": "user", "content": REPAIR.format(problem=problem)}],
                          tag="repair", temperature=temperature, sample=sample)
        v = read_verdict(text, evidence)
        v["repaired"] = True
    return {**base, **v}


async def verify_vote(claim, clip, exclude):
    runs = await asyncio.gather(*[verify_evidence(claim, clip, exclude, True, check=True, temperature=0.7, sample=i + 1)
                                  for i in range(VOTES)])
    counts = Counter(r["label"] for r in runs)
    top = max(counts.values())
    # ties go to the first sample's label among the tied ones, so the result is deterministic
    label = next(r["label"] for r in runs if counts[r["label"]] == top)
    pick = next(r for r in runs if r["label"] == label)
    return {**pick, "votes": dict(counts)}


VERIFIERS = {
    "closed_book": verify_closed,
    "rag": lambda c, clip, ex: verify_evidence(c, clip, ex, decompose=False),
    "rag_qa": lambda c, clip, ex: verify_evidence(c, clip, ex, decompose=True),
    "rag_qa_verify": lambda c, clip, ex: verify_evidence(c, clip, ex, decompose=True, check=True),
    "agent": lambda c, clip, ex: verify_evidence(c, clip, ex, decompose=True, check=True, rounds=2),
    "vote": verify_vote,
}
assert set(VERIFIERS) == set(SYSTEMS), "models.json systems and VERIFIERS must match"


async def run_clip(clip, gold, verify):
    # never let retrieval hand back the answer key for this clip's claims
    exclude = tuple(u for g in gold for u in (g["fact_checking_article"], g["original_claim_url"]) if u)
    extracted = await extract(clip)
    ex_verdicts = await asyncio.gather(*[verify(c["claim"], clip, exclude) for c in extracted])
    or_verdicts = await asyncio.gather(*[verify(g["claim"], clip, exclude) for g in gold])
    return {
        "clip_id": clip["clip_id"],
        "extracted": [{**c, "verdict": v} for c, v in zip(extracted, ex_verdicts)],
        "oracle": {g["id"]: v for g, v in zip(gold, or_verdicts)},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.MODEL, help="key in models.json")
    ap.add_argument("--system", choices=sorted(VERIFIERS), required=True)
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    args = ap.parse_args()
    common.MODEL = resolve(args.model)[0]  # chat() reads the model under test from here

    clips = read_jsonl(DATA / "clips.jsonl")[: args.limit]
    claims = read_jsonl(DATA / "claims.jsonl")
    by_clip = {}
    for c in claims:
        by_clip.setdefault(c["clip_id"], []).append(c)

    verify = VERIFIERS[args.system]
    retr = RETRIEVER if args.system != "closed_book" else None
    print(f"{common.MODEL} / {args.system}: {len(clips)} clips, provider id {resolve(common.MODEL)[2]}, retriever={retr or '-'}")
    preds = await asyncio.gather(*[run_clip(cl, by_clip.get(cl["clip_id"], []), verify) for cl in clips])

    out = RESULTS / common.MODEL / args.system
    write_jsonl(out / "preds.jsonl", preds)
    cost = usage_cost(common.MODEL)
    m = MODELS.get(common.MODEL, {})
    usage = {**USAGE, "latency_s": round(USAGE["latency_s"], 2), "cost_usd": None if cost is None else round(cost, 4),
             "model": common.MODEL, "model_name": m.get("name", common.MODEL), "tier": m.get("tier", "open"),
             "provider_id": resolve(common.MODEL)[2], "system": args.system, "retriever": retr, "clips": len(clips),
             "judge": common.JUDGE_MODEL, "mock": common.MOCK}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    n_ex = sum(len(p["extracted"]) for p in preds)
    print(f"wrote {out}/preds.jsonl: {n_ex} extracted claims, {sum(len(p['oracle']) for p in preds)} gold claims; "
          f"{usage['calls']} calls, cost {usage['cost_usd']}")


if __name__ == "__main__":
    asyncio.run(main())
