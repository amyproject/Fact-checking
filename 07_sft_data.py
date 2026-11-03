"""Build the supervised fine-tuning set for the labeler from AVeriTeC claims outside the test set.

    python 07_sft_data.py

Each example is the exact prompt harness C (rag_qa_verify) sends at test time, filled from
the fact-checker's own research instead of a search engine:
  questions  = the fact-checker's verification questions
  evidence   = the fact-checker's answers (each with its source URL), shuffled with 2-4
               distractor answers from other claims in the same country (hard negatives)
  target     = {"label", "rationale": fact-checker's justification, "citations": up to 2 gold
               passages, each quoting its sentence closest to the claim + justification;
               fragments under 5 words or sharing <2 content words with the claim are not cited}
Augmentation: 10% of decisive examples lose every gold passage and become "unverified",
so the model learns to abstain when retrieval misses (the most common test-time failure).

Leakage control: drops every test claim, every claim from a test video or with a test
fact-check article, and any claim whose wording overlaps a test claim (Jaccard >= 0.5).
63 of the 65 test claims come from AVeriTeC's train split, so this matters.

Writes data/sft/{train,val}.jsonl (Fireworks chat format) and data/sft/stats.json.
"""
import importlib
import json
import random
import re
from collections import Counter

import pandas as pd

from common import DATA, GOLD_MAP, read_jsonl, write_jsonl
from retrieval import tokens

run = importlib.import_module("02_run")
tbuild = importlib.import_module("01_build_testset")
traps = importlib.import_module("06_traps")

SEED, VAL_SHARE, DROPOUT, JACCARD = 0, 0.1, 0.10, 0.5
UNANSWERABLE = re.compile(r"^\s*no answer could be found", re.I)


def txt(x):
    """AVeriTeC leaves some text fields as NaN."""
    return x if isinstance(x, str) else ""


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]


def best_sentence(text, target):
    t = set(tokens(target))
    sents = sentences(text) or [text.strip()]
    return max(sents, key=lambda s: len(t & set(tokens(s))))


def short(text, words=60):
    w = " ".join(text.split()).split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def jaccard(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / max(len(a | b), 1)


def main():
    rng = random.Random(SEED)
    rows = []
    for split in ("train", "dev"):
        df = pd.read_parquet(DATA / "raw" / f"averitec_{split}.parquet")
        for i, r in enumerate(df.itertuples(index=False)):
            rows.append((f"{split}-{i}", r))
    test = read_jsonl(DATA / "claims.jsonl")
    test_ids = {c["id"] for c in test}
    test_videos = {c["video"] for c in test}
    test_articles = {c["fact_checking_article"] for c in test if c["fact_checking_article"]}

    dropped = Counter()
    pool = []
    for rid, r in rows:
        if rid in test_ids:
            dropped["is a test claim"] += 1
        elif tbuild.video_key(txt(r.original_claim_url)) in test_videos:
            dropped["same video as a test claim"] += 1
        elif txt(r.fact_checking_article) in test_articles:
            dropped["same fact-check article as a test claim"] += 1
        elif any(jaccard(r.claim, t["claim"]) >= JACCARD for t in test):
            dropped[f"wording overlaps a test claim (Jaccard >= {JACCARD})"] += 1
        elif r.label not in GOLD_MAP:
            dropped["no label"] += 1
        else:
            qa = []
            for q in (r.questions if r.questions is not None else []):
                for a in traps.answers(q):
                    text = str(a.get("answer", "")).strip()
                    if text and not UNANSWERABLE.match(text) and a.get("answer_type") != "Unanswerable":
                        qa.append({"q": q.get("question", ""), "text": text[:900],
                                   "url": a.get("source_url") or "https://unknown-source"})
            pool.append({"id": rid, "r": r, "qa": qa, "loc": txt(r.location_ISO_code).upper()})

    by_loc = {}
    for p in pool:
        for a in p["qa"]:
            by_loc.setdefault(p["loc"], []).append((p["id"], a))
    every = [x for v in by_loc.values() for x in v]

    examples, aug = [], 0
    for p in pool:
        r = p["r"]
        label = GOLD_MAP[r.label]
        gold = p["qa"][:6]
        if not gold and label != "unverified":
            dropped["decisive label but no usable evidence"] += 1
            continue
        drop = label != "unverified" and rng.random() < DROPOUT
        candidates = [a for i, a in by_loc.get(p["loc"], []) if i != p["id"]] or [a for i, a in every if i != p["id"]]
        distract = rng.sample(candidates, min(len(candidates), rng.randint(2, 4)))
        passages = ([] if drop else [dict(a, gold=True) for a in gold]) + [dict(a, gold=False) for a in distract]
        rng.shuffle(passages)
        evidence = [{"url": a["url"], "text": a["text"]} for a in passages]
        just = " ".join(txt(r.justification).split())
        if drop:
            aug += 1
            target = {"label": "unverified",
                      "rationale": "None of the evidence addresses the claim directly, so it does not settle it either way.",
                      "citations": []}
        else:
            cites = []
            if label != "unverified":
                # cite the (at most 2) gold passages that best match the claim; skip fragments
                # like "NYC with 138, 806" that can't back a label on their own
                ctoks, scored = set(tokens(r.claim)), []
                for i, a in enumerate(passages, 1):
                    if a["gold"]:
                        q = best_sentence(a["text"], r.claim + " " + just)
                        ov = len(ctoks & set(tokens(q)))
                        if len(q.split()) >= 5 and ov >= 2:
                            scored.append((ov, i, q))
                cites = [{"id": i, "quote": q} for _, i, q in sorted(scored, reverse=True)[:2]]
                if not cites:
                    dropped["decisive label but no citable gold passage"] += 1
                    continue
            target = {"label": label, "rationale": short(just) or "The evidence settles the claim.", "citations": cites}
        questions = [q.get("question", "") for q in (r.questions if r.questions is not None else [])][:4]
        qblock = "QUESTIONS TO ANSWER FIRST:\n" + "\n".join(f"- {q}" for q in questions) + "\n" if questions else ""
        prompt = run.VERIFY_EVIDENCE.format(
            date=run.parse_claim_date(r.claim_date) or "unknown", transcript=r.claim, claim=r.claim,
            questions=qblock, evidence=run.numbered(evidence), rules=run.LABEL_RULES)
        examples.append({"id": p["id"], "label": target["label"], "augmented": drop,
                         "messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]})

    # AVeriTeC is mostly Refuted; cap the majority class at 1.5x the next largest
    by_label = Counter(e["label"] for e in examples)
    cap = int(1.5 * sorted(by_label.values())[-2])
    rng.shuffle(examples)
    kept, seen = [], Counter()
    for e in examples:
        if seen[e["label"]] < cap:
            kept.append(e)
            seen[e["label"]] += 1
    n_val = int(len(kept) * VAL_SHARE)
    val, train = kept[:n_val], kept[n_val:]
    out = DATA / "sft"
    write_jsonl(out / "train.jsonl", [{"messages": e["messages"]} for e in train])
    write_jsonl(out / "val.jsonl", [{"messages": e["messages"]} for e in val])
    write_jsonl(out / "index.jsonl", [{"id": e["id"], "split": "val" if i < n_val else "train", "label": e["label"],
                                       "augmented": e["augmented"]} for i, e in enumerate(kept)])
    stats = {
        "averitec_rows": len(rows), "dropped": dict(dropped), "examples_before_cap": len(examples),
        "majority_cap": cap, "train": len(train), "val": len(val),
        "labels_train": dict(Counter(e["label"] for e in train)),
        "augmented_unverified": sum(e["augmented"] for e in kept),
        "prompt_chars_median": sorted(len(e["messages"][0]["content"]) for e in kept)[len(kept) // 2],
        "seed": SEED,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
