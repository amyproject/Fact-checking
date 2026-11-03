"""Fine-tuning v2: SFT data in the exact v2 task format, DPO pairs from dev runs, and the launch commands.

    python 17_sft_v2.py data          # build data/sft_v2/{train,val}.jsonl (no model needed)
    python 17_sft_v2.py dpo           # mine preference pairs from results_v2/dev/* (after dev runs)
    python 17_sft_v2.py launch [--go] # print/run the Fireworks commands: SFT, then DPO on the SFT model

Stage 1, SFT (what v1 did, made harder): each AVeriTeC claim outside every v2 test/dev claim
becomes the v2 labeling prompt, filled from the fact-checker's research: their questions,
their answers as numbered passages (+ same-country distractors), and the target answer:
4-way label, a confidence target (0.9 with gold evidence, 0.8 for evidence-dropout
abstentions, 0.7 for conflicting), every question answered with a cited verbatim sentence,
and the fact-checker's justification as the rationale.
Stage 2, DPO: for each dev claim where one run is verified and another fails a gate (flip,
fabricated quote) or the label, the verified answer is "chosen" and the failed one
"rejected", on the same prompt. Dev claims never appear in the test split.
Stage 3, RL (planned, not built): GRPO with reward = exact label x all gates pass x
citation verified, starting from the DPO checkpoint.
"""
import argparse
import importlib
import json
import random
import shlex
import shutil
import subprocess

from common import DATA, GOLD_MAP, MODELS, REGISTRY, ROOT, read_jsonl, write_jsonl
from retrieval import tokens

sft1 = importlib.import_module("07_sft_data")
run2 = importlib.import_module("13_run_v2")
traps = importlib.import_module("06_traps")
tb = importlib.import_module("01_build_testset")

FOUR = {"Supported": "supported", "Refuted": "refuted", "Not Enough Evidence": "not_enough_evidence",
        "Conflicting Evidence/Cherrypicking": "conflicting"}
OUT = DATA / "sft_v2"
SEED, DROPOUT = 0, 0.10
CONF = {"supported": 0.9, "refuted": 0.9, "conflicting": 0.7, "not_enough_evidence": 0.8}


def build_data():
    import pandas as pd
    rng = random.Random(SEED)
    held = read_jsonl(DATA / "v2" / "claims.jsonl")
    held_ids = {c["id"] for c in held}
    held_videos = {c["video"] for c in held}
    held_articles = {c.get("fact_checking_article") for c in held if c.get("fact_checking_article")}
    held_tok = [set(tokens(c["claim"])) for c in held]
    rows, dropped = [], {}
    for split in ("train", "dev"):
        df = pd.read_parquet(DATA / "raw" / f"averitec_{split}.parquet")
        for i, r in enumerate(df.itertuples(index=False)):
            rows.append((f"{split}-{i}", r))
    pool = []
    for rid, r in rows:
        t = set(tokens(r.claim))
        why = ("held-out claim" if rid in held_ids else
               "same video as a held-out claim" if tb.video_key(sft1.txt(r.original_claim_url)) in held_videos else
               "same fact-check article" if sft1.txt(r.fact_checking_article) in held_articles else
               "wording overlaps a held-out claim" if any(len(t & h) / max(len(t | h), 1) >= 0.5 for h in held_tok) else
               "no label" if r.label not in FOUR else None)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        qa = []
        for q in (r.questions if r.questions is not None else []):
            for a in traps.answers(q)[:1]:
                text = str(a.get("answer", "")).strip()
                ok = text and not sft1.UNANSWERABLE.match(text) and a.get("answer_type") != "Unanswerable"
                qa.append({"q": q.get("question", ""), "text": text[:900] if ok else "",
                           "url": a.get("source_url") or "https://unknown-source"})
        pool.append({"id": rid, "r": r, "qa": qa, "loc": sft1.txt(r.location_ISO_code).upper()})
    by_loc = {}
    for p in pool:
        for a in p["qa"]:
            if a["text"]:
                by_loc.setdefault(p["loc"], []).append((p["id"], a))
    examples = []
    for p in pool:
        r, label = p["r"], FOUR[p["r"].label]
        drop = label in ("supported", "refuted") and rng.random() < DROPOUT
        gold = [a for a in p["qa"] if a["text"]][:6]
        if not gold and label != "not_enough_evidence":
            dropped["decisive label but no evidence"] = dropped.get("decisive label but no evidence", 0) + 1
            continue
        cands = [a for i, a in by_loc.get(p["loc"], []) if i != p["id"]]
        distract = rng.sample(cands, min(len(cands), rng.randint(2, 4)))
        passages = ([] if drop else [dict(a, gold=True) for a in gold]) + [dict(a, gold=False) for a in distract]
        rng.shuffle(passages)
        evidence = [{"url": a["url"], "text": a["text"]} for a in passages]
        just = " ".join(sft1.txt(r.justification).split())
        pos = {id(a): i for i, a in enumerate(passages, 1)}
        questions = []
        for a in p["qa"][:5]:
            hit = next((x for x in passages if x["gold"] and x["text"] == a["text"]), None)
            if hit and not drop:
                quote = sft1.best_sentence(hit["text"], a["q"] + " " + r.claim)
                questions.append({"q": a["q"], "answer": sft1.short(hit["text"], 40), "citations": [{"id": pos[id(hit)], "quote": quote}]})
            else:
                questions.append({"q": a["q"], "answer": "no answer found", "citations": []})
        target_label = "not_enough_evidence" if drop else label
        cites = [c for q in questions for c in q["citations"]][:3] if target_label != "not_enough_evidence" else []
        if target_label in ("supported", "refuted", "conflicting") and not cites:
            dropped["decisive label but no citable passage"] = dropped.get("decisive label but no citable passage", 0) + 1
            continue
        target = {"label": target_label, "confidence": CONF[target_label], "claim_understood": r.claim,
                  "subclaims": [], "questions": questions,
                  "rationale": "None of the evidence addresses the claim, so it does not settle it." if drop else (sft1.short(just, 70) or "The evidence settles the claim."),
                  "citations": cites}
        plan = "VERIFICATION QUESTIONS:\n" + "\n".join(f"- {q['q']}" for q in questions) + "\n" if questions else ""
        prompt = run2.LABEL.format(date=run2.parse_claim_date(r.claim_date) or "unknown", transcript=r.claim, claim=r.claim,
                                   plan=plan, numbers=run2.number_check(r.claim, evidence), evidence=run2.run1.numbered(evidence),
                                   rules=run2.RULES)
        examples.append({"id": p["id"], "label": target_label, "augmented": drop,
                         "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(target)}]})
    rng.shuffle(examples)
    counts = {}
    for e in examples:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    cap = int(1.5 * sorted(counts.values())[-2])
    kept, seen = [], {}
    for e in examples:
        if seen.get(e["label"], 0) < cap:
            kept.append(e)
            seen[e["label"]] = seen.get(e["label"], 0) + 1
    n_val = len(kept) // 10
    write_jsonl(OUT / "train.jsonl", [{"messages": e["messages"]} for e in kept[n_val:]])
    write_jsonl(OUT / "val.jsonl", [{"messages": e["messages"]} for e in kept[:n_val]])
    stats = {"averitec_rows": len(rows), "dropped": dropped, "train": len(kept) - n_val, "val": n_val,
             "labels": {k: sum(1 for e in kept[n_val:] if e["label"] == k) for k in FOUR.values()},
             "augmented": sum(e["augmented"] for e in kept), "majority_cap": cap,
             "questions_per_example": round(sum(len(json.loads(e["messages"][1]["content"])["questions"]) for e in kept) / max(len(kept), 1), 2)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def build_dpo():
    """Pairs on the same dev claim and prompt: a verified answer beats a failed one."""
    import common
    res = ROOT / ("results_v2_mock" if common.MOCK else "results_v2") / "dev"
    by_claim = {}
    for rows_f in res.glob("*/*/rows.jsonl"):
        for r in read_jsonl(rows_f):
            by_claim.setdefault(r["id"], []).append((rows_f.parent, r))
    pairs = []
    for cid, runs in by_claim.items():
        good = [x for x in runs if x[1]["verified"]]
        bad = [x for x in runs if not x[1]["correct"] or not all(x[1]["gates"].values())]
        if good and bad:
            g, b = max(good, key=lambda x: x[1]["score"]), min(bad, key=lambda x: x[1]["score"])
            pairs.append({"claim_id": cid, "chosen_run": str(g[0].relative_to(res)), "rejected_run": str(b[0].relative_to(res)),
                          "chosen": {k: g[1][k] for k in ("pred", "confidence", "rationale", "questions")},
                          "rejected": {k: b[1][k] for k in ("pred", "confidence", "rationale", "questions")},
                          "why_rejected": [k for k, ok in b[1]["gates"].items() if not ok] or ["wrong label"]})
    write_jsonl(DATA / "sft_v2" / "dpo_pairs.jsonl", pairs)
    print(f"{len(pairs)} DPO pairs from {len(by_claim)} dev claims")


def launch(go):
    base = MODELS[REGISTRY["base_open"]]["id"]
    name = "factcheck-labeler-v2"
    cmds = [["firectl", "create", "dataset", f"{name}-train", str(OUT / "train.jsonl")],
            ["firectl", "create", "dataset", f"{name}-val", str(OUT / "val.jsonl")],
            ["firectl", "create", "sftj", "--base-model", base, "--dataset", f"{name}-train", "--evaluation-dataset",
             f"{name}-val", "--output-model", f"{name}-sft", "--epochs", "2", "--learning-rate", "1e-4", "--lora-rank", "32",
             "--max-context-length", "12288"],
            ["firectl", "create", "dataset", f"{name}-dpo", str(OUT / "dpo_pairs.fireworks.jsonl")],
            ["firectl", "create", "dpoj", "--base-model", f"accounts/<your-account>/models/{name}-sft",
             "--dataset", f"{name}-dpo", "--output-model", name, "--epochs", "1", "--learning-rate", "5e-6"]]
    for c in cmds:
        print(" ".join(shlex.quote(x) for x in c))
    print(f"\nThen set SFT2_MODEL_ID=accounts/<your-account>/models/{name} and rerun 15_iterate_v2.py."
          "\nNot yet run end to end: check flags with `firectl create sftj --help` / `firectl create dpoj --help`.")
    if go:
        if not shutil.which("firectl"):
            raise SystemExit("firectl not found")
        for c in cmds:
            subprocess.run(c, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("data", "dpo", "launch"))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    {"data": build_data, "dpo": build_dpo, "launch": lambda: launch(a.go)}[a.step]()
