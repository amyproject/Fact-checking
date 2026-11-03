"""Iterate on the dev split until the improvement is clearly real, then confirm once on the test split.

    python 15_iterate_v2.py

Round 0  dev: base open model off the shelf (closed_book) and with harness `rag`.
Round 1  dev: `rag` + each candidate feature alone. A feature is positive when its rubric-score
         gain over `rag` is > 0 with P(gain > 0) >= 0.8 under a video-level bootstrap.
Round 2  dev: COMBINED = `rag` + every positive feature (and the best single feature, if it beats
         the combination, is kept as the alternative).
Round 3  dev: the fine-tuned model (glm-5.3-ft2) on the best config, if SFT2_MODEL_ID is set.
Round 4  the ideation model (an open model) reads the dev failures and proposes new ideas
         (16_ideate_open.py); they are logged for the next build cycle, not auto-implemented.
Stop     when best-vs-closed_book on dev gains >= min_dev_gain with P >= 0.95, or rounds run out.
Freeze   the winner; run it on the test split once for every evaluated model, plus each
         model off the shelf. Significance on test is computed in the report (McNemar,
         permutation, video bootstrap, Holm over the pre-registered family).

Selection uses the rubric score (continuous, more power on ~80 dev claims); verified accuracy
is logged alongside. Writes results_v2/iteration_log.json.
"""
import json
import os
import subprocess
import sys

import numpy as np

import common
from common import DATA, MODELS, REGISTRY, ROOT, read_jsonl

V2 = REGISTRY["v2"]
RESULTS = ROOT / ("results_v2_mock" if common.MOCK else "results_v2")
PY = sys.executable
FEATS = V2["features"]
STOP = V2["stopping"]
log = {"rounds": [], "runs": {}, "test_looks": 0}


def name_of(flags):
    named = {tuple(sorted(v)): k for k, v in V2["systems"].items()}
    return named.get(tuple(sorted(flags))) or "+".join(f for f in FEATS if f in flags)


def run(model, flags, split, robust=False):
    name = name_of(flags)
    d = RESULTS / split / model / name
    if not (d / "rows.jsonl").exists():
        cmd = [PY, "13_run_v2.py", "--model", model, "--features", ",".join(sorted(flags)), "--split", split]
        subprocess.run(cmd + (["--robust"] if robust else []), check=True)
        subprocess.run([PY, "14_score_v2.py", "--run", f"{split}/{model}/{name}"], check=True)
    rows = {r["id"]: r for r in read_jsonl(d / "rows.jsonl")}
    usage = json.loads((d / "usage.json").read_text())
    n = max(usage["n_verdicts"], 1)
    key = f"{split}/{model}/{name}"
    log["runs"][key] = {"model": model, "flags": sorted(flags), "split": split,
                        "score": float(np.mean([r["score"] for r in rows.values()])),
                        "verified": float(np.mean([r["verified"] for r in rows.values()])),
                        "seconds_per_claim": usage["latency_s"] / n, "calls_per_claim": usage["calls"] / n,
                        "tokens_per_claim": (usage["prompt_tokens"] + usage["completion_tokens"]) / n}
    return key, rows


def gain(a_rows, b_rows, metric="score", reps=4000, seed=0):
    """Mean per-claim gain of a over b, its 95% interval and P(gain > 0), resampling videos."""
    ids = sorted(set(a_rows) & set(b_rows))
    d = np.array([(a_rows[i][metric] - b_rows[i][metric]) / (100 if metric == "score" else 1) for i in ids])
    vids = np.array([a_rows[i]["video"] for i in ids])
    groups = [np.flatnonzero(vids == v) for v in np.unique(vids)]
    rng = np.random.default_rng(seed)
    boots = np.array([np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(reps)],
                     dtype=object)
    gs = np.array([d[idx].mean() for idx in boots])
    return {"gain": float(d.mean()), "lo": float(np.percentile(gs, 2.5)), "hi": float(np.percentile(gs, 97.5)),
            "p_gain": float((gs > 0).mean()), "n": len(ids)}


def record(round_name, entries, decision):
    log["rounds"].append({"round": round_name, "entries": entries, "decision": decision})
    print(f"[{round_name}] {decision}")
    (RESULTS / "iteration_log.json").write_text(json.dumps(log, indent=2))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = V2["base_open"]
    rag = set(V2["systems"]["rag"])
    k0, closed = run(base, set(), "dev")
    k1, ragr = run(base, rag, "dev")
    record("0: baselines", [{"run": k0}, {"run": k1, **gain(ragr, closed)}],
           f"rag vs off the shelf on dev: {gain(ragr, closed)['gain'] * 100:+.1f} rubric points")

    singles, positive = [], []
    for f in V2["candidate_features"]:
        k, rows = run(base, rag | {f}, "dev")
        gn = gain(rows, ragr)
        ok = gn["gain"] > 0 and gn["p_gain"] >= STOP["min_prob_gain"]
        singles.append({"run": k, "feature": f, **gn, "positive": ok,
                        "added_seconds": log["runs"][k]["seconds_per_claim"] - log["runs"][k1]["seconds_per_claim"]})
        if ok:
            positive.append(f)
    record("1: one feature at a time", singles, f"positive on dev: {positive or 'none'}")

    best_flags, best_key, best_rows = rag, k1, ragr
    if positive:
        kc, comb = run(base, rag | set(positive), "dev")
        best_single = max((s for s in singles if s["positive"]), key=lambda s: s["gain"])
        _, single_rows = run(base, rag | {best_single["feature"]}, "dev")
        gc, gs = gain(comb, ragr), gain(single_rows, ragr)
        best_flags, best_key, best_rows = (rag | set(positive), kc, comb) if gc["gain"] >= gs["gain"] else \
            (rag | {best_single["feature"]}, best_single["run"], single_rows)
        record("2: combine every positive feature", [{"run": kc, **gc, "combined": positive}, {"run": best_single["run"], **gs}],
               f"kept {name_of(best_flags)}")
    else:
        record("2: combine every positive feature", [], "nothing to combine; keeping rag")

    final_model = base
    ft = "glm-5.3-ft2"
    if common.MOCK or os.environ.get(MODELS[ft]["id"][4:]):
        kf, ftr = run(ft, best_flags, "dev")
        gf = gain(ftr, best_rows)
        if gf["gain"] > 0 and gf["p_gain"] >= STOP["min_prob_gain"]:
            final_model, best_key, best_rows = ft, kf, ftr
        record("3: fine-tuned model on the best harness", [{"run": kf, **gf}], f"final model: {final_model}")
    else:
        record("3: fine-tuned model on the best harness", [], "skipped: SFT2_MODEL_ID not set (run 17_sft_v2.py first)")

    subprocess.run([PY, "16_ideate_open.py", "--run", best_key], check=False)
    total = gain(best_rows, closed)
    reached = total["gain"] >= STOP["min_dev_gain"] and total["p_gain"] >= 0.95
    record("4: stop rule on dev", [{"run": best_key, **total}],
           f"{'met' if reached else 'NOT met'}: {total['gain'] * 100:+.1f} points vs off the shelf, P = {total['p_gain']:.2f}")

    log["frozen"] = {"model": final_model, "flags": sorted(best_flags), "name": name_of(best_flags), "dev_stop_rule_met": reached}
    log["test_looks"] += 1
    for m in V2["evaluated"]:
        if MODELS[m]["id"].startswith("env:") and not common.MOCK and not os.environ.get(MODELS[m]["id"][4:]):
            continue
        if m != final_model and MODELS[m]["tier"] == "improved" and m != ft:
            continue
        run(m, set(), "test")
        run(m, best_flags, "test", robust=True)
    run(base, rag, "test")
    record("5: frozen system on the test split (one look)", [], f"ran {name_of(best_flags)} for every evaluated model")
    subprocess.run([PY, "18_report_v2.py"], check=True)


if __name__ == "__main__":
    main()
