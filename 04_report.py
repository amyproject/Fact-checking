"""Leaderboard, McNemar, cluster bootstrap and the two charts.

    python 04_report.py --baseline closed_book --systems rag rag_qa

Writes results/report.md, results/mcnemar.png, results/bootstrap.png.
The bootstrap resamples whole videos: claims from one video share a speaker and a topic,
so resampling single claims would make the interval look tighter than it is.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from common import LABELS, RESULTS, read_jsonl

REPS, SEED = 5000, 0
NAMES = {"closed_book": "Baseline: closed-book", "rag": "Harness A: search + cite",
         "rag_qa": "Harness B: questions + search + cite", "sft": "Fine-tuned labeler + search"}


def macro_f1(gold, pred):
    f1s = []
    for lab in LABELS:
        tp = np.sum((pred == lab) & (gold == lab))
        fp = np.sum((pred == lab) & (gold != lab))
        fn = np.sum((pred != lab) & (gold == lab))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


METRICS = {
    "label_acc": lambda d, i: d["correct"][i].mean(),
    "sb_acc": lambda d, i: d["sb_correct"][i].mean(),
    "e2e_acc": lambda d, i: d["e2e_correct"][i].mean(),
    "macro_f1": lambda d, i: macro_f1(d["gold"][i], d["pred"][i]),
    "flip_rate": lambda d, i: d["flip"][i].mean(),
    "cite_ok": lambda d, i: d["n_citations_ok"][i].sum() / max(d["n_citations"][i].sum(), 1),
}


def load(system, ids=None):
    rows = {r["id"]: r for r in read_jsonl(RESULTS / system / "rows.jsonl")}
    ids = ids or sorted(rows)
    cols = ["correct", "sb_correct", "e2e_correct", "flip", "matched", "n_citations",
            "n_citations_ok", "n_citations_resolve"]
    d = {c: np.array([rows[i][c] for i in ids]) for c in cols}
    for c in ("gold", "pred", "video"):
        d[c] = np.array([rows[i][c] for i in ids])
    usage = RESULTS / system / "usage.json"
    d["usage"] = json.loads(usage.read_text()) if usage.exists() else {}
    return ids, d


def mcnemar(base, sys_, key):
    only_sys = int(np.sum((sys_[key] == 1) & (base[key] == 0)))
    only_base = int(np.sum((sys_[key] == 0) & (base[key] == 1)))
    n = only_sys + only_base
    p = binomtest(only_sys, n, 0.5).pvalue if n else 1.0
    return only_sys, only_base, p


def cluster_bootstrap(base, sys_, metric, rng, mask=None):
    """95% CI for (system - baseline) on `metric`, resampling videos with replacement."""
    idx_all = np.arange(len(base["video"])) if mask is None else np.flatnonzero(mask)
    videos = np.unique(base["video"][idx_all])
    by_video = {v: idx_all[base["video"][idx_all] == v] for v in videos}
    f = METRICS[metric]
    point = f(sys_, idx_all) - f(base, idx_all)
    gaps = np.empty(REPS)
    for r in range(REPS):
        pick = rng.choice(videos, size=len(videos), replace=True)
        idx = np.concatenate([by_video[v] for v in pick])
        gaps[r] = f(sys_, idx) - f(base, idx)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="closed_book")
    ap.add_argument("--systems", nargs="+", default=["rag", "rag_qa"])
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    ids, base = load(args.baseline)
    systems = {s: load(s, ids)[1] for s in args.systems}
    allsys = {args.baseline: base, **systems}
    n, n_videos = len(ids), len(np.unique(base["video"]))

    lines = [f"# Results ({n} gold claims, {n_videos} videos)", "",
             f"Model: `{base['usage'].get('model', '?')}`. Baseline: `{args.baseline}`.", "",
             "## Leaderboard", "",
             "| System | Label acc | Source-backed acc | End-to-end acc | Macro-F1 | Flip rate | Citations that check out | Citation URLs that resolve | Cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    every = np.arange(n)
    for s, d in allsys.items():
        nc = d["n_citations"].sum()
        cost = d["usage"].get("cost_usd")
        lines.append(f"| {NAMES.get(s, s)} | {pct(METRICS['label_acc'](d, every))} | {pct(METRICS['sb_acc'](d, every))} | "
                     f"{pct(METRICS['e2e_acc'](d, every))} | {METRICS['macro_f1'](d, every):.3f} | "
                     f"{pct(METRICS['flip_rate'](d, every))} | {d['n_citations_ok'].sum()}/{nc} | "
                     f"{d['n_citations_resolve'].sum()}/{nc} | {'$%.2f' % cost if cost else 'n/a'} |")
    floor = np.mean(base["gold"] == "unverified")
    lines += ["", f"Reference: answering `unverified` to everything scores {pct(floor)} label accuracy and the same "
              "source-backed accuracy, since an abstention needs no citation.", "",
              f"Extraction recall (shared by all systems): {pct(base['matched'].mean())}", ""]

    lines += ["## Is the gain real?", "",
              "McNemar counts only the claims where exactly one system was right. "
              f"Bootstrap: {REPS:,} resamples of whole videos, 95% interval for the gap.", "",
              "| System vs baseline | Metric | Only system right | Only baseline right | McNemar p | Gap | 95% CI | Verdict |",
              "|---|---|---|---|---|---|---|---|"]
    boot = {}
    for s, d in systems.items():
        for key, metric in (("correct", "label_acc"), ("sb_correct", "sb_acc")):
            a, b, p = mcnemar(base, d, key)
            pt, lo, hi = cluster_bootstrap(base, d, metric, rng)
            boot[(s, metric)] = (pt, lo, hi, a, b, p)
            agree = p < 0.05 and (lo > 0 or hi < 0)
            verdict = ("🟢 real gain" if pt > 0 else "real loss") if agree else "⬜️ nss (tests disagree or p ≥ 0.05)"
            lines.append(f"| {NAMES.get(s, s)} | {metric} | {a} | {b} | {p:.2g} | {100 * pt:+.1f} pp | "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {verdict} |")
        pt, lo, hi = cluster_bootstrap(base, d, "macro_f1", rng)
        lines.append(f"| {NAMES.get(s, s)} | macro_f1 (bootstrap only) | | | | {pt:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{'🟢 real gain' if lo > 0 else ('real loss' if hi < 0 else '⬜️ nss')} |")

    # sensitivity: does one long speech carry the result?
    vids, counts = np.unique(base["video"], return_counts=True)
    biggest = vids[np.argmax(counts)]
    mask = base["video"] != biggest
    lines += ["", f"## Sensitivity: without the largest video ({biggest}, {counts.max()} claims)", "",
              "| System vs baseline | Source-backed gap | 95% CI |", "|---|---|---|"]
    for s, d in systems.items():
        pt, lo, hi = cluster_bootstrap(base, d, "sb_acc", rng, mask)
        lines.append(f"| {NAMES.get(s, s)} | {100 * pt:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    # chart 1: McNemar disagreement cells on source-backed accuracy
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [NAMES.get(s, s) for s in systems]
    y = np.arange(len(systems))
    wins = [boot[(s, "sb_acc")][3] for s in systems]
    losses = [boot[(s, "sb_acc")][4] for s in systems]
    ax.barh(y + 0.18, wins, 0.36, label="only the system right", color="#2a9d8f")
    ax.barh(y - 0.18, losses, 0.36, label="only the baseline right", color="#b0b7c3")
    for i, s in enumerate(systems):
        ax.text(max(wins[i], losses[i]) + 0.5, i, f"p = {boot[(s, 'sb_acc')][5]:.2g}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("claims (source-backed correct)")
    ax.set_title("Who wins the disagreements? (McNemar)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "mcnemar.png", dpi=160)

    # chart 2: bootstrap gaps with 95% CI
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rows = [(s, m) for s in systems for m in ("label_acc", "sb_acc")]
    for i, (s, m) in enumerate(rows):
        pt, lo, hi = boot[(s, m)][:3]
        ax.errorbar(100 * pt, i, xerr=[[100 * (pt - lo)], [100 * (hi - pt)]], fmt="o",
                    color="#2a9d8f" if lo > 0 else "#6b7280", capsize=4)
    ax.axvline(0, color="#9ca3af", lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [f"{NAMES.get(s, s)}\n{'label acc' if m == 'label_acc' else 'source-backed acc'}"
                                     for s, m in rows], fontsize=8)
    ax.set_xlabel("gap vs baseline (pp), 95% cluster-bootstrap CI")
    ax.set_title("Would the gap survive a different set of videos?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "bootstrap.png", dpi=160)

    lines += ["", "![McNemar](mcnemar.png)", "", "![Bootstrap](bootstrap.png)", ""]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
