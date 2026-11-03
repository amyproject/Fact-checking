"""Slide deck for v2: the problem, the environment, what was explored, and how each method compares.

    python 19_deck_v2.py          # after 15_iterate_v2.py; writes results_v2[_mock]/deck.html

Reuses 18_report_v2.build_payload() and reduces it to the aggregates the slides draw:
per-run means with video-level bootstrap intervals, McNemar + Holm on the pre-registered
family, the dev iteration log, failure modes, rubric dimensions, the data funnel and one
real training record. deck_template.html draws everything as SVG; arrow keys move between
slides and printing gives one slide per page.
"""
import importlib
import json
import math
from collections import Counter

import numpy as np

import common
from common import DATA, MODELS, REGISTRY, ROOT
from rubric import TRAPS

rep = importlib.import_module("18_report_v2")
V2 = REGISTRY["v2"]
TEMPLATE = ROOT / "deck_template.html"
REPS = 4000


def boot_ci(vals, vids, rng):
    """95% interval of the mean, resampling videos."""
    vals, vids = np.asarray(vals, float), np.asarray(vids)
    groups = [np.flatnonzero(vids == v) for v in np.unique(vids)]
    means = [vals[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(REPS)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    n, k = x + y, min(x, y)
    if not n or x == y:
        return x, y, 1.0
    return x, y, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    P = rep.build_payload()
    rows, claims = P["rows"], P["claims"]
    test = [c for c in claims if c["split"] == "test"]
    tid = [c["id"] for c in test]
    vid = {c["id"]: c["video"] for c in claims}
    it = P["iteration"]
    frozen = it.get("frozen", {"model": V2["base_open"], "name": "rag", "flags": V2["systems"]["rag"]})
    rng = np.random.default_rng(0)

    runs = {}
    for r in P["runs"]:
        if r["split"] != "test":
            continue
        rs = rows[r["key"]]
        ids = [i for i in tid if i in rs]
        v = [rs[i]["verified"] for i in ids]
        lo, hi = boot_ci(v, [vid[i] for i in ids], rng)
        m = MODELS.get(r["model"], {"name": r["model"], "tier": "open"})
        n = max(r["usage"].get("n_verdicts", len(ids)), 1)
        runs[r["key"]] = {
            "model": r["model"], "name": m["name"], "tier": m["tier"], "system": r["system"],
            "harness": "off the shelf" if r["system"] == "closed_book" else "frozen harness" if r["system"] == frozen["name"] else r["system"],
            "verified": float(np.mean(v)), "lo": lo, "hi": hi,
            "exact": float(np.mean([rs[i]["correct"] for i in ids])), "score": float(np.mean([rs[i]["score"] for i in ids])),
            "seconds": r["usage"].get("latency_s", 0) / n, "calls": r["usage"].get("calls", 0) / n,
            "failures": Counter(rs[i]["failure"] for i in ids),
            "dims": {d: float(np.mean([x for x in (rs[i]["dims"].get(d) for i in ids) if x is not None] or [0])) for d in P["rubric"]["dims"]}}

    def key(model, system):
        k = f"test/{model}/{system}"
        return k if k in runs else None
    base = V2["base_open"]
    roles = {"base_off": key(base, "closed_book"), "base_rag": key(base, "rag"), "base_frozen": key(base, frozen["name"]),
             "improved": key(frozen["model"], frozen["name"]), "ft_frozen": key("glm-5.3-ft2", frozen["name"])}
    front = [k for k, r in runs.items() if r["tier"] == "frontier" and r["system"] == "closed_book"]
    roles["frontier_best"] = max(front, key=lambda k: runs[k]["verified"]) if front else None
    roles["frontier_harness"] = key(runs[roles["frontier_best"]]["model"], frozen["name"]) if roles["frontier_best"] else None
    opens = [k for k, r in runs.items() if r["tier"] == "open" and r["system"] == "closed_book"]
    roles["top_open"] = max(opens, key=lambda k: runs[k]["verified"]) if opens else None

    fam = []

    def add(a, b, why):
        if a and b and a != b and not any(f["a"] == a and f["b"] == b for f in fam):
            x, y, p = mcnemar([rows[a][i]["verified"] for i in tid], [rows[b][i]["verified"] for i in tid])
            fam.append({"a": a, "b": b, "why": why, "gap": runs[a]["verified"] - runs[b]["verified"], "better": x, "worse": y, "p": p})
    imp = roles["improved"]
    add(imp, roles["base_off"], "before vs after")
    add(imp, roles["base_rag"], "vs starting harness")
    for k in front:
        add(imp, k, "vs frontier, off the shelf")
    for k, r in runs.items():
        if r["tier"] == "frontier" and r["system"] == frozen["name"]:
            add(imp, k, "vs frontier + same harness")
    order = sorted(range(len(fam)), key=lambda i: fam[i]["p"])
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(fam) - rank) * fam[i]["p"]))
        fam[i]["holm"] = run_max

    example = next((c for c in test if c["source"] == "averitec" and len(c["qa"]) >= 2 and "exact_number" in c["traps"]
                    and len(c["clip"]["transcript"]) < 1400), test[0])
    sft_rec = json.loads(next(open(DATA / "sft_v2" / "train.jsonl")))["messages"]
    stats = P["stats"]
    deck = {
        "mock": common.MOCK, "generated": P["meta"]["generated"],
        "models": {k: {"name": m["name"], "tier": m["tier"]} for k, m in MODELS.items()},
        "roles_cfg": {"gt": V2["ground_truth_model"], "gt_effort": V2["ground_truth_reasoning"]["effort"], "judge2": V2["second_judge"],
                      "ideation": V2["ideation_model"], "evaluated": V2["evaluated"], "base": base},
        "runs": runs, "roles": roles, "family": fam, "frozen": frozen,
        "data": {"test": len(test), "dev": len(claims) - len(test), "videos": stats.get("test_videos"),
                 "candidates": stats.get("claimreview_candidates"), "tried": sum((stats.get("tried") or {}).values()),
                 "unmapped": sum(v for k, v in (stats.get("dropped") or {}).items() if k.startswith("rating doesn't map")),
                 "test_labels": Counter(c["label4"] for c in test), "dev_labels": Counter(c["label4"] for c in claims if c["split"] == "dev"),
                 "sources": Counter(c["source"] for c in test), "fact_checkers": stats.get("fact_checkers", {}),
                 "located_by": stats.get("located_by", {}), "years": stats.get("years", {}),
                 "minutes": [round(c["clip"]["video_seconds"] / 60, 1) for c in test]},
        "traps": sorted(({"key": k, "name": t["name"], "why": t["why"], "n": sum(k in c["traps"] for c in test)} for k, t in TRAPS.items()),
                        key=lambda t: -t["n"]),
        "rubric": {"dims": P["rubric"]["dims"], "gates": P["rubric"]["gates"], "cost": P["rubric"]["cost"],
                   "items": Counter(i["dim"] for i in P["rubric"]["items"]), "methods": Counter(i["method"] for i in P["rubric"]["items"]),
                   "n": len(P["rubric"]["items"])},
        "features": V2["features"], "ideas": [{k: i[k] for k in ("name", "type", "status", "feature", "steps", "build_hours")} | {"ref": i["ref"]["label"]}
                                              for i in P["ideas"]],
        "iteration": {"rounds": it.get("rounds", []), "runs": it.get("runs", {}), "test_looks": it.get("test_looks", 0)},
        "sft": P["sft"], "dpo_pairs": P["dpo_pairs"],
        "sft_example": {"prompt": sft_rec[0]["content"], "target": json.loads(sft_rec[1]["content"])},
        "example": {k: example.get(k) for k in ("claim", "speaker", "claim_date", "label4", "gold_raw", "justification", "traps", "qa", "source")}
                   | {"transcript": example["clip"]["transcript"], "url": example["clip"]["url"]},
    }
    blob = json.dumps(deck, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    out = rep.RESULTS / "deck.html"
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/{}", blob))
    print(f"wrote {out} ({len(blob) // 1024} KB){' (MOCK DATA)' if common.MOCK else ''}")


if __name__ == "__main__":
    main()
