"""Run the full model x system matrix from models.json: run, score, rubric; then the report.

    python run_all.py                 # everything not already scored
    python run_all.py --only glm-5.3  # one model
    python run_all.py --force         # rerun scored pairs too (LLM calls stay cached)

The fine-tuned model runs only once SFT_MODEL_ID is set (or in mock mode).
"""
import argparse
import os
import subprocess
import sys

from common import MOCK, MODELS, REGISTRY, RESULTS


def matrix():
    mx, pairs = REGISTRY["matrix"], []
    for key in MODELS:
        if key in mx:
            systems = mx[key]
        elif key == REGISTRY["base_open"]:
            systems = mx["base_open"]
        else:
            systems = mx["all_models"]
        pairs += [(key, s) for s in systems]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    py = sys.executable
    for model, system in matrix():
        if args.only and model not in args.only:
            continue
        if MODELS[model]["id"].startswith("env:") and not MOCK and not os.environ.get(MODELS[model]["id"][4:]):
            print(f"skip {model}: set {MODELS[model]['id'][4:]} after 08_sft_train.py")
            continue
        run = f"{model}/{system}"
        if (RESULTS / run / "rubric.jsonl").exists() and not args.force:
            print(f"done {run}")
            continue
        steps = [[py, "02_run.py", "--model", model, "--system", system] + (["--limit", str(args.limit)] if args.limit else []),
                 [py, "03_score.py", "--run", run], [py, "09_rubric.py", "--run", run]]
        for s in steps:
            subprocess.run(s, check=True)
    subprocess.run([py, "10_submission.py"], check=True)


if __name__ == "__main__":
    main()
