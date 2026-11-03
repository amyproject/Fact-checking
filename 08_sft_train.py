"""Launch the LoRA fine-tune of the base open model on Fireworks.

    python 08_sft_train.py          # print the commands
    python 08_sft_train.py --go     # run them (needs firectl, logged in)

Uploads data/sft/train.jsonl and val.jsonl, starts a supervised fine-tuning job on the
base open model from models.json, and prints what to put in .env as SFT_MODEL_ID so
`02_run.py --model glm-5.3-ft` serves it. Not yet run end to end: check the flags against
`firectl create sftj --help` before the first job.

Hyperparameters follow the Forge sample (rank 16, lr 1e-4, 1 epoch); 2 epochs is the
first thing to try if validation loss is still falling.
"""
import argparse
import shlex
import shutil
import subprocess

from common import DATA, MODELS, REGISTRY

HP = {"epochs": 1, "learning-rate": 1e-4, "lora-rank": 16, "max-context-length": 8192}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--name", default="factcheck-labeler-v1")
    args = ap.parse_args()
    base = MODELS[REGISTRY["base_open"]]["id"]
    cmds = [
        ["firectl", "create", "dataset", f"{args.name}-train", str(DATA / "sft" / "train.jsonl")],
        ["firectl", "create", "dataset", f"{args.name}-val", str(DATA / "sft" / "val.jsonl")],
        ["firectl", "create", "sftj", "--base-model", base, "--dataset", f"{args.name}-train",
         "--evaluation-dataset", f"{args.name}-val", "--output-model", args.name,
         *[x for k, v in HP.items() for x in (f"--{k}", str(v))]],
    ]
    for c in cmds:
        print(" ".join(shlex.quote(x) for x in c))
    print(f"\nWhen the job finishes: deploy {args.name}, then add to .env:\n"
          f"  SFT_MODEL_ID=accounts/<your-account>/models/{args.name}\n"
          f"and run: python 02_run.py --model glm-5.3-ft --system rag_qa_verify")
    if args.go:
        if not shutil.which("firectl"):
            raise SystemExit("firectl not found: https://docs.fireworks.ai/tools-sdks/firectl/firectl")
        for c in cmds:
            subprocess.run(c, check=True)


if __name__ == "__main__":
    main()
