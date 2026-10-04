"""Build the safety fine-tuning set for defense D2 from the TRAIN split only.

- Harmful seeds (all versions) -> a refusal in the matching language,
  drawn from --refusals, a JSON file written by your annotators:
      {"en": [...], "bn": [...], "banglish": [...]}
- Benign controls (all versions) -> a helpful answer: a reference model's
  response to the same seed and version that was labelled A (answered).
  This teaches the model not to over-refuse Banglish.
- Optionally mix in general instruction data (--general, JSONL with a
  "messages" field) so general ability does not regress.

Output: JSONL with {"messages": [...]} rows, ready for train_lora.py.

Usage:
    python -m banglishjail.defenses.sft_data --seeds data/raw/seeds.csv \
        --refusals data/raw/refusals.json --benign-answers data/raw/final_train.jsonl \
        --general data/raw/general_instructions.jsonl --general-ratio 1.0 \
        --out data/raw/sft_train.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

from banglishjail.data import expand, load_seeds
from banglishjail.io import JsonlWriter, read_jsonl

REFUSAL_BUCKET = {"en": "en", "en_bnscript": "en", "bn": "bn", "banglish_std": "banglish",
                  "banglish_noisy": "banglish", "code_mixed": "banglish"}


def build(seeds, refusals, benign_answers, general=(), general_ratio=0.0, seed=0):
    rng = random.Random(seed)
    answers = {}
    for r in benign_answers:
        label = r.get("final_label", r.get("judge_label"))
        if r.get("is_benign_control") and label == "A" and r.get("response"):
            answers.setdefault((r["seed_id"], r["version"]), r["response"])

    rows, missing = [], 0
    for rec in expand(seeds, split="train"):
        if rec["is_benign_control"]:
            reply = answers.get((rec["seed_id"], rec["version"]))
            if reply is None:
                missing += 1
                continue
        else:
            reply = rng.choice(refusals[REFUSAL_BUCKET[rec["version"]]])
        rows.append({"messages": [{"role": "user", "content": rec["prompt"]},
                                  {"role": "assistant", "content": reply}],
                     "source": "benign" if rec["is_benign_control"] else "harmful",
                     "seed_id": rec["seed_id"], "version": rec["version"]})
    general = list(general)
    n_general = min(len(general), int(len(rows) * general_ratio))
    for g in rng.sample(general, n_general):
        rows.append({"messages": g["messages"], "source": "general"})
    rng.shuffle(rows)
    return rows, missing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", required=True)
    p.add_argument("--refusals", required=True)
    p.add_argument("--benign-answers", required=True, help="labelled responses on the train split")
    p.add_argument("--general")
    p.add_argument("--general-ratio", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    with open(args.refusals, encoding="utf-8") as f:
        refusals = json.load(f)
    rows, missing = build(load_seeds(args.seeds), refusals, read_jsonl(args.benign_answers),
                          read_jsonl(args.general) if args.general else (), args.general_ratio)
    Path(args.out).unlink(missing_ok=True)
    writer = JsonlWriter(args.out)
    for row in rows:
        writer.write(row)
    print(f"wrote {len(rows)} examples; {missing} benign prompts skipped (no answered reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
