"""Token counts per version for open-model tokenizers (Phase 7, step 1).

Expected pattern: Banglish tokenizes about like English, Bangla script into
many more tokens. Report mean tokens per prompt and the ratio to English.

Usage:
    python -m banglishjail.mech.tokenization --seeds data/raw/seeds.csv \
        --tokenizers meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --out results/tokenization.csv
"""

import argparse
import sys

import pandas as pd

from banglishjail.data import expand, load_seeds


def token_table(records, tokenizers):
    rows = []
    for name, tok in tokenizers.items():
        for r in records:
            rows.append({"tokenizer": name, "seed_id": r["seed_id"], "version": r["version"],
                         "chars": len(r["prompt"]),
                         "tokens": len(tok.encode(r["prompt"], add_special_tokens=False))})
    df = pd.DataFrame(rows)
    summary = df.groupby(["tokenizer", "version"]).agg(mean_tokens=("tokens", "mean"),
                                                      chars_per_token=("chars", "sum"),
                                                      total_tokens=("tokens", "sum")).reset_index()
    summary["chars_per_token"] = summary["chars_per_token"] / summary["total_tokens"]
    en = summary[summary["version"] == "en"].set_index("tokenizer")["mean_tokens"]
    summary["ratio_to_en"] = summary.apply(lambda r: r["mean_tokens"] / en.get(r["tokenizer"], float("nan")), axis=1)
    return df, summary.drop(columns="total_tokens")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", required=True)
    p.add_argument("--tokenizers", nargs="+", required=True, help="Hugging Face model ids")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in args.tokenizers}
    _, summary = token_table(expand(load_seeds(args.seeds)), tokenizers)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
