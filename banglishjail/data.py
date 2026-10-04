"""Load, validate, split and expand the seed-prompt spreadsheet.

The annotators' spreadsheet is "wide": one row per seed, one column per
version. The pipeline works on a "long" table: one row per (seed, version).

Usage:
    python -m banglishjail.data validate data/raw/seeds.csv
    python -m banglishjail.data split data/raw/seeds.csv --train-frac 0.3
    python -m banglishjail.data prefill-bnscript data/raw/seeds.csv

prefill-bnscript fills empty `en_bnscript` cells by transliterating the English
version into Bengali script with IndicXlit (pip install
ai4bharat-transliteration). Annotators must then check and correct every cell.
"""

import argparse
import random
import sys
from collections import defaultdict

import pandas as pd

from banglishjail import VERSIONS

REQUIRED_COLUMNS = ["id", "category", "is_benign_control", "split", *VERSIONS]
VALID_SPLITS = {"test", "train", ""}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def load_seeds(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df["is_benign_control"] = df["is_benign_control"].map(_to_bool)
    return df


def validate(df):
    """Return a list of human-readable problems (empty list = valid)."""
    problems = []
    dupes = df["id"][df["id"].duplicated()].unique()
    if len(dupes):
        problems.append(f"duplicate ids: {list(dupes)}")
    for _, row in df.iterrows():
        if not row["id"].strip():
            problems.append("row with empty id")
        if not row["category"].strip():
            problems.append(f"{row['id']}: empty category")
        if row["split"] not in VALID_SPLITS:
            problems.append(f"{row['id']}: split must be test/train, got {row['split']!r}")
        for v in VERSIONS:
            if not str(row[v]).strip():
                problems.append(f"{row['id']}: empty version {v}")
    return problems


def assign_splits(df, train_frac=0.3, seed=13):
    """Assign test/train per seed, stratified by (category, is_benign_control).

    Seeds that already have a split keep it, so re-running is safe.
    """
    df = df.copy()
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        if not row["split"]:
            groups[(row["category"], row["is_benign_control"])].append(idx)
    for key in sorted(groups, key=str):
        idxs = sorted(groups[key], key=lambda i: df.at[i, "id"])
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_frac + 0.5)
        for i, idx in enumerate(idxs):
            df.at[idx, "split"] = "train" if i < n_train else "test"
    return df


def expand(df, versions=VERSIONS, split=None):
    """Wide -> long: one record per (seed, version)."""
    records = []
    for _, row in df.iterrows():
        if split and row["split"] != split:
            continue
        for v in versions:
            records.append(
                {
                    "seed_id": row["id"],
                    "category": row["category"],
                    "is_benign_control": bool(row["is_benign_control"]),
                    "split": row["split"],
                    "version": v,
                    "prompt": row[v],
                    "en_reference": row["en"],
                }
            )
    return records


def prefill_bnscript(df, transliterate):
    """Fill empty en_bnscript cells with transliterate(en). Returns the count filled."""
    filled = 0
    for idx, row in df.iterrows():
        if not str(row["en_bnscript"]).strip() and str(row["en"]).strip():
            df.at[idx, "en_bnscript"] = transliterate(row["en"])
            filled += 1
    return filled


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("seeds")
    s = sub.add_parser("split", help="fill empty split cells in place")
    s.add_argument("seeds")
    s.add_argument("--train-frac", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=13)
    b = sub.add_parser("prefill-bnscript", help="fill empty en_bnscript cells in place")
    b.add_argument("seeds")
    args = p.parse_args(argv)

    df = load_seeds(args.seeds)
    if args.cmd == "validate":
        problems = validate(df)
        for problem in problems:
            print("ERROR", problem)
        print(f"{len(df)} seeds, {len(problems)} problems")
        return 1 if problems else 0
    if args.cmd == "prefill-bnscript":
        from banglishjail.normalize import XlitNormalizer

        filled = prefill_bnscript(df, XlitNormalizer())
        df.to_csv(args.seeds, index=False)
        print(f"filled {filled} en_bnscript cells; check every one by hand")
        return 0
    df = assign_splits(df, args.train_frac, args.seed)
    df.to_csv(args.seeds, index=False)
    print(df.groupby(["split", "is_benign_control"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
