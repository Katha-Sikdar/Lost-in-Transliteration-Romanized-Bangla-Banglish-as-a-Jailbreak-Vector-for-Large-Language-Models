"""Human labelling sheets, judge-vs-human agreement, and final labels (Phase 5).

1. sample: draw a stratified sample of judged responses into a CSV for human
   annotators. The judge's label is NOT included, to avoid anchoring.
   All NEEDS_HUMAN items are always included.
2. agreement: Cohen's kappa between two annotators, and between the judge
   and the (adjudicated) human labels, overall and per version.
3. merge: produce final labels: human label where available, else judge label.

Usage:
    python -m banglishjail.labels sample --judged data/raw/judged.jsonl \
        --out data/raw/label_sheet.csv --per-cell 20
    python -m banglishjail.labels agreement --judged data/raw/judged.jsonl \
        --human data/raw/label_sheet_adjudicated.csv [--human-b data/raw/label_sheet_B.csv]
    python -m banglishjail.labels merge --judged data/raw/judged.jsonl \
        --human data/raw/label_sheet_adjudicated.csv --out data/raw/final.jsonl
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from banglishjail.io import JsonlWriter, read_jsonl

SHEET_COLUMNS = ["key", "seed_id", "is_benign_control", "version", "model_name", "en_reference",
                 "sent_prompt", "response", "translation", "human_label", "human_understood", "annotator", "notes"]


def load_judged(path):
    """One record per key; a real judge label beats an earlier NEEDS_HUMAN."""
    best = {}
    for rec in read_jsonl(path):
        prev = best.get(rec["key"])
        if prev is None or rec["judge_label"] != "NEEDS_HUMAN" or prev["judge_label"] == "NEEDS_HUMAN":
            best[rec["key"]] = rec
    return list(best.values())


def cohen_kappa(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def sample_sheet(judged, per_cell, seed=7):
    rng = random.Random(seed)
    cells = defaultdict(list)
    must = []
    for rec in judged:
        if rec["judge_label"] == "NEEDS_HUMAN":
            must.append(rec)
        else:
            cells[(rec["model_name"], rec["version"], rec["is_benign_control"])].append(rec)
    chosen = list(must)
    for key in sorted(cells, key=str):
        items = sorted(cells[key], key=lambda r: r["key"])
        rng.shuffle(items)
        chosen.extend(items[:per_cell])
    rng.shuffle(chosen)
    rows = [{
        "key": r["key"], "seed_id": r["seed_id"], "is_benign_control": r["is_benign_control"],
        "version": r["version"], "model_name": r["model_name"], "en_reference": r["en_reference"],
        "sent_prompt": r["sent_prompt"], "response": r.get("response", ""),
        "translation": r.get("judge_translated") or "", "human_label": "", "human_understood": "",
        "annotator": "", "notes": "",
    } for r in chosen]
    return pd.DataFrame(rows, columns=SHEET_COLUMNS)


def _human_labels(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["human_label"].str.strip() != ""]
    return dict(zip(df["key"], df["human_label"].str.strip().str.upper()))


def agreement(judged, human, human_b=None):
    by_key = {r["key"]: r for r in judged}
    lines = []
    if human_b is not None:
        common = sorted(set(human) & set(human_b))
        if common:
            k = cohen_kappa([human[x] for x in common], [human_b[x] for x in common])
            lines.append(f"annotator A vs B: kappa={k:.3f} (n={len(common)})")
    pairs = [(by_key[k]["version"], by_key[k]["judge_label"], v) for k, v in human.items()
             if k in by_key and by_key[k]["judge_label"] != "NEEDS_HUMAN"]
    if pairs:
        k = cohen_kappa([p[1] for p in pairs], [p[2] for p in pairs])
        acc = sum(p[1] == p[2] for p in pairs) / len(pairs)
        lines.append(f"judge vs human (all): kappa={k:.3f} accuracy={acc:.3f} (n={len(pairs)})")
        for version in sorted({p[0] for p in pairs}):
            sub = [p for p in pairs if p[0] == version]
            if len(sub) >= 2:
                k = cohen_kappa([p[1] for p in sub], [p[2] for p in sub])
                lines.append(f"  {version:15s} kappa={k:.3f} (n={len(sub)})")
    return lines


def merge(judged, human):
    out = []
    for rec in judged:
        if rec["key"] in human:
            out.append({**rec, "final_label": human[rec["key"]], "label_source": "human"})
        else:
            out.append({**rec, "final_label": rec["judge_label"], "label_source": "judge"})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--judged", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-cell", type=int, default=20, help="per (model, version, benign) cell")
    a = sub.add_parser("agreement")
    a.add_argument("--judged", required=True)
    a.add_argument("--human", required=True)
    a.add_argument("--human-b", help="second annotator's sheet, for inter-annotator kappa")
    m = sub.add_parser("merge")
    m.add_argument("--judged", required=True)
    m.add_argument("--human", required=True)
    m.add_argument("--out", required=True)
    args = p.parse_args(argv)

    judged = load_judged(args.judged)
    if args.cmd == "sample":
        df = sample_sheet(judged, args.per_cell)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} rows to {args.out}")
    elif args.cmd == "agreement":
        human_b = _human_labels(args.human_b) if args.human_b else None
        for line in agreement(judged, _human_labels(args.human), human_b):
            print(line)
    else:
        merged = merge(judged, _human_labels(args.human))
        unresolved = sum(r["final_label"] == "NEEDS_HUMAN" for r in merged)
        Path(args.out).unlink(missing_ok=True)
        writer = JsonlWriter(args.out)
        for rec in merged:
            writer.write(rec)
        print(f"wrote {len(merged)} records; {unresolved} still NEEDS_HUMAN")
        return 1 if unresolved else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
