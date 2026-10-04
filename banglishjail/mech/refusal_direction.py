"""Refusal-direction analysis (Phase 7, steps 2-3), following Arditi et al. (2024).

1. `fit`: compute the difference-in-means refusal direction from ENGLISH
   harmful vs. harmless instructions (two text files, one instruction per
   line; e.g. AdvBench and Alpaca subsets, as in the original paper), at the
   last token of the chat-formatted prompt, for every layer. Pick the layer
   whose projection best separates held-out harmful vs. harmless prompts
   (largest Cohen's d) among the first 80% of layers.
2. `project`: for every seed in the dataset, project each version's
   activation onto the direction (expected: harmful Banglish projects less
   than harmful English), and compute the per-layer cosine similarity of each
   version to the English version of the same seed.

Run on one GPU with >= 24 GB for 7-9B models in bf16.

Usage:
    python -m banglishjail.mech.refusal_direction fit --model meta-llama/Llama-3.1-8B-Instruct \
        --harmful data/raw/en_harmful.txt --harmless data/raw/en_harmless.txt \
        --out results/mech/llama_direction.pt
    python -m banglishjail.mech.refusal_direction project --model meta-llama/Llama-3.1-8B-Instruct \
        --direction results/mech/llama_direction.pt --seeds data/raw/seeds.csv --split test \
        --outdir results/mech/llama
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from banglishjail import VERSIONS
from banglishjail.data import expand, load_seeds


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def chat_format(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def last_token_states(model, tok, prompts, batch_size=8):
    """Return float32 array [n_prompts, n_layers + 1, hidden] of last-token hidden states."""
    import torch

    out = []
    for i in range(0, len(prompts), batch_size):
        batch = [chat_format(tok, p) for p in prompts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def cohens_d(a, b):
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def select_layer(dirs, h_val, b_val, max_frac=0.8):
    """dirs: [layers, hidden]; *_val: [n, layers, hidden]. Returns (layer, d_scores)."""
    scores = []
    for layer in range(dirs.shape[0]):
        unit = dirs[layer] / (np.linalg.norm(dirs[layer]) + 1e-8)
        scores.append(cohens_d(h_val[:, layer] @ unit, b_val[:, layer] @ unit))
    limit = max(1, int(len(scores) * max_frac))
    # layer 0 is the embedding output; skip it
    best = int(np.argmax(scores[1:limit])) + 1
    return best, scores


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def cmd_fit(args):
    import torch

    model, tok = load_model(args.model)
    rng = random.Random(0)
    harmful, harmless = _read_lines(args.harmful), _read_lines(args.harmless)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    n_h, n_b = int(len(harmful) * 0.8), int(len(harmless) * 0.8)
    h_tr = last_token_states(model, tok, harmful[:n_h])
    b_tr = last_token_states(model, tok, harmless[:n_b])
    h_val = last_token_states(model, tok, harmful[n_h:])
    b_val = last_token_states(model, tok, harmless[n_b:])
    dirs = h_tr.mean(axis=0) - b_tr.mean(axis=0)
    layer, scores = select_layer(dirs, h_val, b_val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "directions": torch.tensor(dirs), "layer": layer, "d_scores": scores},
               args.out)
    print(f"selected layer {layer} (hidden_states index), d={scores[layer]:.2f}; saved {args.out}")


def project_seed(states, versions, unit, layer):
    """states: [n_versions, layers, hidden] for one seed. Returns (projection rows, cosine rows)."""
    proj = [{"version": v, "projection": float(states[i, layer] @ unit)} for i, v in enumerate(versions)]
    cos = []
    if "en" in versions:
        en = states[versions.index("en")]
        for i, v in enumerate(versions):
            if v == "en":
                continue
            a = states[i]
            sims = (a * en).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(en, axis=1) + 1e-8)
            cos.extend({"version": v, "layer": layer_idx, "cosine_to_en": float(s)} for layer_idx, s in enumerate(sims))
    return proj, cos


def cmd_project(args):
    import torch

    saved = torch.load(args.direction)
    layer = args.layer if args.layer is not None else saved["layer"]
    direction = saved["directions"][layer].numpy()
    unit = direction / np.linalg.norm(direction)
    model, tok = load_model(args.model)

    records = expand(load_seeds(args.seeds), split=args.split)
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed_id"], []).append(r)
    proj_rows, cos_rows = [], []
    for seed_id, recs in by_seed.items():
        versions = [r["version"] for r in recs]
        states = last_token_states(model, tok, [r["prompt"] for r in recs], batch_size=len(recs))
        proj, cos = project_seed(states, versions, unit, layer)
        meta = {"seed_id": seed_id, "is_benign_control": recs[0]["is_benign_control"],
                "category": recs[0]["category"]}
        proj_rows += [{**meta, **row} for row in proj]
        cos_rows += [{**meta, **row} for row in cos]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    proj_df, cos_df = pd.DataFrame(proj_rows), pd.DataFrame(cos_rows)
    proj_df.to_csv(outdir / "projections.csv", index=False)
    cos_df.to_csv(outdir / "cosine_to_en.csv", index=False)
    summary = proj_df.groupby(["is_benign_control", "version"])["projection"].agg(["mean", "std", "count"])
    print(summary.reindex(VERSIONS, level="version").to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--harmful", required=True)
    f.add_argument("--harmless", required=True)
    f.add_argument("--out", required=True)
    pr = sub.add_parser("project")
    pr.add_argument("--model", required=True)
    pr.add_argument("--direction", required=True)
    pr.add_argument("--seeds", required=True)
    pr.add_argument("--split", choices=["test", "train"])
    pr.add_argument("--layer", type=int, help="override the selected layer")
    pr.add_argument("--outdir", required=True)
    args = p.parse_args(argv)
    cmd_fit(args) if args.cmd == "fit" else cmd_project(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
