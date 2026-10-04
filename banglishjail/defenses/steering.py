"""Activation steering defense (D3) and causal check (Phase 7, step 4).

Adds alpha * (unit refusal direction) to the residual stream at the chosen
layer for every token, then generates. If refusals return for harmful
Banglish, the direction is causally involved. Measure benign over-refusal
too: steering usually increases it.

Output is in the run_eval log format, so it goes straight into
`banglishjail.judge`. The model name gets a "+steer<alpha>" suffix.

Usage:
    python -m banglishjail.defenses.steering --model meta-llama/Llama-3.1-8B-Instruct \
        --direction results/mech/llama_direction.pt --seeds data/raw/seeds.csv \
        --split test --alpha 4.0 --out data/raw/responses_steer.jsonl
"""

import argparse
import datetime as dt
import sys

import numpy as np

from banglishjail.data import expand, load_seeds
from banglishjail.io import JsonlWriter, read_jsonl
from banglishjail.mech.refusal_direction import chat_format, load_model


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers for this architecture")


def add_steering_hook(model, layer, vector):
    """hidden_states[layer] is the output of decoder block layer-1."""
    import torch

    block = decoder_layers(model)[layer - 1]
    vec = torch.tensor(vector)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + vec.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    return block.register_forward_hook(hook)


def main(argv=None):
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--layer", type=int)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    saved = torch.load(args.direction)
    layer = args.layer if args.layer is not None else saved["layer"]
    direction = saved["directions"][layer].numpy()
    vector = (args.alpha * direction / np.linalg.norm(direction)).astype(np.float32)

    model, tok = load_model(args.model)
    handle = add_steering_hook(model, layer, vector)
    name = f"{args.model.split('/')[-1]}+steer{args.alpha}"
    done = {r["key"] for r in read_jsonl(args.out)}
    writer = JsonlWriter(args.out)
    try:
        for rec in expand(load_seeds(args.seeds), split=args.split):
            key = "|".join([name, "direct", "", rec["seed_id"], rec["version"], "0"])
            if key in done:
                continue
            enc = tok(chat_format(tok, rec["prompt"]), return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            writer.write({"key": key, **rec, "condition": "direct", "variant": "", "sample": 0,
                          "sent_prompt": rec["prompt"], "model_name": name, "provider": "hf_local",
                          "model": args.model, "params": {"alpha": args.alpha, "layer": layer, "greedy": True},
                          "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "error": None,
                          "response": text, "stop_reason": "end_turn", "refusal_category": None,
                          "served_model": args.model})
    finally:
        handle.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
