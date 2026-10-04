"""LoRA safety fine-tuning for defense D2 (needs a GPU).

Trains on the JSONL from sft_data.py. Serve the result with vLLM
(`--enable-lora`) or merge it, add it to configs/models.yaml as a new model,
and run the normal evaluation on the TEST split.

Written against trl>=0.12 and peft>=0.13; if the TRL API has changed in
your installed version, check its SFTTrainer docs.

Usage:
    python -m banglishjail.defenses.train_lora --model meta-llama/Llama-3.1-8B-Instruct \
        --data data/raw/sft_train.jsonl --out checkpoints/llama-banglish-safety
"""

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    dataset = load_dataset("json", data_files=args.data, split="train").select_columns(["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=dataset,
        peft_config=LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                       per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
                       lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
                       bf16=True, seed=args.seed, report_to="none"),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved LoRA adapter to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
