# Scripts

The evaluation code lives in the `banglishjail/` Python package and runs with
`python -m banglishjail.<module>`. See `docs/RUNBOOK.md` for the exact command
for each phase.

| Module | Phase | What it does |
|---|---|---|
| `data` | 3 | Validate the seed spreadsheet; assign test/train splits |
| `noise` | 1, 4 | Banglish spelling-noise rules; noise level (edit distance) |
| `cmi` | 3 | Code-Mixing Index |
| `run_eval` | 4 | Send prompts to models, log every response (resumable) |
| `judge` | 5 | LLM judge with two-axis rubric (understood + harmfulness) |
| `labels` | 5 | Human labelling sheets, Cohen's kappa, final labels |
| `stats` | 5 | ASR, comprehension, over-refusal, bootstrap CIs, McNemar + Holm |
| `guard_eval` | 6, 8 | Guard-model recall / false positives, with optional normalization |
| `normalize` | 8 | Back-transliteration (IndicXlit) or LLM normalization |
| `mech.tokenization` | 7 | Tokens per version |
| `mech.refusal_direction` | 7 | Refusal direction, projections, cosine similarity |
| `defenses.sft_data` / `train_lora` | 8 | Safety fine-tuning data and LoRA training |
| `defenses.steering` | 7, 8 | Activation steering / causal check |

`analysis/mixed_effects.R` fits the mixed-effects logistic regression.
