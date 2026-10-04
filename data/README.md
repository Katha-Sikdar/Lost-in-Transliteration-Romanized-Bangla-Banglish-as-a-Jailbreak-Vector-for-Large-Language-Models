# Data

This folder holds only the dataset **schema** and **annotation guidelines**.
Harmful prompts and model responses go in `data/raw/` or `data/private/`,
which are git-ignored and must never be pushed to this public repository.

## Planned record format

Each seed prompt has five parallel versions:

| Field | Description |
|---|---|
| `id` | Seed prompt identifier (unique) |
| `category` | Harm category (HarmBench-aligned plus Bangladesh-specific) |
| `is_benign_control` | `true` for borderline-but-harmless over-refusal probes |
| `en` | English |
| `bn` | Bangla script |
| `banglish_std` | Banglish, standard spelling |
| `banglish_noisy` | Banglish, real-world noisy spelling |
| `code_mixed` | Banglish–English code-mixed |
| `split` | `test` or `train` (filled by `python -m banglishjail.data split`) |
| `notes` | Optional free text |

The authoritative column list is in `banglishjail/data.py`, and a filled
example (benign controls only) is `templates/seeds_template.csv`.

## Templates

| File | Purpose |
|---|---|
| `templates/seeds_template.csv` | Starting spreadsheet for prompt authors |
| `templates/refusals_template.json` | Refusal texts for safety fine-tuning (defense D2) |

## Where things go

| Data | Location |
|---|---|
| Seed spreadsheet, responses, judged and final labels, label sheets | `data/raw/` (git-ignored) |
| Aggregate results (metrics, tests) | `results/` |
