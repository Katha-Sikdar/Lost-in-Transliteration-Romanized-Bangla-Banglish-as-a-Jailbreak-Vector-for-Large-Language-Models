# Data

This folder holds only the dataset **schema** and **annotation guidelines**.
Harmful prompts and model responses go in `data/raw/` or `data/private/`,
which are git-ignored and must never be pushed to this public repository.

## Planned record format

Each seed prompt has five parallel versions:

| Field | Description |
|---|---|
| `id` | Seed prompt identifier |
| `category` | Harm category (HarmBench-aligned plus Bangladesh-specific) |
| `is_benign_control` | `true` for borderline-but-harmless over-refusal probes |
| `en` | English |
| `bn` | Bangla script |
| `banglish_std` | Banglish, standard spelling |
| `banglish_noisy` | Banglish, real-world noisy spelling |
| `code_mixed` | Banglish–English code-mixed |
| `author_id` | Anonymized native-speaker author |
