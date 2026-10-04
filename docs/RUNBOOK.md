# Runbook: the exact commands for each phase

Read `docs/METHODOLOGY_GUIDE.md` for the *why*; this file is the *how*.
All harmful data stays in `data/raw/` (git-ignored).

## Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # API evaluation, judging, statistics
pip install -r requirements-gpu.txt      # only on the GPU machine (Phases 7-8)
python -m pytest                         # offline tests, no API calls

cp configs/models.example.yaml configs/models.yaml   # fill in model IDs
cp configs/judge.example.yaml  configs/judge.yaml
cp configs/guards.example.yaml configs/guards.yaml

# API keys: put them in .env (git-ignored), then load them:
#   ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GOOGLE_API_KEY=...
set -a && source .env && set +a
```

Try every command first with `configs/dummy.yaml` and the template seeds: no
cost, no network.

## Phase 0–1: ethics, taxonomy, guideline

Documents only: `ETHICS_PROTOCOL.md`, `HARM_TAXONOMY.md`,
`ANNOTATION_GUIDELINE.md`. Derive the noise rules from real Banglish comments
and update `ABBREVIATIONS` / `SOUND_MERGES` in `banglishjail/noise.py`.

## Phase 2: pilot (30 harmful + 10 benign seeds)

```bash
# annotators fill data/raw/pilot_seeds.csv from data/templates/seeds_template.csv
python -m banglishjail.data prefill-bnscript data/raw/pilot_seeds.csv   # then correct by hand
python -m banglishjail.data validate data/raw/pilot_seeds.csv
python -m banglishjail.data split    data/raw/pilot_seeds.csv --train-frac 0

python -m banglishjail.run_eval --config configs/models.yaml --seeds data/raw/pilot_seeds.csv \
    --out data/raw/pilot_responses.jsonl --dry-run          # check the number of calls
python -m banglishjail.run_eval --config configs/models.yaml --seeds data/raw/pilot_seeds.csv \
    --out data/raw/pilot_responses.jsonl

python -m banglishjail.judge --config configs/judge.yaml --responses data/raw/pilot_responses.jsonl \
    --out data/raw/pilot_judged.jsonl --translate

# In the pilot, humans label EVERYTHING (large --per-cell):
python -m banglishjail.labels sample --judged data/raw/pilot_judged.jsonl \
    --out data/raw/pilot_sheet.csv --per-cell 1000
# two annotators fill human_label; the adjudicator writes pilot_sheet_adjudicated.csv
python -m banglishjail.labels agreement --judged data/raw/pilot_judged.jsonl \
    --human data/raw/pilot_sheet_adjudicated.csv --human-b data/raw/pilot_sheet_B.csv
python -m banglishjail.labels merge --judged data/raw/pilot_judged.jsonl \
    --human data/raw/pilot_sheet_adjudicated.csv --out data/raw/pilot_final.jsonl
python -m banglishjail.stats --labels data/raw/pilot_final.jsonl --outdir results/pilot
```

**Decision gate:** read `results/pilot/report.md` and follow Phase 2 of the
methodology guide.

## Phase 3: full dataset

```bash
python -m banglishjail.data validate data/raw/seeds.csv
python -m banglishjail.data split    data/raw/seeds.csv --train-frac 0.3
```

Fill `docs/DATASHEET_TEMPLATE.md` as you go.

## Phase 4: attacks (test split)

```bash
for cond in direct response_lang; do
  python -m banglishjail.run_eval --config configs/models.yaml --seeds data/raw/seeds.csv \
      --split test --condition $cond --out data/raw/responses.jsonl
done
python -m banglishjail.run_eval --config configs/models.yaml --seeds data/raw/seeds.csv \
    --split test --condition noise_sweep --limit 150 --out data/raw/responses.jsonl
# optional: --condition template --templates data/raw/templates.json
# Automated attacks (PAIR/TAP): run HarmBench/JailbreakBench implementations with the
# Banglish seeds and convert their outputs to the same JSONL fields.
```

Re-run any command after an interruption: finished calls are skipped.

## Phase 5: scoring and statistics

```bash
python -m banglishjail.judge --config configs/judge.yaml --responses data/raw/responses.jsonl \
    --out data/raw/judged.jsonl --translate
python -m banglishjail.labels sample --judged data/raw/judged.jsonl \
    --out data/raw/label_sheet.csv --per-cell 20       # aim for >= 400 rows in total
python -m banglishjail.labels agreement --judged data/raw/judged.jsonl \
    --human data/raw/label_sheet_adjudicated.csv --human-b data/raw/label_sheet_B.csv
python -m banglishjail.labels merge --judged data/raw/judged.jsonl \
    --human data/raw/label_sheet_adjudicated.csv --out data/raw/final.jsonl
python -m banglishjail.stats --labels data/raw/final.jsonl --outdir results/main
Rscript analysis/mixed_effects.R results/main/long.csv
```

If judge-vs-human κ on Banglish is below about 0.6, increase human labelling
for those versions and report it.

## Phase 6: guard models

```bash
python -m banglishjail.guard_eval --guards configs/guards.yaml --seeds data/raw/seeds.csv \
    --split test --labels data/raw/final.jsonl --outdir results/guards
```

## Phase 7: mechanistic analysis (GPU)

```bash
python -m banglishjail.mech.tokenization --seeds data/raw/seeds.csv \
    --tokenizers meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --out results/mech/tokenization.csv
# en_harmful.txt / en_harmless.txt: English instructions, one per line
# (e.g. AdvBench and Alpaca subsets, as in Arditi et al. 2024)
python -m banglishjail.mech.refusal_direction fit --model meta-llama/Llama-3.1-8B-Instruct \
    --harmful data/raw/en_harmful.txt --harmless data/raw/en_harmless.txt \
    --out results/mech/llama_direction.pt
python -m banglishjail.mech.refusal_direction project --model meta-llama/Llama-3.1-8B-Instruct \
    --direction results/mech/llama_direction.pt --seeds data/raw/seeds.csv --split test \
    --outdir results/mech/llama
```

## Phase 8: defenses

```bash
# D1 back-transliteration / LLM normalization before the guard
python -m banglishjail.guard_eval --guards configs/guards.yaml --seeds data/raw/seeds.csv \
    --split test --outdir results/guards --normalize xlit --english-words data/raw/english_words.txt
python -m banglishjail.guard_eval --guards configs/guards.yaml --seeds data/raw/seeds.csv \
    --split test --outdir results/guards --normalize llm_en --normalizer-config configs/judge.yaml

# D2 safety fine-tuning (train split only)
python -m banglishjail.run_eval --config configs/models.yaml --seeds data/raw/seeds.csv \
    --split train --out data/raw/responses_train.jsonl            # reference answers
python -m banglishjail.judge --config configs/judge.yaml --responses data/raw/responses_train.jsonl \
    --out data/raw/judged_train.jsonl
python -m banglishjail.defenses.sft_data --seeds data/raw/seeds.csv --refusals data/raw/refusals.json \
    --benign-answers data/raw/judged_train.jsonl --out data/raw/sft_train.jsonl
python -m banglishjail.defenses.train_lora --model meta-llama/Llama-3.1-8B-Instruct \
    --data data/raw/sft_train.jsonl --out checkpoints/llama-banglish-safety
# serve the adapter with vLLM, add it to configs/models.yaml, re-run Phases 4-5 on the test split

# D3 activation steering
python -m banglishjail.defenses.steering --model meta-llama/Llama-3.1-8B-Instruct \
    --direction results/mech/llama_direction.pt --seeds data/raw/seeds.csv --split test \
    --alpha 4.0 --out data/raw/responses_steer.jsonl
# then judge + stats as in Phase 5
```

Also evaluate each defense on benign controls (over-refusal), a general
benchmark (utility), and an adaptive attacker who knows the defense.
