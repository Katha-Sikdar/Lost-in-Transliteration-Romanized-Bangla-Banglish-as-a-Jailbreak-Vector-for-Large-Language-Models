# Lost in Transliteration: Romanized Bangla (Banglish) as a Jailbreak Vector for Large Language Models

Research project studying whether LLM safety alignment and guard models fail
on **Banglish**, Bangla written in the Latin alphabet with no standard
spelling and frequent English code-mixing.

**Core hypothesis:** LLMs understand Banglish well enough to follow harmful
requests, but not well enough to recognize them as harmful.

## Research questions

1. Is the attack success rate higher for Banglish than for English and for Bangla script?
2. Does real-world spelling variation (vowel dropping, abbreviations) raise it further?
3. Do guard models (Llama Guard, ShieldGemma, moderation APIs) detect harmful Banglish?
4. Does harmful Banglish activate the model's refusal direction less strongly?
5. Do back-transliteration and Banglish safety fine-tuning close the gap?

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX: `main.tex`, Introduction + Related Work, Methodology, `references.bib` |
| `docs/` | Methodology guide, runbook, annotation guideline, harm taxonomy, ethics protocol and consent form, datasheet and disclosure templates |
| `banglishjail/` | Evaluation pipeline (Python), run with `python -m banglishjail.<module>` |
| `configs/` | Example model, judge and guard configs; `dummy.yaml` for free offline dry runs |
| `data/` | Dataset schema and templates (no harmful content) |
| `analysis/` | Mixed-effects regression in R |
| `tests/` | Offline tests (no API calls) |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                                   # offline tests
python -m banglishjail.run_eval --config configs/dummy.yaml \
    --seeds data/templates/seeds_template.csv --out data/raw/dry_run.jsonl
```

Then follow `docs/RUNBOOK.md` phase by phase. Read `docs/METHODOLOGY_GUIDE.md`
for why each step exists.

## Status

- [x] Introduction and Related Work draft
- [x] Methodology section draft
- [x] Annotation guideline, harm taxonomy, ethics protocol
- [x] Evaluation, scoring, statistics, guard, mechanistic and defense code (tested offline)
- [ ] Ethics/IRB approval
- [ ] Noise rules derived from real Banglish comments
- [ ] Pilot (30 harmful + 10 benign seeds)
- [ ] Full dataset collection
- [ ] Model and guard-model evaluation
- [ ] Refusal-direction analysis
- [ ] Defenses
- [ ] Results, Discussion and Abstract

## Ethics and responsible release

- Harmful prompts and model outputs are **never committed to this public
  repository**. `.gitignore` blocks the raw-data folders.
- The final benchmark will be released under gated access only.
- Findings will be disclosed to affected model providers before publication.
- Annotator studies require ethics/IRB approval before data collection.
