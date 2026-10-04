# Lost in Transliteration: Why Romanization Breaks LLM Safety

A controlled, mechanistic study of LLM safety on **Banglish**: Bangla written
in the Latin alphabet, with no standard spelling and frequent English
code-mixing.

Recent benchmarks (BanglaVeilGuard, BanglaSafe, and work on code-mixed
phonetic perturbations; see `paper/intro_related_work.tex`) show *that*
Banglish weakens LLM safety. This project asks *why*, and how to fix it.

**Core hypothesis:** LLMs understand Banglish through the same pathways as
English, but their refusal behaviour does not transfer to the Latin-script
form of Bangla.

## Design

Each seed request is written in six parallel versions. The first four form a
language × script factorial, which separates the effect of the *script* from
the effect of the *language*:

| | Latin script | Bengali script |
|---|---|---|
| **English** | `en`: How are you? | `en_bnscript`: হাউ আর ইউ? |
| **Bangla** | `banglish_std`: tumi kemon acho? | `bn`: তুমি কেমন আছো? |

plus `banglish_noisy` (real-world spelling) and `code_mixed` (Banglish–English).

## Research questions

1. **Script vs. language:** does the safety gap come from the Latin script, the Bangla language, or their combination?
2. **Comprehension-controlled ASR:** how much of the gap remains after excluding responses where the model misunderstood?
3. **Dose–response:** how does attack success grow with real-world spelling noise?
4. **Mechanism:** does harmful Banglish project less onto the refusal direction, and does adding the direction restore refusals?
5. **Defenses under adaptive attack:** do normalization, safety fine-tuning, steering and existing Bangla guards (BanglaVeilGuard) survive an attacker who knows the defense?

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

- [x] Introduction and Related Work draft (repositioned against 2024–2026 related work)
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
- [ ] Read the full BanglaVeilGuard, BanglaSafe and comprehension–containment papers; refine Related Work
- [ ] Results, Discussion and Abstract

## Ethics and responsible release

- Harmful prompts and model outputs are **never committed to this public
  repository**. `.gitignore` blocks the raw-data folders.
- The final benchmark will be released under gated access only.
- Findings will be disclosed to affected model providers before publication.
- Annotator studies require ethics/IRB approval before data collection.
