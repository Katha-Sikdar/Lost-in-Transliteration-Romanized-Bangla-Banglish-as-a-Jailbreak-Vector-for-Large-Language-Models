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
| `paper/` | LaTeX draft (Introduction, Related Work) and `references.bib` |
| `data/` | Dataset schema and annotation guidelines (no harmful content) |
| `scripts/` | Evaluation and analysis scripts |

## Status

- [x] Introduction and Related Work draft
- [ ] Annotation guideline and dataset schema
- [ ] Pilot evaluation (20–30 prompts)
- [ ] Full dataset collection
- [ ] Model and guard-model evaluation
- [ ] Refusal-direction analysis
- [ ] Defenses
- [ ] Methodology, Results and Discussion sections

## Ethics and responsible release

- Harmful prompts and model outputs are **never committed to this public
  repository**. `.gitignore` blocks the raw-data folders.
- The final benchmark will be released under gated access only.
- Findings will be disclosed to affected model providers before publication.
- Annotator studies require ethics/IRB approval before data collection.
