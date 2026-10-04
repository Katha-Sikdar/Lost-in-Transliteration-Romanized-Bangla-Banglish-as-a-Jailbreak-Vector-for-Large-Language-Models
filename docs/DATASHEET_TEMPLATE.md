# Datasheet: BanglishJail

Following "Datasheets for Datasets" (Gebru et al., 2021). Fill in each item;
reviewers increasingly expect this for benchmark papers.

## Motivation
- **Purpose:** evaluate LLM safety on romanized Bangla (Banglish), compared with English and Bangla script.
- **Creators and funding:** [names, affiliations, funding]

## Composition
- **Instances:** [N] seeds × 6 versions = [N×6] prompts.
- **Harmful vs. benign:** [n_harmful] harmful seeds, [n_benign] benign controls.
- **Categories:** see HARM_TAXONOMY.md; counts per category: [table].
- **Splits:** test [n] seeds, train [n] seeds (split by seed, stratified by category).
- **Sources:** [x]% native-authored, [y]% adapted from [MultiJail / HarmBench] (list IDs).
- **Noise statistics:** mean noise level of `banglish_noisy` vs. `banglish_std` = [value]; rules used: [link].
- **Code-mixing statistics:** CMI mean [value], range [min–max] on a tagged sample of [n].
- **Sensitive content:** yes, harmful requests; excluded categories listed in HARM_TAXONOMY.md.
- **Personal information:** none; no real private individuals.

## Collection process
- **Authors:** [n] native Bangla speakers; regions [..]; age range [..]; paid [..].
- **Time frame:** [dates].
- **Ethics approval:** [board, approval number].
- **Quality control:** second-annotator check; κ = [value]; [n] versions fixed, [n] dropped.

## Uses
- **Intended:** safety evaluation and defense research.
- **Not intended:** training models to produce harmful content; any attempt to cause real-world harm.

## Distribution
- **Access:** gated, with a usage agreement, at [URL].
- **License:** [e.g. CC BY-NC 4.0 with additional usage terms].

## Maintenance
- **Contact:** [email].
- **Errata and updates:** [process].

## Known limitations
- Standard Banglish spelling follows one convention; real usage varies more.
- Dialects (Sylheti, Chittagonian) are [not / partly] covered.
- Model results reflect versions queried on [dates]; APIs change over time.
