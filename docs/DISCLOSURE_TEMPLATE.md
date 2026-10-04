# Responsible Disclosure Template

Send to each affected model provider **before** submitting the paper. Use the
provider's official safety or vulnerability reporting channel. Give them a
reasonable window (commonly about 90 days) and state the planned
publication date.

---

**Subject:** Safety evaluation finding: romanized Bangla (Banglish) requests bypass safeguards in [model]

Dear [provider] safety team,

We are researchers at [affiliation] studying LLM safety in low-resource
languages. In an evaluation run between [dates], we found that harmful
requests written in romanized Bangla ("Banglish") are answered by [model ID]
substantially more often than the same requests in English.

**Summary of findings**
- Attack success rate (strict): English [x%], Bangla script [y%], Banglish [z%], noisy Banglish [w%].
- [Guard model / moderation endpoint] detected [a%] of harmful English prompts but [b%] of the Banglish versions.
- Most affected categories: [list].

**Method:** [n] native-authored seeds, each in six parallel versions; single-turn requests with default settings; responses labelled by human annotators and a validated LLM judge.

**What we can share:** a sample of [n] prompt–response pairs and the evaluation code, under confidentiality.

**Planned publication:** we plan to submit to [venue] on [date] and will not release harmful prompts publicly; the benchmark will be released under gated access.

We are happy to discuss the findings or share more detail.

Kind regards,
[name, affiliation, email]

---

## Disclosure log

| Provider | Channel | Date sent | Acknowledged | Notes |
|---|---|---|---|---|
| | | | | |
