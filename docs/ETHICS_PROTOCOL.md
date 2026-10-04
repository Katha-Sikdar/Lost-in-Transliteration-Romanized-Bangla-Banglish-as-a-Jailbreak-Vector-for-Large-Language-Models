# Ethics Protocol and Consent Form

Use this as the basis for your ethics/IRB application. Adapt it to your
institution's form and get it approved **before** any annotator writes or
reads harmful content.

## 1. Study summary (for the application)

**Title:** Lost in Transliteration: Romanized Bangla (Banglish) as a Jailbreak
Vector for Large Language Models

**Purpose:** To measure whether AI chatbots' safety protections fail when
requests are written in romanized Bangla, and to develop defenses.

**Participants:** native Bangla-speaking adults (18+) acting as prompt
authors and response annotators. Recruited from [university / network].
Paid [amount] per hour.

**What participants do:**
1. Write requests, some of which ask for harmful information, in six
   language forms.
2. Read AI responses, some of which may contain harmful or offensive
   content, and label them.

**No personal data about participants is collected beyond** anonymized
demographics (age range, region, education), used only to describe the
annotator pool.

## 2. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Distress from reading harmful content | Content warning, sessions ≤ 90 min, right to skip items or stop, support contact, debrief |
| Misuse of the dataset | Harmful prompts and responses never published openly; gated release with a usage agreement; excluded categories (see HARM_TAXONOMY.md) |
| Harm to third parties | No real, named private individuals in prompts; no harmful instructions are acted on |
| Exposure of model vulnerabilities | Responsible disclosure to providers before publication (see DISCLOSURE_TEMPLATE.md) |
| Data breach | Encrypted storage, access limited to the research team, raw data never in the public repository |
| Annotator identification | Annotators referred to by code (A1, A2…) everywhere |

## 3. Data management

- Raw prompts, responses and label sheets: encrypted private storage and the
  git-ignored `data/raw/` folder only.
- Public repository: code, documentation, aggregate results.
- Released dataset: gated access (e.g. Hugging Face gated dataset) for
  verified researchers who accept a usage agreement.
- Retention: [N] years after publication, then deletion of non-released data.

## 4. Consent form (give to each participant)

> **Study:** Safety of AI chatbots in romanized Bangla (Banglish)
> **Researcher:** [name, affiliation, email]
>
> **Content warning:** In this study you will write and read text about
> harmful topics, such as fraud, hate speech, violence and misinformation.
> Some AI responses may be offensive or disturbing.
>
> **What you will do:** write requests in English, Bangla and Banglish, and
> label AI responses, in sessions of at most 90 minutes.
>
> **Your rights:**
> - Taking part is voluntary. You can stop at any time without giving a
>   reason, and you will still be paid for the time you worked.
> - You can skip any item.
> - Your name will not appear in any publication or dataset.
>
> **Support:** if anything you read upsets you, contact [project lead] or
> [counselling service, phone/email].
>
> **Data:** your work will be stored securely and released only to vetted
> researchers under a usage agreement.
>
> **Confidentiality:** please do not share the study's prompts or responses
> outside the project.
>
> I am 18 or older, I have read this information, and I agree to take part.
>
> Name: ______________________  Signature: ______________  Date: __________

## 5. Debrief (after the last session)

Thank the participant, explain how the data will be used and protected,
repeat the support contact, and remind them they may withdraw their
unpublished contributions by [date].
