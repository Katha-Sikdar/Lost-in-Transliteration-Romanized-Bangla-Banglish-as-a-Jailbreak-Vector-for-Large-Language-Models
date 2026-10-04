"""Spelling-noise perturbation for Banglish and noise-level measurement.

Used for the noise-sweep condition (A1): the standard Banglish version is
perturbed at a chosen word-level rate. The default rules are a starting point;
replace them with the rule set you derive from real social-media Banglish in
Phase 1b (see docs/ANNOTATION_GUIDELINE.md) and report that rule set in the paper.

Usage:
    python -m banglishjail.noise "tumi kemon acho bhalo?" --level 1.0
"""

import argparse
import random
import re

# Whole-word abbreviations commonly seen in Banglish chat.
ABBREVIATIONS = {
    "ki": "k",
    "kemon": "kmn",
    "bhalo": "vlo",
    "valo": "vlo",
    "tumi": "tmi",
    "tomake": "tmk",
    "amake": "amk",
    "keno": "kno",
    "kintu": "kntu",
    "jonno": "jnno",
    "korbo": "krbo",
    "please": "plz",
    "thanks": "thnx",
}

# Sound-merging substitutions, applied left to right.
SOUND_MERGES = [("bh", "v"), ("ph", "f"), ("sh", "s"), ("ee", "i"), ("oo", "u")]

VOWELS = set("aeiou")
WORD_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def _match_case(original, new):
    if original.isupper() and len(original) > 1:
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _vowel_drop(word):
    """Drop interior vowels of words with 4+ letters: kemon -> kmn."""
    if len(word) < 4:
        return word
    inner = "".join(c for c in word[1:-1] if c not in VOWELS)
    return word[0] + inner + word[-1]


def perturb_word(word):
    """Return (new_word, rule_name) for one alphabetic word."""
    lower = word.lower()
    if lower in ABBREVIATIONS:
        return _match_case(word, ABBREVIATIONS[lower]), "abbreviation"
    merged = lower
    for src, dst in SOUND_MERGES:
        merged = merged.replace(src, dst)
    if merged != lower:
        return _match_case(word, merged), "sound_merge"
    dropped = _vowel_drop(lower)
    if dropped != lower:
        return _match_case(word, dropped), "vowel_drop"
    return word, None


def perturb(text, level, seed=0):
    """Perturb each word with probability `level` (0..1).

    Deterministic for a given (text, level, seed). Returns (new_text, rules_applied).
    """
    if not 0 <= level <= 1:
        raise ValueError("level must be between 0 and 1")
    rng = random.Random(f"{seed}|{text}")
    out, applied = [], []
    for token in WORD_RE.findall(text):
        if token.isalpha() and rng.random() < level:
            new, rule = perturb_word(token)
            if rule:
                applied.append(rule)
            out.append(new)
        else:
            out.append(token)
    return "".join(out), applied


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def noise_level(standard, noisy):
    """Character edit distance normalised by the standard version's length."""
    if not standard:
        return 0.0
    return levenshtein(standard.lower(), noisy.lower()) / len(standard)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    new, rules = perturb(args.text, args.level, args.seed)
    print(new)
    print(f"rules: {rules}  noise_level: {noise_level(args.text, new):.3f}")


if __name__ == "__main__":
    main()
