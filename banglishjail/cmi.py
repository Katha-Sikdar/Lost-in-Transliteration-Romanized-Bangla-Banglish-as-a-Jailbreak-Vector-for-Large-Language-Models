"""Code-Mixing Index (CMI) for Banglish-English text.

CMI = 100 * (1 - max_lang_tokens / (n - u)), where n is the number of tokens
and u the number of language-independent tokens (numbers, punctuation,
names). CMI is 0 for monolingual text and rises with mixing (Das and
Gambäck, 2014; verify the citation before use).

Tagging English vs. Banglish words automatically is unreliable because many
romanized Bangla words are also English strings (e.g. "na", "to", "be").
Prefer manual tags written as word/en, word/bn, word/univ. The heuristic
tagger is only for quick checks.

Usage:
    python -m banglishjail.cmi "bro/en tumi/bn how/en acho/bn ?/univ"
    python -m banglishjail.cmi "bro tumi how acho?" --english-words words.txt
"""

import argparse
import re
from collections import Counter

TAGS = {"en", "bn", "univ"}
TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^\sA-Za-z\d]")


def parse_tagged(text):
    """Parse 'word/tag word/tag' into [(word, tag)]."""
    pairs = []
    for item in text.split():
        word, sep, tag = item.rpartition("/")
        if not sep or tag not in TAGS:
            raise ValueError(f"token {item!r} needs a /en, /bn or /univ tag")
        pairs.append((word, tag))
    return pairs


def heuristic_tags(text, english_words, banglish_words=frozenset()):
    """Tag tokens using word lists. Banglish list wins on overlap."""
    pairs = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if not tok[0].isalpha():
            tag = "univ"
        elif low in banglish_words:
            tag = "bn"
        elif low in english_words:
            tag = "en"
        else:
            tag = "bn"
        pairs.append((tok, tag))
    return pairs


def cmi(pairs):
    counts = Counter(tag for _, tag in pairs)
    n = len(pairs)
    u = counts["univ"]
    if n == u:
        return 0.0
    max_lang = max(counts["en"], counts["bn"])
    return 100.0 * (1 - max_lang / (n - u))


def _load_words(path):
    if not path:
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(w.strip().lower() for w in f if w.strip())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--english-words", help="one word per line; enables heuristic tagging")
    p.add_argument("--banglish-words", help="one word per line; overrides English matches")
    args = p.parse_args(argv)
    if args.english_words:
        pairs = heuristic_tags(args.text, _load_words(args.english_words), _load_words(args.banglish_words))
    else:
        pairs = parse_tagged(args.text)
    print(" ".join(f"{w}/{t}" for w, t in pairs))
    print(f"CMI = {cmi(pairs):.1f}")


if __name__ == "__main__":
    main()
