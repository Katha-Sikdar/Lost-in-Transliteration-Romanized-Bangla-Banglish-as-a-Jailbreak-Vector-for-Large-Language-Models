"""Normalize Banglish before moderation (defense D1).

Two methods:
    xlit   word-level back-transliteration to Bangla script with AI4Bharat's
           IndicXlit (pip install ai4bharat-transliteration). Words found in
           an English word list are left unchanged, so code-mixed English
           survives.
    llm    ask a model (any configured client) to rewrite the text in Bangla
           script, or translate it to English.

Usage:
    python -m banglishjail.normalize "tumi kemon acho" --method xlit
"""

import argparse
import re

TOKEN_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")

LLM_INSTRUCTIONS = {
    "bn": ("Rewrite the following text in Bangla script. It may be romanized Bangla (Banglish) "
           "mixed with English. Keep the meaning exactly. Output only the rewritten text."),
    "en": ("Translate the following text into English. It may be romanized Bangla (Banglish), "
           "Bangla script or mixed. Output only the translation."),
}


class XlitNormalizer:
    def __init__(self, english_words=frozenset(), beam_width=4):
        from ai4bharat.transliteration import XlitEngine

        self.engine = XlitEngine("bn", beam_width=beam_width, rescore=True)
        self.english_words = english_words

    def _word(self, word):
        out = self.engine.translit_word(word, lang_code="bn", topk=1)
        if isinstance(out, dict):  # older package versions return {"bn": [...]}
            out = out.get("bn", [word])
        return out[0] if out else word

    def __call__(self, text):
        parts = []
        for tok in TOKEN_RE.findall(text):
            if tok[0].isalpha() and tok.lower() not in self.english_words:
                parts.append(self._word(tok))
            else:
                parts.append(tok)
        return "".join(parts)


class LLMNormalizer:
    def __init__(self, client, target="bn"):
        self.client = client
        self.instruction = LLM_INSTRUCTIONS[target]

    def __call__(self, text):
        result = self.client.generate(f"{self.instruction}\n\n{text}")
        if result["stop_reason"] == "refusal" or not result["text"].strip():
            return None
        return result["text"].strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--method", choices=["xlit"], default="xlit")
    args = p.parse_args(argv)
    print(XlitNormalizer()(args.text))


if __name__ == "__main__":
    main()
