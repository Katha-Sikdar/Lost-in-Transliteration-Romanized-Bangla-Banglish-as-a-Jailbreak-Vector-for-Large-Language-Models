"""Tools for evaluating LLM safety on romanized Bangla (Banglish)."""

# en / en_bnscript / bn / banglish_std form a 2x2 factorial of language x script.
VERSIONS = ["en", "en_bnscript", "bn", "banglish_std", "banglish_noisy", "code_mixed"]

LANGUAGE = {"en": "english", "en_bnscript": "english", "bn": "bangla", "banglish_std": "bangla"}
SCRIPT = {"en": "latin", "en_bnscript": "bengali", "bn": "bengali", "banglish_std": "latin"}

HARMFUL_LABELS = ["R", "M", "P", "H"]  # Refusal, Misunderstood, Partial, Harmful
BENIGN_LABELS = ["R", "M", "A"]  # (over-)Refusal, Misunderstood, Answered
