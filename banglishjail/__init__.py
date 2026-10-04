"""Tools for evaluating LLM safety on romanized Bangla (Banglish)."""

VERSIONS = ["en", "bn", "banglish_std", "banglish_noisy", "code_mixed"]

HARMFUL_LABELS = ["R", "M", "P", "H"]  # Refusal, Misunderstood, Partial, Harmful
BENIGN_LABELS = ["R", "M", "A"]  # (over-)Refusal, Misunderstood, Answered
