"""Chinese front end (see zh_text.py). Re-exported under the upstream name
so prepare/annotate/train/infer keep working unchanged."""
from .zh_text import *  # noqa: F401,F403
from .zh_text import (PAD, BOS, EOS, SP, UNK, SPECIALS, PUNCTUATION, PHONES, SYMBOLS,
                      SYMBOL_TO_ID, PHONE_SET, PHONE_IDS, VOWEL_SET, VOICELESS,
                      NothingToPronounce, Phonemized, frontend, frontend_identity,
                      frontend_runtime, frontend_fingerprint, normalize_text, chunks,
                      tokenize, phonemize, is_phone, FRONTEND_VERSION, TONE_LEVELS)
VOWELS = tuple(sorted(VOWEL_SET))
CONSONANTS = tuple(p for p in PHONES if p not in VOWEL_SET)
FRONTENDS = ("zh",)
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
def set_frontend(name):
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}")
