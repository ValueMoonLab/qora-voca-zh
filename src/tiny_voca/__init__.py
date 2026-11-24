"""qora-voca-zh: a small three-voice Mandarin TTS trained from scratch.

Fork of tiny-voca (Apache-2.0, Muhammed Nazeem) with a Chinese front end.
The speech model loads no pretrained weights. A pretrained Mandarin phoneme
recogniser is used only by the annotation step, and an ASR model only by
evaluation.
"""

from .config import Config
from .model import TinyVoca

__all__ = ["Config", "TinyVoca"]
