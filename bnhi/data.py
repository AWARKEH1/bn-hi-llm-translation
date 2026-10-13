"""Bengali -> Hindi parallel data.

The sentences come from the WMT21 Bengali-Hindi test set (FLORES-101 devtest,
CC BY-SA 4.0). The original project fine-tuned on its first 503 sentences and
then evaluated on the whole set, so half of the test sentences had been seen
in training. Here the two halves are kept strictly separate:

* ``train`` - 503 sentences: fine-tuning data and pool of few-shot examples
* ``test``  - 509 sentences: used only for the final evaluation
"""

from __future__ import annotations

import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

Pair = tuple[str, str]


def load_split(name: str, data_dir: Path = DATA_DIR) -> list[Pair]:
    """Return the (bengali, hindi) pairs of ``train`` or ``test``."""
    bn = (data_dir / f"{name}.bn").read_text(encoding="utf-8").splitlines()
    hi = (data_dir / f"{name}.hi").read_text(encoding="utf-8").splitlines()
    if len(bn) != len(hi):
        raise ValueError(f"{name}: {len(bn)} Bengali vs {len(hi)} Hindi lines")
    return list(zip(bn, hi))


def train_val_split(pairs: list[Pair], n_val: int = 50, seed: int = 0) -> tuple[list[Pair], list[Pair]]:
    """Hold out a few training pairs to monitor the fine-tuning loss."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def few_shot_examples(pairs: list[Pair], k: int = 5, seed: int = 0, max_words: int = 25) -> list[Pair]:
    """Pick k fixed, reasonably short examples from the TRAIN split."""
    rng = random.Random(seed)
    short = [p for p in pairs if len(p[0].split()) <= max_words]
    return rng.sample(short, k)
