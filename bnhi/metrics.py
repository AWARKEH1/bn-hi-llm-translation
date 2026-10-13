"""Translation metrics: BLEU, chrF++ and share of outputs written in Devanagari."""

from __future__ import annotations


def devanagari_ratio(text: str) -> float:
    """Share of letters that are Devanagari (Hindi script) in ``text``."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters)


def score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    from sacrebleu.metrics import BLEU, CHRF

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    return {
        "bleu": round(BLEU().corpus_score(hypotheses, [references]).score, 2),
        # chrF++ (character n-grams + word bigrams) is more reliable than BLEU for Indic languages
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, [references]).score, 2),
        # outputs that are mostly in Devanagari (the original project saw romanised Hindi)
        "devanagari_outputs_pct": round(
            100 * sum(devanagari_ratio(h) >= 0.8 for h in hypotheses) / len(hypotheses), 1),
    }
