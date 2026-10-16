"""Logit lens: does the model "think in English" when translating Bengali -> Hindi?

Following Wendler et al. (2024), "Do Llamas Work in English?", each layer's
hidden state at the last prompt position is projected onto the vocabulary
(final norm + output matrix). For every word we track, layer by layer:

* p(English) - probability of the English translation of the word (e.g. "flower")
* p(Hindi)   - probability of the correct Hindi answer (e.g. "फूल")

If English rises in the middle layers before Hindi takes over in the last
layers, the model goes through an English-like "concept space".

Usage:
    python -m bnhi.logit_lens
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# (Bengali, Hindi, English)
WORDS = [
    ("ফুল", "फूल", "flower"), ("জল", "पानी", "water"), ("বই", "किताब", "book"),
    ("বাড়ি", "घर", "house"), ("সূর্য", "सूरज", "sun"), ("চাঁদ", "चाँद", "moon"),
    ("গাছ", "पेड़", "tree"), ("পাখি", "पक्षी", "bird"), ("মাছ", "मछली", "fish"),
    ("নদী", "नदी", "river"), ("মা", "माँ", "mother"), ("রাজা", "राजा", "king"),
]


def build_prompt(index: int, n_demos: int = 4) -> str:
    """Few-shot word-translation prompt that ends right before the Hindi answer."""
    demos = [w for i, w in enumerate(WORDS) if i != index][:n_demos]
    lines = [f'বাংলা: "{bn}" - हिन्दी: "{hi}"' for bn, hi, _ in demos]
    lines.append(f'বাংলা: "{WORDS[index][0]}" - हिन्दी: "')
    return "\n".join(lines)


def first_token_ids(tokenizer, word: str, variants: bool = False) -> list[int]:
    """Id(s) of the first token of a word (with/without leading space and capital)."""
    forms = [word, " " + word, word.capitalize(), " " + word.capitalize()] if variants else [word]
    return sorted({tokenizer(f, add_special_tokens=False)["input_ids"][0] for f in forms})


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from .model import load

    model, tokenizer = load()
    norm, head = model.model.norm, model.lm_head

    p_en, p_hi, top_tokens = [], [], {}
    for i, (bn, hi, en) in enumerate(WORDS):
        enc = tokenizer(build_prompt(i), return_tensors="pt").to(model.device)
        with torch.no_grad():
            hidden = model(**enc, output_hidden_states=True).hidden_states
        en_ids, hi_ids = first_token_ids(tokenizer, en, variants=True), first_token_ids(tokenizer, hi)
        row_en, row_hi, row_top = [], [], []
        for layer, h in enumerate(hidden[1:], start=1):
            h_last = h[0, -1]
            # the last hidden state is already normalised by the model
            h_last = h_last if layer == len(hidden) - 1 else norm(h_last)
            probs = torch.softmax(head(h_last).float(), dim=-1)
            row_en.append(probs[en_ids].sum().item())
            row_hi.append(probs[hi_ids].sum().item())
            row_top.append(tokenizer.decode([int(probs.argmax())]))
        p_en.append(row_en)
        p_hi.append(row_hi)
        top_tokens[f"{bn} -> {hi} ({en})"] = row_top
        print(f"{en:7s} final top token: {row_top[-1]!r}")

    p_en, p_hi = np.array(p_en), np.array(p_hi)
    layers = np.arange(1, p_en.shape[1] + 1)
    plt.figure(figsize=(7, 4))
    for values, label, color in [(p_en, "English word (e.g. 'flower')", "#4C72B0"),
                                 (p_hi, "Hindi answer (first token)", "#DD8452")]:
        mean, std = values.mean(0), values.std(0)
        plt.plot(layers, mean, color=color, label=label)
        plt.fill_between(layers, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                         color=color, alpha=0.15)
    plt.xlabel("layer"); plt.ylabel("probability (logit lens)")
    plt.title(f"Bengali → Hindi word translation, {len(WORDS)} words")
    plt.legend(); plt.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    plt.savefig(RESULTS / "figures" / "logit_lens.png", dpi=150)

    summary = {
        "n_words": len(WORDS),
        "peak_layer_english": int(layers[p_en.mean(0).argmax()]),
        "peak_prob_english": round(float(p_en.mean(0).max()), 3),
        "final_prob_english": round(float(p_en.mean(0)[-1]), 3),
        "final_prob_hindi": round(float(p_hi.mean(0)[-1]), 3),
        "mean_p_english_per_layer": [round(float(v), 4) for v in p_en.mean(0)],
        "mean_p_hindi_per_layer": [round(float(v), 4) for v in p_hi.mean(0)],
        "top_token_per_layer": top_tokens,
    }
    (RESULTS / "logit_lens.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith(("mean_", "top_"))}, indent=2))


if __name__ == "__main__":
    main()
