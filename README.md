# Bengali → Hindi machine translation with an open LLM

> How well does an open 7B LLM translate Bengali into Hindi — and does LoRA fine-tuning beat few-shot prompting?
> A clean, leakage-free re-run of a university project with **Qwen2.5-7B-Instruct** (4-bit), plus a **logit-lens** look inside the model.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Transformers](https://img.shields.io/badge/🤗-Transformers%20%7C%20PEFT-yellow) ![QLoRA](https://img.shields.io/badge/QLoRA-4--bit-green)

## What is compared

All systems translate the **same 509 held-out sentences** with the **same prompt** and **greedy decoding** (deterministic):

| System | What the model sees |
|---|---|
| Zero-shot | instruction + sentence |
| 5-shot prompting | instruction + 5 example pairs from the training split + sentence |
| QLoRA fine-tuned | instruction + sentence, after 2 epochs of LoRA on 453 training pairs (50 kept for validation) |

Metrics: **BLEU**, **chrF++** (more reliable for Indic languages) and the share of outputs actually written in Devanagari.

## Results

*509 test sentences (WMT21 bn-hi / FLORES-101 devtest). Run: [`colab_run.ipynb`](colab_run.ipynb).*

| System | BLEU | chrF++ | Devanagari outputs |
|---|---|---|---|
| Zero-shot | _to be filled_ | _to be filled_ | _to be filled_ |
| 5-shot prompting | _to be filled_ | _to be filled_ | _to be filled_ |
| QLoRA fine-tuned | _to be filled_ | _to be filled_ | _to be filled_ |

<!-- figures: results/figures/finetune_loss.png, results/figures/logit_lens.png -->

### Logit lens

For 12 simple words (flower, water, book…), each layer's hidden state is projected onto the vocabulary while the model translates the Bengali word into Hindi. If the **English** word becomes likely in the middle layers before the **Hindi** answer appears in the last layers, the model goes through an English-like "concept space" (Wendler et al., 2024).

_Result to be filled._

## What changed compared with the original project

The original report compared Mistral-7B (BLEU 5.88) with LLaMA-3.1-8B (BLEU 15.12). Re-reading the code showed that these numbers cannot be compared or trusted:

1. **Test-set leakage** — Mistral was fine-tuned on the first 503 sentences of the WMT21 test set, then evaluated on the whole test set. Here the two halves are kept apart: 503 sentences for training / few-shot examples, 509 for testing only.
2. **Unfair comparison** — Mistral was evaluated fine-tuned, zero-shot, with greedy decoding; LLaMA without fine-tuning, 2-shot, with random sampling (`temperature=0.8`), which also makes its score non-reproducible. Here one model, one prompt format and greedy decoding are used for every system.
3. **No "before / after"** — the effect of fine-tuning was never measured on the same model. Here zero-shot, few-shot and fine-tuned runs share the same model and test set.
4. **Fine-tuning bug** — the LLaMA training code padded with token id `50256` (GPT-2's end-of-text), an ordinary token in LLaMA's vocabulary, appended to every answer. Here padding uses the model's pad token, the loss covers only the Hindi answer, and answers end with the real end-of-turn token (checked by a unit test).
5. **Metric reading** — a BERTScore of 0.83 does not mean "83 % of the text is relevant" (raw BERTScore is always high). chrF++ is reported instead.

Model change: LLaMA-3.1 requires accepting Meta's licence; **Qwen2.5-7B-Instruct** (Apache 2.0, strong multilingual model) is used instead so that anyone can re-run the notebook.

## Quick start

Easiest: open [`colab_run.ipynb`](colab_run.ipynb) in Google Colab with a T4 GPU and run all cells.

```bash
git clone https://github.com/AWARKEH1/bn-hi-llm-translation.git
cd bn-hi-llm-translation
pip install -r requirements.txt          # needs a CUDA GPU with ~16 GB

python -m bnhi.translate --mode zero_shot
python -m bnhi.translate --mode few_shot --shots 5
python -m bnhi.finetune --epochs 2
python -m bnhi.translate --mode finetuned
python -m bnhi.logit_lens
pytest
```

## Project structure

```
bnhi/
├── data.py        # leakage-free train / test split, few-shot examples
├── prompts.py     # one chat prompt for every system
├── model.py       # Qwen2.5-7B-Instruct in 4-bit (bitsandbytes)
├── translate.py   # batched greedy translation + scoring
├── finetune.py    # QLoRA training (answer-only loss, correct padding)
├── logit_lens.py  # layer-by-layer English vs Hindi probabilities
└── metrics.py     # BLEU, chrF++, Devanagari share
data/              # train.{bn,hi} (503) and test.{bn,hi} (509)
results/           # metrics.json, translations, figures
```

## Limitations

- 503 training sentences is very little for fine-tuning; results show what is achievable on a free GPU, not the state of the art.
- Automatic metrics only; no human evaluation.
- Logit lens uses the first token of each word; Qwen splits Devanagari into small pieces, so the Hindi curve is an approximation.

## Credits

Project of the **MSc NLP, IDMC – Université de Lorraine** (2024–2025), by
G M Arafat Rahman, Abbas Awarkeh and Mohammad Al Takach.

- Original repository: [arafat04/bn-hi-MT-improvement-using-llm](https://github.com/arafat04/bn-hi-MT-improvement-using-llm)
- This repository is my reworked version: leakage-free evaluation, fair comparison, fixed fine-tuning, logit-lens re-implementation, tests and documentation.

Data: WMT21 Bengali–Hindi test set, built on FLORES-101 (Goyal et al., 2022), CC BY-SA 4.0. Logit lens: Wendler, Veselovsky, Monea & West (2024), *Do Llamas Work in English? On the Latent Language of Multilingual Transformers*. Model: `Qwen/Qwen2.5-7B-Instruct`.
