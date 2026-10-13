"""Translate the TEST split and score it.

Usage:
    python -m bnhi.translate --mode zero_shot
    python -m bnhi.translate --mode few_shot --shots 5
    python -m bnhi.translate --mode finetuned --adapter results/lora_adapter

Decoding is greedy (deterministic), so every run gives the same scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import few_shot_examples, load_split
from .metrics import score
from .prompts import build_messages, clean_output

RESULTS = Path(__file__).resolve().parents[1] / "results"


def translate_batch(model, tokenizer, sources, examples=None, batch_size=16, max_new_tokens=256):
    import torch

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    outputs = []
    for i in range(0, len(sources), batch_size):
        prompts = [tokenizer.apply_chat_template(build_messages(s, examples), tokenize=False,
                                                 add_generation_prompt=True)
                   for s in sources[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs += [clean_output(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        print(f"  {min(i + batch_size, len(sources))}/{len(sources)} sentences", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "finetuned"], required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--adapter", default=str(RESULTS / "lora_adapter"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N test sentences")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from .model import load

    test = load_split("test")[: args.limit]
    sources, references = [bn for bn, _ in test], [hi for _, hi in test]
    examples = few_shot_examples(load_split("train"), k=args.shots) if args.mode == "few_shot" else None
    model, tokenizer = load(adapter=args.adapter if args.mode == "finetuned" else None)

    name = f"{args.mode}_{args.shots}" if args.mode == "few_shot" else args.mode
    print(f"Translating {len(sources)} test sentences ({name})...")
    hypotheses = translate_batch(model, tokenizer, sources, examples, batch_size=args.batch_size)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"translations_{name}.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    metrics = {"n_sentences": len(sources), **score(hypotheses, references)}
    all_path = RESULTS / "metrics.json"
    all_metrics = json.loads(all_path.read_text()) if all_path.exists() else {}
    all_metrics[name] = metrics
    all_path.write_text(json.dumps(all_metrics, indent=2))
    print(json.dumps({name: metrics}, indent=2))


if __name__ == "__main__":
    main()
