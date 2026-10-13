"""QLoRA fine-tuning of Qwen2.5-7B-Instruct on the 503 TRAIN sentence pairs.

Fixes compared with the original LLaMA fine-tuning code:
* padding uses the model's own pad token (the original used 50256, GPT-2's id,
  which is an ordinary token for LLaMA and was appended to every answer);
* the loss is computed on the Hindi answer only (prompt and padding masked);
* the answer ends with the model's real end-of-turn token, so it learns to stop.

Usage:
    python -m bnhi.finetune --epochs 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import load_split, train_val_split
from .prompts import build_messages

RESULTS = Path(__file__).resolve().parents[1] / "results"
IGNORE = -100


def build_example(tokenizer, source: str, target: str, max_len: int = 512) -> dict:
    """Token ids of prompt + answer, with labels only on the answer."""
    messages = build_messages(source)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target}],
                                         tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the prompt is not a prefix of the full conversation")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a batch; padding is ignored by the loss and the attention."""
    import torch

    width = max(len(x["input_ids"]) for x in batch)
    ids, labels, mask = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * (width - n))
        labels.append(x["labels"] + [IGNORE] * (width - n))
        mask.append([1] * n + [0] * (width - n))
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from .model import load

    torch.manual_seed(args.seed)
    model, tokenizer = load(for_training=True)
    # gradient checkpointing keeps activations small enough for a 16 GB T4
    # (prepare_model_for_kbit_training is skipped: it would upcast the 2 x 545M
    # embedding / output parameters to fp32 and run out of memory)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train_pairs, val_pairs = train_val_split(load_split("train"))
    train = [build_example(tokenizer, bn, hi) for bn, hi in train_pairs]
    val = [build_example(tokenizer, bn, hi) for bn, hi in val_pairs]
    pad = tokenizer.pad_token_id
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, pad))
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=lambda b: collate(b, pad))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = [model(**{k: v.to(model.device) for k, v in b.items()}).loss.item() for b in val_loader]
        model.train()
        return sum(losses) / len(losses)

    history = [{"epoch": 0, "train_loss": None, "val_loss": val_loss()}]
    print(f"epoch 0  val loss {history[0]['val_loss']:.4f}")
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, n = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            (loss / args.grad_accum).backward()
            running, n = running + loss.item(), n + 1
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        history.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss()})
        print(f"epoch {epoch}  train loss {running / n:.4f}  val loss {history[-1]['val_loss']:.4f}")

    RESULTS.mkdir(exist_ok=True)
    model.save_pretrained(RESULTS / "lora_adapter")
    (RESULTS / "finetune_history.json").write_text(json.dumps(history, indent=2))

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs[1:], [h["train_loss"] for h in history[1:]], "o-", label="train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], "o-", label="validation loss")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy (Hindi tokens only)")
    plt.title("QLoRA fine-tuning - Qwen2.5-7B-Instruct")
    plt.legend(); plt.tight_layout()
    (RESULTS / "figures").mkdir(exist_ok=True)
    plt.savefig(RESULTS / "figures" / "finetune_loss.png", dpi=150)


if __name__ == "__main__":
    main()
