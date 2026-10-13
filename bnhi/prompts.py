"""Chat prompts used for prompting, fine-tuning and evaluation (one format everywhere)."""

from __future__ import annotations

SYSTEM = ("You are a professional translator. Translate the user's Bengali text into Hindi "
          "(Devanagari script). Answer with the Hindi translation only.")


def build_messages(source: str, examples: list[tuple[str, str]] | None = None) -> list[dict]:
    """System prompt + optional few-shot turns + the sentence to translate."""
    messages = [{"role": "system", "content": SYSTEM}]
    for bn, hi in examples or []:
        messages.append({"role": "user", "content": bn})
        messages.append({"role": "assistant", "content": hi})
    messages.append({"role": "user", "content": source})
    return messages


def clean_output(text: str) -> str:
    """Keep the first non-empty line (models sometimes add explanations)."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""
