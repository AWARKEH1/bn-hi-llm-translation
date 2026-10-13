"""Tests that run without a GPU or model download."""

from bnhi.data import few_shot_examples, load_split, train_val_split
from bnhi.finetune import IGNORE, build_example, collate
from bnhi.logit_lens import WORDS, build_prompt
from bnhi.metrics import devanagari_ratio
from bnhi.prompts import build_messages, clean_output


def test_splits_are_disjoint_and_aligned():
    train, test = load_split("train"), load_split("test")
    assert len(train) == 503 and len(test) == 509
    assert not {bn for bn, _ in train} & {bn for bn, _ in test}


def test_few_shot_examples_come_from_train_only():
    train, test = load_split("train"), load_split("test")
    examples = few_shot_examples(train, k=5)
    assert len(examples) == 5 and all(e in train for e in examples)
    assert not set(examples) & set(test)


def test_train_val_split():
    tr, val = train_val_split(load_split("train"), n_val=50)
    assert len(val) == 50 and len(tr) == 453 and not set(tr) & set(val)


def test_messages_and_cleaning():
    msgs = build_messages("আমি", [("বই", "किताब")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert clean_output("\n  नमस्ते दुनिया \nNote: translated") == "नमस्ते दुनिया"


def test_devanagari_ratio():
    assert devanagari_ratio("नमस्ते") == 1.0
    assert devanagari_ratio("namaste") == 0.0


class FakeTokenizer:
    """Character-level stand-in for a chat tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_loss_only_on_the_answer():
    tok = FakeTokenizer()
    ex = build_example(tok, "বই", "किताब")
    answer = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE]
    assert "".join(map(chr, answer)) == "किताब</end>"   # answer + end-of-turn, nothing else


def test_collate_pads_with_ignored_labels():
    batch = collate([{"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]},
                     {"input_ids": [4], "labels": [4]}], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_logit_lens_prompt_hides_the_answer():
    for i, (bn, hi, _) in enumerate(WORDS):
        prompt = build_prompt(i)
        assert prompt.endswith(f'"{bn}" - हिन्दी: "') and hi not in prompt.splitlines()[-1]
