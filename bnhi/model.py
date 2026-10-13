"""Load Qwen2.5-7B-Instruct in 4-bit (fits a free 16 GB Colab T4)."""

from __future__ import annotations

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def load(model_name: str = MODEL_NAME, adapter: str | None = None, for_training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right" if for_training else "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                 device_map="auto", torch_dtype=torch.float16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if not for_training:
        model.eval()
    return model, tokenizer
