from typing import List, Tuple

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer


def load_model(language: str, device: str = "cpu", model_name: str = "sbintuitions/sarashina2-7b") -> transformers.PreTrainedModel:
    if language == "en":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif language == "ja":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Invalid language: {language}")
    model.to(device)
    model.eval()
    return model


def load_tokenizer(language: str, special_tokens: list = ["<unk>", "<s>", "</s>", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "▁", "<0x0A>"],
                   tokenizer_name: str = "sbintuitions/sarashina2-7b") -> Tuple[transformers.PreTrainedTokenizer, List[str]]:
    if language == "en":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif language == "ja":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Invalid language: {language}")

    return tokenizer, special_tokens
