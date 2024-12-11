from typing import List, Tuple

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "leia-llm/Leia-Swallow-7b") -> transformers.PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # model.to(device)
    model.eval()
    return model


def load_tokenizer(language: str, special_tokens: list = ["<unk>", "<s>", "</s>", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "▁", "<0x0A>"],
                   tokenizer_name: str = "leia-llm/Leia-Swallow-7b") -> Tuple[transformers.PreTrainedTokenizer, List[str]]:
    if language == "en" or language == "ja":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Invalid language: {language}")

    return tokenizer, special_tokens
