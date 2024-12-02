from typing import List, Tuple

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer


def load_model(language: str, device: str = "cpu", model_name: str = "llm-jp/llm-jp-3-1.8b") -> transformers.PreTrainedModel:
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
                   tokenizer_name: str = "llm-jp/llm-jp-3-1.8b") -> Tuple[transformers.PreTrainedTokenizer, bool, List[str]]:
    if language == "en":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif language == "ja":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # tokenizer.do_lower_case = True
    else:
        raise ValueError(f"Invalid language: {language}")

    # gpt2-medium and rugpt3medium_based_on_gpt2 use byte-level vocab and need to be handled slight
    # differently when encoding/decoding
    byte_level_vocab = language in ["en"]

    return tokenizer, byte_level_vocab, special_tokens
