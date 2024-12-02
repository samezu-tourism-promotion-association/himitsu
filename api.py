from fastapi import FastAPI, Query
import himitsu
from typing import Annotated

app = FastAPI()


@app.get("/encode")
async def encode(secret: Annotated[str, Query(description="秘密文。ビット列である必要があります")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01, device=Query("cpu", enum=["cpu", "cuda:0", "mps"], description="使用するデバイス"),
                 language=Query("ja", enum=["en", "ja"], description="文章の言語"), model_name=Query("llm-jp/llm-jp-3-1.8b", enum=["llm-jp/llm-jp-3-1.8b"], description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(language, device, model_name=model_name)
    tokenizer, byte_level_vocab, special_tokens = himitsu.load_tokenizer(
        language, tokenizer_name=model_name)
    encoded = himitsu.encode(
        model=model,
        tokenizer=tokenizer,
        secret=secret,
        prompt=prompt,
        min_prob=float(min_prob),
        special_tokens=special_tokens,
        byte_level_vocab=byte_level_vocab,
    )
    return encoded


@app.get("/decode")
async def decode(cover_text: Annotated[str, Query(description="デコードする文章")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01,  device=Query("cpu", enum=["cpu", "cuda:0", "mps"], description="使用するデバイス"),
                 language=Query("ja", enum=["en", "ja"], description="文章の言語"), model_name=Query("llm-jp/llm-jp-3-1.8b", enum=["llm-jp/llm-jp-3-1.8b"], description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(language, device, model_name=model_name)
    tokenizer, byte_level_vocab, special_tokens = himitsu.load_tokenizer(
        language, tokenizer_name=model_name)

    decoded = himitsu.decode(
        model=model,
        tokenizer=tokenizer,
        cover_text=cover_text,
        prompt=prompt,
        min_prob=float(min_prob),
        special_tokens=special_tokens,
        byte_level_vocab=byte_level_vocab,
    )
    return decoded
