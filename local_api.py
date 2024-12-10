from fastapi import FastAPI, Query
from typing import Annotated
import himitsu

app = FastAPI()

models = [
    "sbintuitions/sarashina2-7b",
    "llm-jp/llm-jp-3-1.8b",
    "leia-llm/Leia-Swallow-7b",
    "rinna/youri-7b",
    "augmxnt/shisa-gamma-7b-v1"
]


@app.get("/encode")
async def encode(secret: Annotated[str, Query(description="秘密文。ビット列である必要があります")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01, device=Query("cpu", enum=["cpu", "cuda:0", "mps"], description="使用するデバイス"),
                 language=Query("ja", enum=["en", "ja"], description="文章の言語"), model_name=Query("sbintuitions/sarashina2-7b", enum=models, description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(language, device, model_name=model_name)
    tokenizer, special_tokens = himitsu.load_tokenizer(
        language, tokenizer_name=model_name)
    encoded = himitsu.encode(
        model=model,
        tokenizer=tokenizer,
        secret=secret,
        prompt=prompt,
        min_prob=float(min_prob),
        special_tokens=special_tokens,
    )
    return encoded


@app.get("/decode")
async def decode(cover_text: Annotated[str, Query(description="デコードする文章")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01,  device=Query("cpu", enum=["cpu", "cuda:0", "mps"], description="使用するデバイス"),
                 language=Query("ja", enum=["en", "ja"], description="文章の言語"), model_name=Query("sbintuitions/sarashina2-7b", enum=["sbintuitions/sarashina2-7b"], description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(language, device, model_name=model_name)
    tokenizer, special_tokens = himitsu.load_tokenizer(
        language, tokenizer_name=model_name)

    decoded = himitsu.decode(
        model=model,
        tokenizer=tokenizer,
        cover_text=cover_text,
        prompt=prompt,
        min_prob=float(min_prob),
        special_tokens=special_tokens,
    )
    return decoded
