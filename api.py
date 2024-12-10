import modal
from fastapi import FastAPI, Query
from typing import Annotated
import himitsu

web_app = FastAPI()

models = [
    "sbintuitions/sarashina2-7b",
    "llm-jp/llm-jp-3-1.8b",
    "leia-llm/Leia-Swallow-7b",
    "rinna/youri-7b",
    "augmxnt/shisa-gamma-7b-v1"
]


def download_model():
    from huggingface_hub import snapshot_download
    for model in models:
        snapshot_download(model)


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "fastapi[standard]>=0.115.5",
        "hf-transfer>=0.1.8",
        "huggingface-hub>=0.26.2",
        "protobuf>=5.28.3",
        "sentencepiece>=0.2.0",
        "torch>=2.5.1",
        "transformers>=4.46.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

app = modal.App("himitsu", image=image)


@web_app.get("/encode")
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


@web_app.get("/decode")
async def decode(cover_text: Annotated[str, Query(description="デコードする文章")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01,  device=Query("cpu", enum=["cpu", "cuda:0", "mps"], description="使用するデバイス"),
                 language=Query("ja", enum=["en", "ja"], description="文章の言語"), model_name=Query("sbintuitions/sarashina2-7b", enum=models, description="言語モデル（Hugging Face）")):
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


@app.function(gpu="A100-80GB", timeout=10000)
@modal.asgi_app()
def fastapi_app():
    return web_app
