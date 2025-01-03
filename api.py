import modal
from fastapi import FastAPI, Query
from typing import Annotated
import himitsu
from starlette.middleware.cors import CORSMiddleware

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

models = [
    "llm-jp/llm-jp-3-1.8b",
    "leia-llm/Leia-Swallow-7b",
    "rinna/youri-7b",
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
        "accelerate>=1.2.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)
)

app = modal.App("himitsu", image=image)


@web_app.get("/encode")
async def encode(secret: Annotated[str, Query(description="秘密文。ビット列である必要があります")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01,
                 model_name=Query("leia-llm/Leia-Swallow-7b", enum=models, description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(model_name=model_name)
    tokenizer, special_tokens = himitsu.load_tokenizer(
        tokenizer_name=model_name)
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
async def decode(cover_text: Annotated[str, Query(description="デコードする文章")], prompt: Annotated[str, Query(description="生成プロンプト")], min_prob: float = 0.01,
                 model_name=Query("leia-llm/Leia-Swallow-7b", enum=models, description="言語モデル（Hugging Face）")):
    model = himitsu.load_model(model_name=model_name)
    tokenizer, special_tokens = himitsu.load_tokenizer(
        tokenizer_name=model_name)

    decoded = himitsu.decode(
        model=model,
        tokenizer=tokenizer,
        cover_text=cover_text,
        prompt=prompt,
        min_prob=float(min_prob),
        special_tokens=special_tokens,
    )
    return decoded[:(len(decoded) // 8) * 8]


@app.function(gpu="A100-80GB:2", timeout=10000)
@modal.asgi_app()
def fastapi_app():
    return web_app
