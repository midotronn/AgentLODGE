#!/usr/bin/env python3
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    chat = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=5,
    )
    print("chat:", chat.choices[0].message.content)
except Exception as exc:
    print("chat err:", exc)

for model in (os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"), "dall-e-3"):
    try:
        kwargs = {"model": model, "prompt": "red ballgown sketch", "size": "1024x1024", "n": 1}
        if model.startswith("dall-e"):
            kwargs["quality"] = "standard"
        img = client.images.generate(**kwargs)
        item = img.data[0]
        print(f"image {model}:", bool(item.b64_json or item.url))
    except Exception as exc:
        print(f"image {model} err:", exc)
