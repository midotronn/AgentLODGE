"""Costume image generation via OpenAI or Gemini."""

from __future__ import annotations

import base64
from pathlib import Path

from agentlodge.config import Settings


def build_costume_prompt(costume_description: str) -> str:
    return (
        "Full-body fashion illustration of a dancer wearing: "
        f"{costume_description}. Studio lighting, clean white background, high detail."
    )


def generate_costume_image(
    costume_description: str,
    output_path: Path,
    settings: Settings,
) -> None:
    settings.validate_image_backend()
    prompt = build_costume_prompt(costume_description)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.image_backend == "openai":
        _generate_openai(prompt, output_path, settings.openai_api_key)
    else:
        _generate_gemini(prompt, output_path, settings.gemini_api_key)


def _generate_openai(prompt: str, output_path: Path, api_key: str | None) -> None:
    import os
    import urllib.request

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    kwargs: dict = {"model": model, "prompt": prompt, "size": "1024x1024", "n": 1}
    if model.startswith("dall-e"):
        kwargs["quality"] = "standard"

    response = client.images.generate(**kwargs)
    item = response.data[0]
    if item.b64_json:
        output_path.write_bytes(base64.b64decode(item.b64_json))
        return
    if item.url:
        urllib.request.urlretrieve(item.url, str(output_path))
        return
    raise RuntimeError("OpenAI image generation returned no image data")


def _generate_gemini(prompt: str, output_path: Path, api_key: str | None) -> None:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
    response = model.generate_content(
        prompt,
        generation_config={"response_modalities": ["TEXT", "IMAGE"]},
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data)
            output_path.write_bytes(data)
            return

    raise RuntimeError("Gemini image generation returned no image data")
