"""LLM costume-describer agent driven by audio features.

Turns an :class:`~agentlodge.audio.preprocess.AudioDescriptor` (plus an optional
genre hint) into a concise costume description that is then handed to the image
generator. This is an LLM-only agent: it raises if no API key is available or if
the model response cannot be parsed, rather than falling back to a default.
"""

from __future__ import annotations

import json
import os
import re

from agentlodge.audio.preprocess import AudioDescriptor


def _build_prompt(descriptor: AudioDescriptor, genre: str | None) -> str:
    genre_line = f"- genre hint: {genre}\n" if genre else ""
    return f"""You are a costume designer for a music-driven dance performance.
Design a dance costume whose look and energy match the character of the song,
based only on the following audio analysis.

Audio analysis:
- duration_seconds: {descriptor.duration_seconds:.1f}
- tempo: {descriptor.bpm:.0f} BPM ({descriptor.tempo_feel})
- energy: {descriptor.energy_level} (rms={descriptor.rms_energy:.3f})
- timbre: {descriptor.brightness} (spectral centroid {descriptor.spectral_centroid_hz:.0f} Hz)
- rhythmic density: {descriptor.rhythmic_density} ({descriptor.onset_rate_per_second:.1f} onsets/s)
- key: {descriptor.key} {descriptor.mode}
- mood: {descriptor.mood}
{genre_line}
Translate this musical character into a concrete costume: describe fabrics,
silhouette, colors, and standout details in one vivid sentence. Do not mention
music, audio, tempo, or these analysis terms in the costume description itself.
Respond with JSON only:
{{"costume_description": "a single vivid sentence describing the costume"}}
"""


def _parse_response(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Costume agent response did not contain JSON")
    payload = json.loads(match.group(), strict=False)
    description = str(payload.get("costume_description", "")).strip()
    if not description:
        raise ValueError("Costume agent returned an empty description")
    return description


def describe_costume(
    descriptor: AudioDescriptor,
    api_key: str | None,
    *,
    genre: str | None = None,
) -> str:
    """Generate a costume description from audio features via OpenAI chat.

    Raises:
        ValueError: if ``api_key`` is missing or the response cannot be parsed.
    """
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required to generate a costume description from audio."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": _build_prompt(descriptor, genre),
            }
        ],
    )
    text = response.choices[0].message.content or ""
    return _parse_response(text)
