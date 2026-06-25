"""LLM-based dance selection agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agentlodge.audio.preprocess import SongMetadata
from agentlodge.dance.metrics import DanceMetrics


@dataclass
class SelectionResult:
    selected_model: str
    reasoning: str
    used_fallback: bool = False


def _build_prompt(
    lodge_metrics: DanceMetrics,
    edge_metrics: DanceMetrics,
    metadata: SongMetadata,
) -> str:
    return f"""You are selecting the better dance generation for a music-driven choreography task.

Song metadata:
- duration_seconds: {metadata.duration_seconds:.2f}
- estimated_bpm: {metadata.bpm:.1f}

LODGE output:
- beat_alignment_score: {lodge_metrics.beat_alignment_score:.4f}
- motion_diversity: {lodge_metrics.motion_diversity:.4f}
- summary: {lodge_metrics.summary}

EDGE output:
- beat_alignment_score: {edge_metrics.beat_alignment_score:.4f}
- motion_diversity: {edge_metrics.motion_diversity:.4f}
- summary: {edge_metrics.summary}

Choose the better dance considering beat alignment with the music, motion diversity, and suitability for the song duration/tempo.
Respond with JSON only:
{{"selected_model": "lodge" or "edge", "reasoning": "brief explanation"}}
"""


def _parse_response(text: str) -> SelectionResult:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Agent response did not contain JSON")
    payload = json.loads(match.group())
    selected = payload["selected_model"].lower().strip()
    if selected not in {"lodge", "edge"}:
        raise ValueError(f"Invalid selected_model: {selected}")
    return SelectionResult(
        selected_model=selected,
        reasoning=str(payload.get("reasoning", "")).strip(),
    )


def select_dance(
    lodge_metrics: DanceMetrics,
    edge_metrics: DanceMetrics,
    metadata: SongMetadata,
    api_key: str | None,
) -> SelectionResult:
    if not api_key:
        return SelectionResult(
            selected_model="lodge",
            reasoning="No OPENAI_API_KEY configured; defaulting to LODGE.",
            used_fallback=True,
        )

    try:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": _build_prompt(lodge_metrics, edge_metrics, metadata),
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_response(text)
    except Exception as exc:
        return SelectionResult(
            selected_model="lodge",
            reasoning=f"Selection agent failed ({exc}); defaulting to LODGE.",
            used_fallback=True,
        )
