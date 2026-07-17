"""LLM choreography storyboard agent.

Given a song's detected musical form (see ``agentlodge.audio.structure``), this agent authors a
high-level *choreographic plan* -- one :class:`SectionPlan` per musical section -- describing how
the dance should be composed across the whole song: an overall energy/narrative arc, a movement
vocabulary and preferred generator per section, and (optionally) which sections should reuse a
recurring motif. The structure-aware assembler (``agentlodge.dance.story``) then realizes this
plan training-free by arranging/retiming LODGE and EDGE material with inertialized transitions.

This mirrors ``agentlodge.agent.selector``: an OpenAI chat model produces a strict-JSON plan, and
a deterministic rule-based fallback is used when no API key is configured or the call fails -- so
the feature is fully functional offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agentlodge.audio.structure import MusicStructure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentlodge.audio.preprocess import AudioDescriptor, SongMetadata

logger = logging.getLogger(__name__)

VOCABULARY = (
    "grounded_minimal",     # low energy, small/still
    "sustained_lyrical",    # slow, flowing, expressive
    "flowing_smooth",       # mid energy, continuous
    "expansive_traveling",  # mid-high, covering space
    "percussive_sharp",     # high, accented/staccato
    "explosive_fast",       # peak, big dynamic movement
)
GENERATORS = ("lodge", "edge", "auto")


@dataclass
class SectionPlan:
    section_index: int
    role: str
    target_intensity: float          # [0, 1]
    vocabulary: str
    generator_bias: str              # lodge | edge | auto
    reuse_of: int | None = None      # earlier same-label section index, or None
    variation: dict = field(default_factory=lambda: {"mirror": False, "retime": 1.0, "amplitude": 1.0})

    def to_dict(self) -> dict:
        return {
            "section_index": self.section_index,
            "role": self.role,
            "target_intensity": round(self.target_intensity, 3),
            "vocabulary": self.vocabulary,
            "generator_bias": self.generator_bias,
            "reuse_of": self.reuse_of,
            "variation": self.variation,
        }


@dataclass
class Storyboard:
    arc: str
    plans: list[SectionPlan]
    reasoning: str = ""
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "arc": self.arc,
            "reasoning": self.reasoning,
            "used_fallback": self.used_fallback,
            "plans": [p.to_dict() for p in self.plans],
        }

    def describe(self) -> str:
        """Human-readable multi-line summary of the plan (for INFO logging / inspection)."""
        source = "rule-based fallback" if self.used_fallback else "LLM"
        lines = [
            f"arc      : {self.arc}",
            f"source   : {source}",
            f"reasoning: {self.reasoning or '(none)'}",
            "sections :",
        ]
        for p in self.plans:
            reuse = f" reuse<-{p.reuse_of}" if p.reuse_of is not None else ""
            var = ""
            if p.reuse_of is not None:
                v = p.variation
                flags = []
                if v.get("mirror"):
                    flags.append("mirror")
                if abs(float(v.get("retime", 1.0)) - 1.0) > 1e-3:
                    flags.append(f"retime={v['retime']}")
                if abs(float(v.get("amplitude", 1.0)) - 1.0) > 1e-3:
                    flags.append(f"amp={v['amplitude']}")
                if flags:
                    var = " [" + ",".join(flags) + "]"
            lines.append(
                f"  [{p.section_index}] {p.role:<8} intensity={p.target_intensity:.2f} "
                f"bias={p.generator_bias:<5} vocab={p.vocabulary}{reuse}{var}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- deterministic fallback
def _vocab_for_energy(e: float) -> str:
    if e >= 0.78:
        return "explosive_fast"
    if e >= 0.58:
        return "percussive_sharp"
    if e >= 0.42:
        return "expansive_traveling"
    if e >= 0.25:
        return "flowing_smooth"
    return "grounded_minimal"


def _first_same_label(structure: MusicStructure, i: int) -> int | None:
    """Earliest earlier section with the same repetition label as section ``i``."""
    label = structure.sections[i].label
    for j in range(i):
        if structure.sections[j].label == label:
            return j
    return None


def _rule_based_storyboard(structure: MusicStructure, *, motif_reuse: bool) -> Storyboard:
    """Derive a storyboard directly from the numeric structure (no LLM)."""
    plans: list[SectionPlan] = []
    sections = structure.sections
    for i, sec in enumerate(sections):
        e = float(sec.energy)
        bias = "edge" if e >= 0.55 else "lodge"
        reuse = _first_same_label(structure, i) if motif_reuse else None
        variation = {"mirror": False, "retime": 1.0, "amplitude": 1.0}
        if reuse is not None:
            src = sections[reuse]
            # even/odd recurrence -> mirror alternate occurrences for variety
            variation = {
                "mirror": (i - reuse) % 2 == 1,
                "retime": round(sec.n_frames / max(src.n_frames, 1), 4),
                "amplitude": round(float(np.clip((e + 1e-3) / (src.energy + 1e-3), 0.7, 1.4)), 3),
            }
        plans.append(SectionPlan(
            section_index=i, role=sec.role, target_intensity=e,
            vocabulary=_vocab_for_energy(e), generator_bias=bias,
            reuse_of=reuse, variation=variation,
        ))
    arc = _describe_arc(structure)
    return Storyboard(arc=arc, plans=plans,
                      reasoning="rule-based storyboard from detected structure", used_fallback=True)


def _describe_arc(structure: MusicStructure) -> str:
    roles = [s.role for s in structure.sections]
    peak = structure.climax_index
    return (f"{len(roles)} sections building toward the {roles[peak] if roles else 'peak'} "
            f"at section {peak}, then resolving: " + " -> ".join(roles))


# --------------------------------------------------------------------------- LLM path
def _build_prompt(structure: MusicStructure, metadata: "SongMetadata",
                  descriptor: "AudioDescriptor | None") -> str:
    lines = []
    for i, s in enumerate(structure.sections):
        same = _first_same_label(structure, i)
        lines.append(
            f"  [{i}] role~{s.role}  {s.start_sec:.1f}-{s.end_sec:.1f}s  "
            f"energy={s.energy:.2f}  repeat_label={s.label}"
            + (f"  (repeats section {same})" if same is not None else "")
        )
    desc = ""
    if descriptor is not None:
        desc = (f"\nAcoustic feel: tempo={getattr(descriptor, 'tempo_feel', '?')}, "
                f"energy={getattr(descriptor, 'energy_level', '?')}, "
                f"brightness={getattr(descriptor, 'brightness', '?')}, "
                f"key={getattr(descriptor, 'key', '?')} {getattr(descriptor, 'mode', '')}, "
                f"mood={getattr(descriptor, 'mood', '?')}.")
    return f"""You are a choreographer authoring a high-level STORYBOARD for a music-driven dance.
The dance is assembled from two generators: LODGE (smooth, flowing, graceful, sustained) and EDGE
(sharp, percussive, energetic). Design a coherent whole-song composition with an energy/narrative
arc (build -> climax -> resolution), sectional contrast, and optional recurring motifs.

Song: duration={getattr(metadata, 'duration_seconds', 0.0):.1f}s, bpm={getattr(metadata, 'bpm', 0.0):.0f},
climax at section {structure.climax_index}.{desc}

Sections (already detected from the audio; DO NOT invent new ones):
{chr(10).join(lines)}

For EACH section (same count and order), decide:
- target_intensity: float 0..1 following the overall arc (peak near section {structure.climax_index}).
- vocabulary: one of {list(VOCABULARY)}.
- generator_bias: "lodge" (flowing/graceful), "edge" (sharp/energetic), or "auto".
- reuse_of: an EARLIER section index with the SAME repeat_label to recur its motif, else null.
- variation: {{"mirror": bool, "retime": 1.0, "amplitude": 1.0}} (only meaningful when reuse_of set).

Respond with JSON ONLY, exactly:
{{"arc": "one-sentence description of the energy/narrative arc",
  "reasoning": "brief justification",
  "plans": [
    {{"section_index": 0, "role": "intro", "target_intensity": 0.1, "vocabulary": "grounded_minimal",
      "generator_bias": "lodge", "reuse_of": null,
      "variation": {{"mirror": false, "retime": 1.0, "amplitude": 1.0}}}}
  ]}}
"""


def _coerce_plan(raw: dict, idx: int, structure: MusicStructure) -> SectionPlan:
    role = str(raw.get("role") or structure.sections[idx].role)
    intensity = float(np.clip(float(raw.get("target_intensity", structure.sections[idx].energy)), 0.0, 1.0))
    vocab = str(raw.get("vocabulary", "")).strip()
    if vocab not in VOCABULARY:
        vocab = _vocab_for_energy(intensity)
    bias = str(raw.get("generator_bias", "auto")).lower().strip()
    if bias not in GENERATORS:
        bias = "auto"
    reuse = raw.get("reuse_of")
    reuse_of = None
    if isinstance(reuse, int) and 0 <= reuse < idx:
        # only accept reuse of an earlier section that shares the repeat label
        if structure.sections[reuse].label == structure.sections[idx].label:
            reuse_of = reuse
    var = raw.get("variation") or {}
    variation = {
        "mirror": bool(var.get("mirror", False)) if reuse_of is not None else False,
        "retime": float(var.get("retime", 1.0)) if reuse_of is not None else 1.0,
        "amplitude": float(np.clip(float(var.get("amplitude", 1.0)), 0.7, 1.4)) if reuse_of is not None else 1.0,
    }
    return SectionPlan(section_index=idx, role=role, target_intensity=intensity,
                       vocabulary=vocab, generator_bias=bias, reuse_of=reuse_of, variation=variation)


def _parse_response(text: str, structure: MusicStructure) -> Storyboard:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("storyboard response contained no JSON")
    payload = json.loads(match.group(), strict=False)
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list) or len(raw_plans) != len(structure.sections):
        raise ValueError(
            f"storyboard has {len(raw_plans) if isinstance(raw_plans, list) else 0} plans, "
            f"expected {len(structure.sections)}"
        )
    plans = [_coerce_plan(dict(raw_plans[i]), i, structure) for i in range(len(structure.sections))]
    return Storyboard(
        arc=str(payload.get("arc", "")).strip() or _describe_arc(structure),
        plans=plans,
        reasoning=str(payload.get("reasoning", "")).strip(),
        used_fallback=False,
    )


def author_storyboard(structure: MusicStructure, metadata: "SongMetadata",
                      descriptor: "AudioDescriptor | None", api_key: str | None,
                      *, motif_reuse: bool = True, chat_model: str | None = None) -> Storyboard:
    """Author a :class:`Storyboard` for ``structure`` (LLM if ``api_key`` else rule-based)."""
    if not structure.sections:
        return Storyboard(arc="(empty)", plans=[], reasoning="no sections", used_fallback=True)
    if not api_key:
        board = _rule_based_storyboard(structure, motif_reuse=motif_reuse)
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": _build_prompt(structure, metadata, descriptor)}],
            )
            text = response.choices[0].message.content or ""
            board = _parse_response(text, structure)
        except Exception as exc:  # noqa: BLE001 - robust fallback on any failure
            logger.warning("Storyboard agent failed (%s); using rule-based fallback", exc)
            board = _rule_based_storyboard(structure, motif_reuse=motif_reuse)
    logger.info("Storyboard authored (%s):\n%s",
                "rule-based fallback" if board.used_fallback else "LLM", board.describe())
    return board
