"""Structure-aware ("story") dance assembly.

Generalizes the hybrid assembler: instead of segmenting on raw downbeats and optimizing only local
coherence, it assembles one dance over the storyboard's musical **sections**, choosing per section
which material best realizes the plan (preferred generator + target energy) while staying smooth,
optionally reusing a recurring motif (retimed / mirrored), and joining source changes with the same
training-free inertialized transition used by the hybrid.

Two stages are kept separate so the decision logic is testable without the heavy rotation backend:
  * :func:`select_sources` -- pure-numpy per-section material selection (no torch),
  * :func:`assemble_story` -- concatenation + inertialized blending at source changes (uses torch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agentlodge.audio.structure import MusicStructure, Section
from agentlodge.agent.storyboard import Storyboard, SectionPlan
from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
from agentlodge.dance.transition import amplitude_scale, blend_onto, mirror, retime, to_zup

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentlodge.audio.preprocess import SongMetadata

logger = logging.getLogger(__name__)

_KIN = 135  # translation(3) + rotation(132); excludes the 4 contact labels
DEFAULT_EXPRESSIVENESS = 4.0

# Per-section selection weights. The storyboard encodes the musical intent (form alignment, energy
# arc, sectional contrast), so plan adherence (arc + generator bias) drives selection; local
# coherence is a lighter tie-breaker (seam smoothness is separately handled by inertialization).
_W_COH = 0.4     # weight on the (bounded, relative) local-coherence penalty
_W_ARC = 0.6     # penalty for missing the plan's target energy
_W_BIAS = 0.5    # bonus for matching the plan's preferred generator
_W_REUSE = 0.3   # bonus for reusing a motif the plan asked for


@dataclass
class StoryResult:
    motion: np.ndarray                                   # (L, 139) Z-up assembled dance
    schedule: list = field(default_factory=list)         # [(a, b, source, role), ...]
    storyboard: Storyboard | None = None
    structure: MusicStructure | None = None
    reasoning: str = ""
    section_scores: list = field(default_factory=list)   # per-section chosen source + costs

    def schedule_summary(self) -> str:
        from agentlodge.config import FPS
        return ", ".join(f"{a / FPS:.1f}-{b / FPS:.1f}s:{src}[{role}]"
                         for a, b, src, role in self.schedule)


# --------------------------------------------------------------------------- scoring helpers
def _energy_mean(clip: np.ndarray) -> float:
    if clip.shape[0] < 2:
        return 0.0
    rv = np.linalg.norm(np.diff(clip[:, :3], axis=0, prepend=clip[:1, :3]), axis=1)
    jm = np.linalg.norm(np.diff(clip[:, :_KIN], axis=0, prepend=clip[:1, :_KIN]), axis=1)
    return float(np.mean(0.6 * rv + 0.4 * jm))


def _coh_badness(clip: np.ndarray) -> float:
    """Standalone local coherence penalty (jerk + foot skating); lower = smoother."""
    if clip.shape[0] < 6:
        return 0.0
    kin = clip[:, :_KIN]
    jerk = float(np.mean(np.linalg.norm(np.diff(kin, n=3, axis=0), axis=1)))
    contact = clip[:, _KIN:139].mean(axis=1)[1:]
    horiz = np.linalg.norm(np.diff(clip[:, [0, 1]], axis=0), axis=1)
    foot = float(np.mean(horiz * contact))
    return jerk + foot


def _coh_penalty(values: dict) -> dict:
    """Bounded RELATIVE coherence penalty in [0, 1].

    ``0`` for the smoothest candidate; others penalized by their fractional excess badness over
    the best, clipped at 1.0. Unlike min-max normalization this reflects the *actual* magnitude of
    the difference, so near-equal candidates get near-equal penalties (letting the storyboard's
    arc/bias decide) while a genuinely much rougher candidate is still penalized.
    """
    vals = {k: float(v) for k, v in values.items()}
    bmin = min(vals.values())
    return {k: float(np.clip((v - bmin) / (bmin + 1e-6), 0.0, 1.0)) for k, v in vals.items()}


def _bias_bonus(source: str, plan: SectionPlan) -> float:
    bonus = 0.0
    gen = "reuse" if source.startswith("reuse") else source
    if plan.generator_bias in {"lodge", "edge"} and gen == plan.generator_bias:
        bonus -= _W_BIAS
    if source.startswith("reuse") and plan.reuse_of is not None:
        bonus -= _W_REUSE
    return bonus


# --------------------------------------------------------------------------- section selection
def _clip_sections(structure: MusicStructure, n: int) -> list[Section]:
    """Clip/trim sections to the available frame count ``n`` (drop empties)."""
    from agentlodge.config import FPS

    out: list[Section] = []
    for s in structure.sections:
        a, b = int(s.start_frame), min(int(s.end_frame), n)
        if b - a < 2:
            continue
        out.append(Section(start_frame=a, end_frame=b, start_sec=a / FPS, end_sec=b / FPS,
                           label=s.label, role=s.role, energy=s.energy))
    if out:
        last = out[-1]
        out[-1] = Section(last.start_frame, n, last.start_frame / FPS, n / FPS,
                          last.label, last.role, last.energy)
    return out


def select_sources(lodge_z: np.ndarray, edge_z: np.ndarray, structure: MusicStructure,
                   storyboard: Storyboard, *, motif_reuse: bool = True,
                   energy_shaping: bool = False) -> list[dict]:
    """Choose, per section, the material realizing the storyboard (pure numpy, no blending).

    Returns an ordered list of dicts: ``{a, b, source, role, clip, costs}``. ``source`` is
    ``"lodge"``, ``"edge"`` or ``"reuse:<i>"``. ``clip`` is the raw (pre-blend) chosen slice.
    """
    n = min(lodge_z.shape[0], edge_z.shape[0])
    sections = _clip_sections(structure, n)
    plans_by_idx = {p.section_index: p for p in storyboard.plans}
    chosen_raw: dict[int, np.ndarray] = {}   # section_index -> selected raw clip (for reuse)
    decisions: list[dict] = []

    for i, sec in enumerate(sections):
        a, b = sec.start_frame, sec.end_frame
        plan = plans_by_idx.get(i, SectionPlan(section_index=i, role=sec.role,
                                               target_intensity=sec.energy,
                                               vocabulary="", generator_bias="auto"))
        cands: dict[str, np.ndarray] = {"lodge": lodge_z[a:b], "edge": edge_z[a:b]}

        if (motif_reuse and plan.reuse_of is not None
                and plan.reuse_of in chosen_raw):
            reuse_clip = retime(chosen_raw[plan.reuse_of], b - a)
            if plan.variation.get("mirror"):
                reuse_clip = mirror(reuse_clip)
            if energy_shaping and abs(float(plan.variation.get("amplitude", 1.0)) - 1.0) > 1e-3:
                reuse_clip = amplitude_scale(reuse_clip, float(plan.variation["amplitude"]))
            cands[f"reuse:{plan.reuse_of}"] = reuse_clip

        # energy match: normalize candidate energies to [0,1] within the section.
        energies = {k: _energy_mean(v) for k, v in cands.items()}
        emin, emax = min(energies.values()), max(energies.values())
        erel = {k: (0.0 if emax <= emin else (v - emin) / (emax - emin))
                for k, v in energies.items()}
        coh_pen = _coh_penalty({k: _coh_badness(v) for k, v in cands.items()})

        costs = {}
        for k in cands:
            arc_pen = abs(erel[k] - float(plan.target_intensity))
            costs[k] = _W_COH * coh_pen[k] + _W_ARC * arc_pen + _bias_bonus(k, plan)
        source = min(costs, key=costs.get)

        gen = "reuse" if source.startswith("reuse") else source
        matched_bias = plan.generator_bias in {"lodge", "edge"} and gen == plan.generator_bias
        chosen_raw[i] = cands[source]
        decisions.append({
            "a": a, "b": b, "source": source, "role": sec.role,
            "clip": cands[source],
            "costs": {k: round(v, 4) for k, v in costs.items()},
            "target_intensity": float(plan.target_intensity),
            "plan_bias": plan.generator_bias,
            "matched_bias": bool(matched_bias),
            "vocabulary": plan.vocabulary,
            "energies": {k: round(float(energies[k]), 4) for k in cands},
            "chosen_cost": round(float(costs[source]), 4),
        })
    return decisions


# --------------------------------------------------------------------------- assembly
def _continuous(prev: dict | None, cur: dict) -> bool:
    """True if ``cur`` continues ``prev`` with no discontinuity (same generator, contiguous)."""
    if prev is None:
        return False
    if prev["source"].startswith("reuse") or cur["source"].startswith("reuse"):
        return False
    return prev["source"] == cur["source"] and prev["b"] == cur["a"]


def assemble_story(decisions: list[dict], *, blend_frames: int = 15) -> np.ndarray:
    """Concatenate chosen section clips, inertially blending at each source discontinuity."""
    committed: np.ndarray | None = None
    prev: dict | None = None
    for cur in decisions:
        seg = cur["clip"]
        if seg.shape[0] == 0:
            continue
        if committed is None:
            committed = seg.copy()
        elif _continuous(prev, cur):
            committed = np.concatenate([committed, seg], axis=0)
        else:
            blended = blend_onto(committed[-2:], seg, blend_frames,
                                 canonical_yaw=None, align_facing=False)
            committed = np.concatenate([committed, blended], axis=0)
        prev = cur
    return committed if committed is not None else np.zeros((0, 139), dtype=np.float32)


def build_story_dance(lodge_motion: np.ndarray, edge_motion: np.ndarray,
                      structure: MusicStructure, storyboard: Storyboard,
                      metadata: "SongMetadata", *, blend_frames: int = 15,
                      motif_reuse: bool = True, energy_shaping: bool = False) -> StoryResult:
    """Assemble a structure-aware dance from independent LODGE and EDGE motions + a storyboard."""
    lodge = to_zup(to_agentlodge139(ensure_lodge139(lodge_motion)))  # native -> AgentLODGE, Y->Z up
    edge = to_agentlodge139(ensure_lodge139(edge_motion))            # EDGE already Z-up
    n = min(lodge.shape[0], edge.shape[0], structure.total_frames)
    if n < 30:
        raise ValueError(f"Motions too short for story assembly ({n} frames)")
    lodge, edge = lodge[:n], edge[:n]

    decisions = select_sources(lodge, edge, structure, storyboard,
                               motif_reuse=motif_reuse, energy_shaping=energy_shaping)
    if not decisions:
        raise ValueError("no usable sections for story assembly")

    motion = assemble_story(decisions, blend_frames=blend_frames)
    schedule = [(d["a"], d["b"], d["source"], d["role"]) for d in decisions]
    _score_keys = ("a", "b", "source", "role", "costs", "target_intensity",
                   "plan_bias", "matched_bias", "vocabulary", "energies", "chosen_cost")
    section_scores = [{k: d[k] for k in _score_keys} for d in decisions]
    n_reuse = sum(1 for d in decisions if d["source"].startswith("reuse"))
    n_lodge = sum(1 for d in decisions if d["source"] == "lodge")
    n_edge = sum(1 for d in decisions if d["source"] == "edge")
    n_honored = sum(1 for d in decisions if d["matched_bias"])
    n_biased = sum(1 for d in decisions if d["plan_bias"] in {"lodge", "edge"})

    from agentlodge.config import FPS
    arc = storyboard.arc if storyboard is not None else "?"
    logger.info("Story: realizing %d-section plan (arc: %s)", len(decisions), arc)
    for d in decisions:
        a, b = d["a"], d["b"]
        mark = "=bias" if d["matched_bias"] else ("~auto" if d["plan_bias"] == "auto" else "!=bias")
        cost_str = " ".join(f"{k}:{v:.3f}" for k, v in d["costs"].items())
        logger.info(
            "  %5.1f-%5.1fs %-8s -> %-8s (%s) plan[bias=%-5s tgtE=%.2f] chose_cost=%.3f | costs %s",
            a / FPS, b / FPS, d["role"], d["source"], mark,
            d["plan_bias"], d["target_intensity"], d["chosen_cost"], cost_str,
        )

    schedule_summary = ", ".join(
        f"{a / FPS:.1f}-{b / FPS:.1f}s:{src}[{role}]" for a, b, src, role in schedule
    )
    reasoning = (
        f"story assembly: {len(decisions)} sections "
        f"({n_lodge} LODGE, {n_edge} EDGE, {n_reuse} motif-reuse); "
        f"storyboard bias honored in {n_honored}/{n_biased} explicitly-biased sections; "
        f"schedule: {schedule_summary}"
    )
    logger.info("Story schedule: %s", schedule_summary)
    return StoryResult(motion=motion, schedule=schedule, storyboard=storyboard,
                       structure=structure, reasoning=reasoning, section_scores=section_scores)
