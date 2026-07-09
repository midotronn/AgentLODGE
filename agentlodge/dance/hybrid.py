"""Hybrid LODGE+EDGE dance assembly.

Builds ONE dance whose time-segments are taken from an independently generated LODGE dance and
EDGE dance, choosing per segment which generator best serves the whole piece, and joining the
chosen runs with training-free inertialized transitions (see ``transition.py``).

Pipeline:
  1. Unify frames: convert LODGE (Y-up) to the Z-up EDGE frame; trim both to a common length.
  2. Segment the timeline on musical downbeats (>= a minimum segment length).
  3. Per segment, pick the generator with the better local coherence (metric scheduler; an
     optional LLM scheduler can override). No switch penalty (per design).
  4. Merge consecutive same-generator segments into runs and concatenate them, applying an
     inertialized transition at every generator switch.

Output is a Z-up 139-dim motion the existing Y-Bot renderer consumes directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

from agentlodge.audio.preprocess import SongMetadata
from agentlodge.config import FPS
from agentlodge.dance.format import ensure_lodge139
from agentlodge.dance.transition import blend_onto, to_zup

logger = logging.getLogger(__name__)

_KIN = 135  # translation(3) + rotation(132); excludes the 4 contact labels
_BEAT_TOL = 3
_EPS = 1e-8


@dataclass
class HybridResult:
    motion: np.ndarray                     # (L, 139) Z-up hybrid motion
    schedule: list = field(default_factory=list)   # [(start, end, 'lodge'|'edge'), ...]
    reasoning: str = ""
    segment_scores: list = field(default_factory=list)


# --------------------------------------------------------------------------- segmentation
def segment_boundaries(
    metadata: SongMetadata,
    total_frames: int,
    *,
    min_seg_seconds: float = 8.5,
    beats_per_bar: int = 4,
) -> list[int]:
    """Downbeat-aligned segment boundaries, each segment >= ``min_seg_seconds``.

    Falls back to a uniform grid if beat information is unavailable.
    """
    min_frames = max(int(min_seg_seconds * FPS), 1)
    beats = np.asarray(getattr(metadata, "beat_frames", []), dtype=np.int64)
    beats = beats[(beats > 0) & (beats < total_frames)]
    downbeats = beats[::beats_per_bar] if beats.size else np.array([], dtype=np.int64)

    bounds = [0]
    for db in downbeats:
        if db - bounds[-1] >= min_frames and total_frames - db >= min_frames:
            bounds.append(int(db))
    if len(bounds) == 1:  # no usable downbeats -> uniform grid
        step = max(min_frames, total_frames // 4 or total_frames)
        bounds = list(range(0, total_frames, step)) or [0]
        if total_frames - bounds[-1] < min_frames and len(bounds) > 1:
            bounds.pop()
    bounds.append(total_frames)
    # dedupe/sort
    return sorted(set(int(b) for b in bounds if 0 <= b <= total_frames))


# --------------------------------------------------------------------------- scoring
def _segment_metrics(motion: np.ndarray, a: int, b: int, metadata: SongMetadata) -> dict:
    """Local coherence signals for motion[a:b]: smoothness, foot stability, beat sync."""
    seg = motion[a:b]
    if seg.shape[0] < 6:
        return {"jerk": 0.0, "foot_skate": 0.0, "beat_sync": 0.0, "energy": 0.0}
    kin = seg[:, :_KIN]
    jerk = float(np.mean(np.linalg.norm(np.diff(kin, n=3, axis=0), axis=1)))
    # foot skate proxy: horizontal root speed while any contact channel is active
    contact = seg[:, _KIN:139].mean(axis=1)[1:]
    horiz = np.linalg.norm(np.diff(seg[:, [0, 1]], axis=0), axis=1)
    foot_skate = float(np.mean(horiz * contact))
    # beat sync within the window
    root_vel = np.linalg.norm(np.diff(seg[:, :3], axis=0, prepend=seg[:1, :3]), axis=1)
    joint_mv = np.linalg.norm(np.diff(kin, axis=0, prepend=kin[:1]), axis=1)
    energy = 0.6 * root_vel + 0.4 * joint_mv
    import librosa
    dbeats = librosa.onset.onset_detect(onset_envelope=energy, sr=FPS, hop_length=1, units="frames")
    mbeats = np.asarray(metadata.beat_frames, dtype=np.int64)
    mbeats = mbeats[(mbeats >= a) & (mbeats < b)] - a
    if len(dbeats) and len(mbeats):
        aligned = sum(1 for d in dbeats if np.min(np.abs(mbeats - d)) <= _BEAT_TOL)
        beat_sync = aligned / len(dbeats)
    else:
        beat_sync = 0.0
    return {"jerk": jerk, "foot_skate": foot_skate, "beat_sync": float(beat_sync),
            "energy": float(np.mean(energy))}


def _energy_mean(motion: np.ndarray) -> float:
    """Mean per-frame kinematic energy (expressiveness / amount of movement)."""
    if motion.shape[0] < 2:
        return 0.0
    rv = np.linalg.norm(np.diff(motion[:, :3], axis=0, prepend=motion[:1, :3]), axis=1)
    jm = np.linalg.norm(np.diff(motion[:, :_KIN], axis=0, prepend=motion[:1, :_KIN]), axis=1)
    return float(np.mean(0.6 * rv + 0.4 * jm))


def _badness(m: dict, expressiveness: float) -> float:
    """Lower is better. Mirrors the whole-dance objective per segment: penalise unsmoothness /
    foot sliding, reward beat sync + expressiveness (energy)."""
    return m["jerk"] + m["foot_skate"] - m["beat_sync"] - expressiveness * m.get("energy", 0.0)


# Whole-dance objective (lower = better). Combines smoothness/stability (penalised) with
# musicality + expressiveness (rewarded), so the optimum is a dance that is both smooth AND
# musical/energetic -- which a LODGE+EDGE mix can achieve better than either pure generator.
# seam_spikiness carries the (real) switch cost since it rises with churn.
_SCORE_WEIGHTS = {
    "smoothness_jerk": 1.0,
    "jitter_accel": 1.0,
    "seam_spikiness": 0.05,
    "contact_flicker": 0.2,
    "energy_stability": 2.0,
    "bas_trend": -1.0,      # higher is better -> negative weight
    "beat_alignment": -0.5,  # higher is better -> negative weight
}
# Default expressiveness weight: reward per-frame energy (movement). ~4 balances EDGE's smoothness
# against LODGE's energy so the coherence-optimal dance genuinely mixes both (see design plan).
DEFAULT_EXPRESSIVENESS = 4.0
_DIVERSITY_WEIGHT = 0.2


def whole_dance_score(
    motion: np.ndarray, metadata: SongMetadata, *,
    expressiveness: float = DEFAULT_EXPRESSIVENESS,
) -> tuple[float, dict]:
    """Scalar whole-dance cost (lower=better) + breakdown: penalise unsmoothness/instability,
    reward musicality (beat) + expressiveness (energy) + variety (diversity)."""
    from agentlodge.dance.metrics import compute_metrics

    dm = compute_metrics(motion, metadata, "hybrid", "")
    c = dm.coherence.to_dict()
    bd = {
        "smoothness_jerk": c["smoothness_jerk"],
        "jitter_accel": c["jitter_accel"],
        "seam_spikiness": c["seam_spikiness"],
        "contact_flicker": c["contact_flicker"],
        "energy_stability": c["energy_stability"],
        "bas_trend": c["bas_trend"],
        "beat_alignment": dm.beat_alignment_score,
    }
    energy = _energy_mean(motion)
    diversity = dm.motion_diversity
    cost = (sum(_SCORE_WEIGHTS[k] * bd[k] for k in _SCORE_WEIGHTS)
            - expressiveness * energy - _DIVERSITY_WEIGHT * diversity)
    bd["expressiveness"] = energy
    bd["diversity"] = diversity
    return float(cost), bd


def _labels(schedule) -> list[str]:
    return [g for _, _, g in schedule]


def _schedule_from_labels(bounds: list[int], picks: list[str]) -> list[tuple[int, int, str]]:
    return [(bounds[i], bounds[i + 1], picks[i]) for i in range(len(picks))]


# --------------------------------------------------------------------------- scheduling
def select_schedule(
    lodge: np.ndarray,
    edge: np.ndarray,
    bounds: list[int],
    metadata: SongMetadata,
    *,
    scheduler: str = "metric",
    api_key: str | None = None,
    blend_frames: int = 15,
    expressiveness: float = DEFAULT_EXPRESSIVENESS,
    canonical_facing: bool = True,
) -> tuple[list[tuple[int, int, str]], list[dict], str]:
    """Pick a generator per segment. Returns (schedule, per-segment score rows, reasoning)."""
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        lm = _segment_metrics(lodge, a, b, metadata)
        em = _segment_metrics(edge, a, b, metadata)
        lb, eb = _badness(lm, expressiveness), _badness(em, expressiveness)
        rows.append({"a": a, "b": b, "lodge": lm, "edge": em,
                     "lodge_badness": lb, "edge_badness": eb})

    reasoning = "metric scheduler: per-segment pick of the lower-badness generator"
    schedule = [(r["a"], r["b"], "lodge" if r["lodge_badness"] <= r["edge_badness"] else "edge")
                for r in rows]

    if scheduler in {"llm", "llm_global"} and not api_key:
        logger.warning(
            "Scheduler '%s' requested but no OpenAI API key is configured; "
            "falling back to the metric scheduler.", scheduler
        )

    if scheduler == "llm" and api_key:
        try:
            schedule, reasoning = _llm_schedule(rows, metadata, api_key, schedule)
            logger.info("Using LLM scheduler: %s", reasoning)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("LLM scheduler failed (%s); falling back to the metric schedule.", exc)
    elif scheduler == "llm_global" and api_key:
        try:
            schedule, reasoning = _global_llm_schedule(
                lodge, edge, bounds, rows, metadata, api_key, schedule,
                blend_frames=blend_frames, expressiveness=expressiveness,
                canonical_facing=canonical_facing,
            )
            logger.info("Using global LLM scheduler: %s", reasoning)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning(
                "Global LLM scheduler failed (%s); falling back to the metric schedule.", exc
            )
    return schedule, rows, reasoning


def _llm_schedule(rows, metadata, api_key, fallback):
    from openai import OpenAI

    lines = []
    for i, r in enumerate(rows):
        lines.append(
            f"seg {i} [{r['a']/FPS:.1f}-{r['b']/FPS:.1f}s]: "
            f"LODGE(jerk={r['lodge']['jerk']:.3f},foot={r['lodge']['foot_skate']:.3f},beat={r['lodge']['beat_sync']:.2f}) "
            f"EDGE(jerk={r['edge']['jerk']:.3f},foot={r['edge']['foot_skate']:.3f},beat={r['edge']['beat_sync']:.2f})"
        )
    prompt = (
        "Assemble ONE hybrid dance by choosing, for each time-segment, whether LODGE or EDGE "
        "drives it, to maximise the WHOLE dance's quality (smoothness=low jerk, stable feet=low "
        "foot, strong musical sync=high beat). Switching is FREE (transitions are auto-blended); "
        "choose whatever makes the overall piece best.\n\n"
        f"Song ~{metadata.duration_seconds:.0f}s, ~{metadata.bpm:.0f} bpm. Segments:\n"
        + "\n".join(lines)
        + '\n\nRespond JSON only: {"schedule": ["lodge"|"edge", ...one per segment...], '
        '"reasoning": "brief"}'
    )
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    payload = json.loads(m.group(), strict=False)
    picks = [str(p).lower().strip() for p in payload["schedule"]]
    if len(picks) != len(rows) or any(p not in {"lodge", "edge"} for p in picks):
        raise ValueError("LLM schedule length/labels invalid")
    schedule = [(r["a"], r["b"], picks[i]) for i, r in enumerate(rows)]
    return schedule, "llm scheduler: " + str(payload.get("reasoning", "")).strip()


def _global_llm_schedule(lodge, edge, bounds, rows, metadata, api_key, seed, *,
                         blend_frames=15, rounds=4, expressiveness=DEFAULT_EXPRESSIVENESS,
                         canonical_facing=True):
    """Globally-aware scheduler: the LLM proposes whole schedules, we ASSEMBLE + measure the real
    whole-dance cost, and feed that back so it optimises the true objective (with switch cost)
    over a few rounds. Returns the best-measured schedule.
    """
    from openai import OpenAI

    def evaluate(picks):
        sched = _schedule_from_labels(bounds, picks)
        motion = assemble(lodge, edge, sched, blend_frames=blend_frames,
                          canonical_facing=canonical_facing)
        cost, bd = whole_dance_score(motion, metadata, expressiveness=expressiveness)
        return cost, bd, sched

    nseg = len(rows)
    seed_picks = _labels(seed)
    best_cost, best_bd, best_sched = evaluate(seed_picks)
    tried = {tuple(seed_picks): best_cost}
    history = [(seed_picks, best_cost)]

    seg_lines = [
        f"seg{i} [{r['a']/FPS:.1f}-{r['b']/FPS:.1f}s]: "
        f"LODGE(jerk={r['lodge']['jerk']:.3f},beat={r['lodge']['beat_sync']:.2f},energy={r['lodge']['energy']:.3f}) "
        f"EDGE(jerk={r['edge']['jerk']:.3f},beat={r['edge']['beat_sync']:.2f},energy={r['edge']['energy']:.3f})"
        for i, r in enumerate(rows)
    ]
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    for rnd in range(rounds):
        hist_str = "\n".join(f"  {p} -> cost {c:.4f}" for p, c in history[-6:])
        prompt = (
            "You optimise a HYBRID dance assembled from a LODGE dance and an EDGE dance. For each "
            "segment you choose 'lodge' or 'edge'; chosen segments are concatenated and joined with "
            "automatic inertialized transitions. MINIMISE the measured whole-dance cost:\n"
            f"  cost = smoothness_jerk + jitter_accel + 0.05*seam_spikiness + 0.2*contact_flicker "
            f"+ 2*energy_stability - bas_trend - 0.5*beat_alignment - {expressiveness:g}*expressiveness "
            "- 0.2*diversity   (LOWER is better).\n"
            "So a good dance is SMOOTH (low jerk) AND musical/expressive (high beat sync + energy + "
            "variety). EDGE is usually smoother (low jerk); LODGE is usually more energetic and "
            "on-beat -- so mixing can beat either alone.\n"
            "GLOBAL effects not visible per-segment: every generator SWITCH warps ~0.5s of the "
            "incoming segment and raises seam_spikiness; frequent switching also hurts macro "
            "coherence. Prefer FEW, well-placed switches -- switch to LODGE for energetic/on-beat "
            "sections, stay on EDGE for smoothness.\n\n"
            f"Song ~{metadata.duration_seconds:.0f}s ~{metadata.bpm:.0f}bpm, {nseg} segments:\n"
            + "\n".join(seg_lines)
            + f"\n\nBest so far: {_labels(best_sched)} -> cost {best_cost:.4f}; breakdown {best_bd}\n"
            f"Measured attempts:\n{hist_str}\n\n"
            f"Propose a NEW schedule (exactly {nseg} entries of 'lodge'/'edge') that LOWERS the "
            'measured cost. JSON only: {"schedule": [...], "reasoning": "brief"}'
        )
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            mm = re.search(r"\{.*\}", text, re.DOTALL)
            payload = json.loads(mm.group(), strict=False)
            picks = [str(p).lower().strip() for p in payload["schedule"]]
            if len(picks) != nseg or any(p not in {"lodge", "edge"} for p in picks):
                continue
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("global-LLM round %d failed: %s", rnd, exc)
            continue
        if tuple(picks) in tried:
            history.append((picks, tried[tuple(picks)]))
            continue
        cost, bd, sched = evaluate(picks)
        tried[tuple(picks)] = cost
        history.append((picks, cost))
        if cost < best_cost:
            best_cost, best_bd, best_sched = cost, bd, sched

    return best_sched, (f"llm_global scheduler: best measured whole-dance cost {best_cost:.4f} "
                        f"over {len(tried)} evaluated schedules")


# --------------------------------------------------------------------------- assembly
def _merge_runs(schedule: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Merge consecutive same-generator segments into contiguous runs."""
    runs: list[tuple[int, int, str]] = []
    for a, b, gen in schedule:
        if runs and runs[-1][2] == gen and runs[-1][1] == a:
            pa, _, pg = runs[-1]
            runs[-1] = (pa, b, pg)
        else:
            runs.append((a, b, gen))
    return runs


def assemble(lodge: np.ndarray, edge: np.ndarray, schedule, blend_frames: int = 15,
             canonical_facing: bool = True) -> np.ndarray:
    """Concatenate the scheduled runs, inertially blending at each generator switch.

    When ``canonical_facing`` is True, every run is anchored to the opening run's facing so the
    dancer keeps a consistent orientation (no cumulative rotation drift across switches).
    """
    from agentlodge.dance.transition import root_yaw

    src = {"lodge": lodge, "edge": edge}
    runs = _merge_runs(schedule)
    committed: np.ndarray | None = None
    canon: float | None = None
    for a, b, gen in runs:
        seg = src[gen][a:b].copy()
        if seg.shape[0] == 0:
            continue
        if committed is None:
            committed = seg
            canon = root_yaw(seg[0]) if canonical_facing else None
        else:
            blended = blend_onto(committed[-2:], seg, blend_frames, canonical_yaw=canon)
            committed = np.concatenate([committed, blended], axis=0)
    return committed if committed is not None else src["lodge"]


def build_hybrid(
    lodge_motion: np.ndarray,
    edge_motion: np.ndarray,
    metadata: SongMetadata,
    *,
    min_seg_seconds: float = 8.5,
    blend_frames: int = 15,
    scheduler: str = "metric",
    api_key: str | None = None,
    expressiveness: float = DEFAULT_EXPRESSIVENESS,
    canonical_facing: bool = True,
) -> HybridResult:
    """Assemble a hybrid dance from independently generated LODGE and EDGE motions."""
    lodge = to_zup(ensure_lodge139(lodge_motion))      # LODGE Y-up -> Z-up
    edge = ensure_lodge139(edge_motion)                # EDGE already Z-up
    n = min(lodge.shape[0], edge.shape[0])
    if n < FPS:
        raise ValueError(f"Motions too short to hybridize ({n} frames)")
    lodge, edge = lodge[:n], edge[:n]

    bounds = segment_boundaries(metadata, n, min_seg_seconds=min_seg_seconds)
    schedule, rows, reasoning = select_schedule(
        lodge, edge, bounds, metadata, scheduler=scheduler, api_key=api_key,
        blend_frames=blend_frames, expressiveness=expressiveness,
        canonical_facing=canonical_facing,
    )
    motion = assemble(lodge, edge, schedule, blend_frames=blend_frames,
                      canonical_facing=canonical_facing)
    runs = _merge_runs(schedule)
    seg_desc = ", ".join(f"{a/FPS:.1f}-{b/FPS:.1f}s:{g}" for a, b, g in runs)
    logger.info("Hybrid schedule (%d runs): %s", len(runs), seg_desc)
    return HybridResult(motion=motion, schedule=schedule, reasoning=reasoning,
                        segment_scores=rows)
