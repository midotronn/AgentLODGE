"""Best-of-K seed selection for the diffusion dance generators (EDGE / LODGE).

Both generators are seed-stochastic and neither explicitly optimizes beat alignment, so beat sync
varies noticeably across seeds. This module samples K candidates for a musical section and selects
the one that best matches a composite objective dominated by **Beat Alignment Score** (the
pipeline's main weakness), following the inference-time-scaling result that random best-of-N with a
task verifier is a strong, low-risk use of extra compute (Ma et al., 2025, arXiv:2501.09732).

It is **generator-agnostic**: pass a ``generate_fn(seed) -> (L, 139) motion`` closure that wraps a
real seeded LODGE or EDGE call, so the scoring/selection logic here is fully unit-testable without
the heavy models. Candidates are independent -> the K generations can be run in parallel.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from agentlodge.dance.beat_metrics import (
    _kinematic_speed,
    beat_alignment_score,
    foot_contact_consistency,
)

# Composite weights: BAS dominates (the weakness); foot-contact + energy match are secondary.
_W_BAS = 0.6
_W_FOOT = 0.25
_W_ENERGY = 0.15


def _mean_speed(motion: np.ndarray) -> float:
    return float(np.mean(_kinematic_speed(motion)))


def score_candidates(motions: Sequence[np.ndarray], music_beat_frames, *,
                     target_intensity: float | None = None, sigma_frames: float = 3.0,
                     w_bas: float = _W_BAS, w_foot: float = _W_FOOT,
                     w_energy: float = _W_ENERGY) -> list[dict]:
    """Score each candidate motion. Energy match is computed *across the candidate set* (min-max),
    so ``target_intensity`` in [0,1] means "prefer the livelier / calmer candidate"."""
    n = len(motions)
    bas = [beat_alignment_score(m, music_beat_frames, sigma_frames=sigma_frames) for m in motions]
    foot = [foot_contact_consistency(m) for m in motions]
    speeds = np.array([_mean_speed(m) for m in motions], dtype=np.float64)
    if target_intensity is not None and n > 1 and speeds.max() > speeds.min():
        erel = (speeds - speeds.min()) / (speeds.max() - speeds.min())
        ems = 1.0 - np.abs(erel - float(target_intensity))
    else:
        ems = np.ones(n)
    out = []
    for i in range(n):
        total = w_bas * bas[i] + w_foot * foot[i] + w_energy * float(ems[i])
        out.append({
            "bas": round(float(bas[i]), 4),
            "foot_contact_consistency": round(float(foot[i]), 4),
            "energy_match": round(float(ems[i]), 4),
            "total": round(float(total), 4),
        })
    return out


def select_best(motions: Sequence[np.ndarray], music_beat_frames, **kw) -> tuple[int, list[dict]]:
    """Return ``(best_index, per_candidate_scores)`` — argmax of the composite score."""
    if not motions:
        raise ValueError("select_best: no candidates")
    scores = score_candidates(motions, music_beat_frames, **kw)
    best = int(np.argmax([s["total"] for s in scores]))
    return best, scores


def best_of_k(generate_fn: Callable[[int], np.ndarray | None], seeds: Sequence[int],
              music_beat_frames, *, target_intensity: float | None = None,
              min_frames: int = 3, **score_kw) -> tuple[np.ndarray, int, dict]:
    """Generate one candidate per seed via ``generate_fn(seed)``, score, and return the best.

    Returns ``(best_motion, best_seed, report)``. ``report`` records every seed's scores and the
    winner. Invalid / too-short / failed generations (``None``) are skipped. Raises if none succeed.
    """
    motions: list[np.ndarray] = []
    used: list[int] = []
    for s in seeds:
        try:
            m = generate_fn(int(s))
        except Exception:  # noqa: BLE001 - a bad seed shouldn't kill the whole search
            m = None
        if m is not None and np.asarray(m).ndim == 2 and np.asarray(m).shape[0] >= min_frames:
            motions.append(np.asarray(m, dtype=np.float32))
            used.append(int(s))
    if not motions:
        raise ValueError("best_of_k: no valid candidates were generated")
    best, scores = select_best(motions, music_beat_frames, target_intensity=target_intensity,
                               **score_kw)
    report = {
        "k": len(motions),
        "seeds": used,
        "scores": scores,
        "winner_index": best,
        "winner_seed": used[best],
        "winner_bas": scores[best]["bas"],
    }
    return motions[best], used[best], report
