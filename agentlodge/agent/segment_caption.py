"""Natural-language descriptions of dance segments + plan↔realization alignment (verifier).

The storyboard LLM reasons over energy *numbers*; describing each segment in words lets it (and an
editing agent) reason over *descriptions*, which the LLM-reasoning literature (CoT / ReAct /
Reflexion) shows improves decisions. It also makes the storyboard's ``vocabulary`` field -- which
the assembler otherwise ignores -- actually mean something, via a vocabulary↔energy alignment.

Captions here are **kinematic-feature templated** (pure-numpy, FK-free, deterministic) so they run
anywhere and are unit-testable. A learned captioner (MotionGPT / TM2T) and a learned text-motion
critic (TMR) can be dropped in later: :func:`plan_realization_alignment` accepts an optional
``tmr_score`` from such a critic.
"""

from __future__ import annotations

import numpy as np

from agentlodge.config import FPS
from agentlodge.dance.beat_metrics import _kinematic_speed, kinematic_beats

_KIN = 135
_CONTACT = slice(135, 139)
_EPS = 1e-8

# Movement-vocabulary (storyboard) → expected normalized energy, for vocabulary↔motion matching.
VOCAB_ENERGY = {
    "grounded_minimal": 0.12,
    "sustained_lyrical": 0.33,
    "flowing_smooth": 0.35,
    "expansive_traveling": 0.55,
    "percussive_sharp": 0.70,
    "explosive_fast": 0.90,
}


def segment_features(motion: np.ndarray) -> dict:
    """Scale-free, FK-free descriptors of a 139-dim segment used to build a caption."""
    L = int(motion.shape[0])
    speed = _kinematic_speed(motion, smooth_frames=max(1, FPS // 10))
    mean_e = float(np.mean(speed))
    cov = float(np.std(speed) / (mean_e + _EPS))              # coefficient of variation (punchiness)
    t = np.arange(L)
    trend = float(np.corrcoef(t, speed)[0, 1]) if L > 2 and speed.std() > _EPS else 0.0
    xy = motion[:, :2]
    net = float(np.linalg.norm(xy[-1] - xy[0])) if L > 1 else 0.0
    path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if L > 1 else 0.0
    directionality = float(net / (path + _EPS))               # 1 = travels one way, 0 = stays put
    contact_rate = float(np.mean(motion[:, _CONTACT]))
    kb = kinematic_beats(motion)
    if kb.size >= 3:
        d = np.diff(kb).astype(float)
        regularity = float(np.clip(1.0 - np.std(d) / (np.mean(d) + _EPS), 0.0, 1.0))
    else:
        regularity = 0.0
    return {
        "mean_energy": mean_e, "cov": cov, "trend": trend, "directionality": directionality,
        "contact_rate": contact_rate, "regularity": regularity, "n_motion_beats": int(kb.size),
    }


def _level(energy_norm: float) -> str:
    if energy_norm >= 0.78:
        return "explosive"
    if energy_norm >= 0.55:
        return "energetic"
    if energy_norm >= 0.30:
        return "moderate"
    return "calm"


def caption_segment(motion: np.ndarray, *, energy_norm: float | None = None) -> str:
    """A concise natural-language description of a dance segment.

    ``energy_norm`` (0..1, e.g. the segment's energy relative to the song) adds an absolute energy
    word; omit it and the caption uses only scale-free descriptors.
    """
    f = segment_features(motion)
    trend = ("building" if f["trend"] > 0.25 else
             "settling" if f["trend"] < -0.25 else "steady")
    dyn = ("punchy, accented" if f["cov"] > 0.9 else
           "varied" if f["cov"] > 0.5 else "even, sustained")
    rhythm = "on a regular pulse" if f["regularity"] > 0.6 else "with free timing"
    ground = "mostly grounded" if f["contact_rate"] > 0.5 else "often airborne"
    space = ("traveling across the floor" if f["directionality"] > 0.5 else
             "staying centered" if f["directionality"] < 0.2 else "shifting in place")
    lead = f"A {_level(energy_norm)}, " if energy_norm is not None else "A "
    return (f"{lead}{trend} phrase with {dyn} dynamics, {rhythm}, {ground}, {space}.")


def vocabulary_match(energy_norm: float, vocabulary: str) -> float:
    """How well a segment's (normalized) energy matches the plan's movement vocabulary, in [0,1]."""
    tgt = VOCAB_ENERGY.get(str(vocabulary))
    if tgt is None:
        return 1.0  # empty / unknown vocabulary -> neutral (don't penalize)
    return float(np.clip(1.0 - abs(float(energy_norm) - tgt), 0.0, 1.0))


def plan_realization_alignment(plan, energy_norm: float, *, tmr_score: float | None = None) -> float:
    """Verifier score in [0,1]: does a realized segment match its plan's intent?

    Combines target-intensity match with vocabulary↔energy match. If a learned text-motion critic
    (e.g. TMR) is available, pass its similarity as ``tmr_score`` to blend it in.
    """
    intensity = 1.0 - abs(float(energy_norm) - float(getattr(plan, "target_intensity", energy_norm)))
    vocab = vocabulary_match(energy_norm, getattr(plan, "vocabulary", ""))
    base = 0.5 * intensity + 0.5 * vocab
    if tmr_score is not None:
        base = 0.6 * base + 0.4 * float(tmr_score)
    return float(np.clip(base, 0.0, 1.0))
