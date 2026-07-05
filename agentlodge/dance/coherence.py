"""Long-term coherence metrics for generated dance motion.

Both LODGE and EDGE build long-form dances by stitching shorter segments together
(LODGE global+local windows with 8-frame blends; EDGE 5s clips with 2.5s overlaps),
so the quality that matters most is how coherent the motion stays over the whole song:
smooth transitions across seams, no jitter or sudden jumps, stable foot planting, and
sustained musical synchronization without freezing or looping.

All signals here are computed directly from the raw 139-dim motion representation
(no forward kinematics), so they are cheap and run in the main process. The layout is
the AgentLODGE ``ensure_lodge139`` layout::

    [ root_translation(3) | 22-joint 6D rotation(132) | foot contact(4) ]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from agentlodge.audio.preprocess import SongMetadata
from agentlodge.config import FPS
from agentlodge.dance.format import ensure_lodge139

NUM_WINDOWS = 6
BEAT_TOLERANCE_FRAMES = 3
_KIN = 135  # translation(3) + rotation(132); excludes the 4 contact labels
_EPS = 1e-8


@dataclass
class CoherenceProfile:
    """Long-term coherence signals for a single dance (higher window arrays = per-time)."""

    smoothness_jerk: float  # mean |3rd derivative| of pose; lower = smoother
    jitter_accel: float  # mean |2nd derivative|; lower = less high-frequency jitter
    seam_spikiness: float  # p95/median of acceleration; higher = sudden stitch jumps
    foot_skating: float  # root drift while feet are in contact; lower = feet stay planted
    contact_flicker: float  # foot contact on/off toggles per second; lower = stabler planting
    bas_trend: float  # slope of per-window beat-sync; negative = sync degrades over time
    energy_stability: float  # std of per-window energy; lower = steadier dynamics
    window_energy: list[float] = field(default_factory=list)
    window_bas: list[float] = field(default_factory=list)
    window_smoothness: list[float] = field(default_factory=list)
    window_variety: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "smoothness_jerk": round(self.smoothness_jerk, 5),
            "jitter_accel": round(self.jitter_accel, 5),
            "seam_spikiness": round(self.seam_spikiness, 4),
            "foot_skating": round(self.foot_skating, 5),
            "contact_flicker": round(self.contact_flicker, 4),
            "bas_trend": round(self.bas_trend, 5),
            "energy_stability": round(self.energy_stability, 5),
            "window_energy": [round(v, 4) for v in self.window_energy],
            "window_bas": [round(v, 4) for v in self.window_bas],
            "window_smoothness": [round(v, 4) for v in self.window_smoothness],
            "window_variety": [round(v, 4) for v in self.window_variety],
        }


def _kinematic_energy(motion: np.ndarray) -> np.ndarray:
    root_vel = np.linalg.norm(
        np.diff(motion[:, :3], axis=0, prepend=motion[:1, :3]), axis=1
    )
    joint_motion = np.linalg.norm(
        np.diff(motion[:, :_KIN], axis=0, prepend=motion[:1, :_KIN]), axis=1
    )
    return 0.6 * root_vel + 0.4 * joint_motion


def _window_bas(kin_energy: np.ndarray, music_beats_in_window: np.ndarray) -> float:
    if kin_energy.shape[0] < 3 or music_beats_in_window.size == 0:
        return 0.0
    dance_beats = librosa.onset.onset_detect(
        onset_envelope=kin_energy, sr=FPS, hop_length=1, units="frames"
    )
    if len(dance_beats) == 0:
        return 0.0
    aligned = sum(
        1
        for db in dance_beats
        if np.min(np.abs(music_beats_in_window - db)) <= BEAT_TOLERANCE_FRAMES
    )
    return float(aligned / len(dance_beats))


def _degenerate() -> CoherenceProfile:
    zeros = [0.0] * NUM_WINDOWS
    return CoherenceProfile(
        smoothness_jerk=0.0,
        jitter_accel=0.0,
        seam_spikiness=0.0,
        foot_skating=0.0,
        contact_flicker=0.0,
        bas_trend=0.0,
        energy_stability=0.0,
        window_energy=list(zeros),
        window_bas=list(zeros),
        window_smoothness=list(zeros),
        window_variety=[0.0] * (NUM_WINDOWS - 1),
    )


def compute_coherence(motion: np.ndarray, metadata: SongMetadata) -> CoherenceProfile:
    """Compute long-term coherence signals from a motion array (139 or 151 dims)."""
    m = ensure_lodge139(motion)
    length = m.shape[0]
    if length < 8:
        return _degenerate()

    kin = m[:, :_KIN]
    contact = m[:, _KIN:139]

    # Derivatives of the kinematic (non-contact) channels.
    acc = np.diff(kin, n=2, axis=0)
    jerk = np.diff(kin, n=3, axis=0)
    acc_norm = np.linalg.norm(acc, axis=1)
    smoothness_jerk = float(np.mean(np.linalg.norm(jerk, axis=1)))
    jitter_accel = float(np.mean(acc_norm))
    seam_spikiness = float(np.percentile(acc_norm, 95) / (np.median(acc_norm) + _EPS))

    # Foot skating proxy: horizontal root speed (Y-up -> x,z) while feet are in contact.
    horiz_speed = np.linalg.norm(np.diff(m[:, [0, 2]], axis=0), axis=1)
    contact_active = contact.mean(axis=1)[1:]
    foot_skating = float(np.mean(horiz_speed * contact_active))

    # Contact flicker: on/off toggles per second across the four contact channels.
    contact_bin = (contact > 0.5).astype(np.float32)
    toggles = float(np.abs(np.diff(contact_bin, axis=0)).sum())
    contact_flicker = toggles / (length / FPS)

    # Per-window temporal trends.
    energy_full = _kinematic_energy(m)
    bounds = np.linspace(0, length, NUM_WINDOWS + 1, dtype=int)
    music_beats = np.asarray(metadata.beat_frames, dtype=np.int64)

    window_energy: list[float] = []
    window_bas: list[float] = []
    window_smoothness: list[float] = []
    window_means: list[np.ndarray] = []
    for w in range(NUM_WINDOWS):
        a, b = int(bounds[w]), int(bounds[w + 1])
        seg = m[a:b]
        if seg.shape[0] < 4:
            window_energy.append(0.0)
            window_bas.append(0.0)
            window_smoothness.append(0.0)
            window_means.append(np.zeros(_KIN, dtype=np.float32))
            continue
        window_energy.append(float(np.mean(energy_full[a:b])))
        mb = music_beats[(music_beats >= a) & (music_beats < b)] - a
        window_bas.append(_window_bas(_kinematic_energy(seg), mb))
        jseg = np.diff(seg[:, :_KIN], n=3, axis=0)
        window_smoothness.append(
            float(np.mean(np.linalg.norm(jseg, axis=1))) if jseg.shape[0] else 0.0
        )
        window_means.append(seg[:, :_KIN].mean(axis=0))

    window_variety = [
        float(np.linalg.norm(window_means[i + 1] - window_means[i]))
        for i in range(len(window_means) - 1)
    ]

    idx = np.arange(len(window_bas))
    bas_trend = float(np.polyfit(idx, window_bas, 1)[0]) if len(window_bas) > 1 else 0.0
    energy_stability = float(np.std(window_energy))

    return CoherenceProfile(
        smoothness_jerk=smoothness_jerk,
        jitter_accel=jitter_accel,
        seam_spikiness=seam_spikiness,
        foot_skating=foot_skating,
        contact_flicker=contact_flicker,
        bas_trend=bas_trend,
        energy_stability=energy_stability,
        window_energy=window_energy,
        window_bas=window_bas,
        window_smoothness=window_smoothness,
        window_variety=window_variety,
    )
