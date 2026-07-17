"""Beat-alignment metrics for a generated/assembled dance (FK-free, pure-numpy + scipy).

The pipeline had *no* beat-alignment metric, yet beat sync is its main weakness. These quantify
how well a dance's kinematic "beats" (moments the body decelerates into a pose) land on the music
beats, following the Beat Alignment Score (BAS) of Li et al. (AI Choreographer / AIST++, ICCV
2021).

Motion beats are detected FK-free from the AgentLODGE 139-dim motion's kinematic-velocity envelope
(same channels ``[trans(3) | rot(132)]`` used by ``story_metrics._frame_energy``), so this runs
anywhere and is unit-testable without forward kinematics or the audio stack. A future FK-based
variant (foot/joint world positions) can replace :func:`kinematic_beats` without changing the BAS
interface.

Conventions
-----------
* ``music_beat_frames`` are integer indices in **motion-frame units** (30 FPS) -- the same
  convention ``structure.py`` uses (``metadata.beat_frames`` compared against motion frame counts).
  ``librosa.beat.beat_track(..., hop_length=HOP_LENGTH)`` returns beats at this rate because
  ``SAMPLE_RATE == FPS * HOP_LENGTH``.
* BAS is in ``[0, 1]`` (higher = better beat alignment).
"""

from __future__ import annotations

import numpy as np

from agentlodge.config import FPS

_KIN = 135  # trans(3) + rot(132); excludes the 4 contact labels
_TRANS_XY = slice(0, 2)
_CONTACT = slice(135, 139)
_EPS = 1e-8


def _kinematic_speed(motion: np.ndarray, smooth_frames: int = 0) -> np.ndarray:
    """Per-frame speed of the kinematic channels (root translation + joint rotations)."""
    kin = motion[:, :_KIN]
    speed = np.linalg.norm(np.diff(kin, axis=0, prepend=kin[:1]), axis=1)
    if smooth_frames and smooth_frames > 1:
        k = np.ones(int(smooth_frames), dtype=np.float64) / int(smooth_frames)
        speed = np.convolve(speed, k, mode="same")
    return speed


def kinematic_beats(motion: np.ndarray, *, smooth_frames: int | None = None,
                    min_distance: int | None = None,
                    prominence: float | None = None) -> np.ndarray:
    """Frames where the body "hits" a beat = local minima of the kinematic-speed envelope.

    A dancer typically decelerates into an accented pose on the beat, so speed troughs are the
    standard proxy for kinematic beats (Li et al. 2021). Returns sorted integer frame indices.
    """
    from scipy.signal import find_peaks

    L = int(motion.shape[0])
    if L < 3:
        return np.zeros(0, dtype=int)
    smooth = smooth_frames if smooth_frames is not None else max(1, FPS // 10)
    speed = _kinematic_speed(motion, smooth)
    dist = min_distance if min_distance is not None else max(1, FPS // 8)
    prom = prominence if prominence is not None else float(np.std(speed) * 0.1 + _EPS)
    # troughs of speed == peaks of (-speed)
    peaks, _ = find_peaks(-speed, distance=dist, prominence=prom)
    return peaks.astype(int)


def beat_alignment_score(motion: np.ndarray, music_beat_frames, *, sigma_frames: float = 3.0,
                         motion_beats: np.ndarray | None = None) -> float:
    """Beat Alignment Score in ``[0, 1]`` (Li et al., AIST++).

    ``BAS = mean_{b in motion_beats} exp(-min_a |b - a|^2 / (2*sigma^2))`` where ``a`` ranges over
    ``music_beat_frames``. ``sigma_frames`` ~3 (=0.1 s @30 FPS) is strict; ~5 is lenient.
    """
    mb = np.asarray(music_beat_frames, dtype=np.float64)
    mb = mb[np.isfinite(mb)]
    kb = motion_beats if motion_beats is not None else kinematic_beats(motion)
    kb = np.asarray(kb, dtype=np.float64)
    if mb.size == 0 or kb.size == 0:
        return 0.0
    nearest = np.min(np.abs(kb[:, None] - mb[None, :]), axis=1)
    return float(np.mean(np.exp(-(nearest ** 2) / (2.0 * float(sigma_frames) ** 2))))


def beat_coverage(motion: np.ndarray, music_beat_frames, *, tol_frames: int | None = None,
                  motion_beats: np.ndarray | None = None) -> float:
    """Two-sided complement to BAS: fraction of MUSIC beats with a motion beat within ``tol``.

    BAS only scores motion beats against music beats (a frozen dancer with one lucky motion beat
    can score high); coverage penalizes music beats that have no corresponding motion beat.
    """
    mb = np.asarray(music_beat_frames, dtype=np.float64)
    mb = mb[np.isfinite(mb)]
    kb = motion_beats if motion_beats is not None else kinematic_beats(motion)
    kb = np.asarray(kb, dtype=np.float64)
    if mb.size == 0 or kb.size == 0:
        return 0.0
    tol = tol_frames if tol_frames is not None else max(1, FPS // 8)
    hits = int(np.sum([np.min(np.abs(kb - b)) <= tol for b in mb]))
    return float(hits / mb.size)


def foot_contact_consistency(motion: np.ndarray, *, move_thresh: float | None = None) -> float:
    """FK-free foot-skate proxy in ``[0, 1]`` (higher = less sliding).

    Uses the 4 contact labels + root horizontal (XY) velocity: of the frames a foot is labelled
    in-contact, the fraction whose root is *not* sliding faster than ``move_thresh``. Not the full
    physical PFC (which needs foot joint positions/acceleration via FK) -- a cheap first proxy.
    """
    L = int(motion.shape[0])
    if L < 2:
        return 1.0
    contact = motion[:, _CONTACT].mean(axis=1)
    horiz = np.linalg.norm(np.diff(motion[:, _TRANS_XY], axis=0, prepend=motion[:1, _TRANS_XY]), axis=1)
    thr = move_thresh if move_thresh is not None else float(np.median(horiz) + _EPS)
    in_contact = contact > 0.5
    n = int(in_contact.sum())
    if n == 0:
        return 1.0
    sliding = int(np.sum(in_contact & (horiz > thr)))
    return float(1.0 - sliding / n)


def compute_beat_metrics(motion: np.ndarray, music_beat_frames, *,
                         sigma_frames: float = 3.0) -> dict:
    """Aggregate beat metrics for an assembled dance given the song's beat frames."""
    kb = kinematic_beats(motion)
    return {
        "beat_alignment": round(beat_alignment_score(
            motion, music_beat_frames, sigma_frames=sigma_frames, motion_beats=kb), 4),
        "beat_coverage": round(beat_coverage(motion, music_beat_frames, motion_beats=kb), 4),
        "foot_contact_consistency": round(foot_contact_consistency(motion), 4),
        "n_motion_beats": int(kb.size),
    }
