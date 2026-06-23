"""Beat alignment and motion diversity metrics."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from agentlodge.config import FPS, HOP_LENGTH, SAMPLE_RATE
from agentlodge.audio.preprocess import SongMetadata


BEAT_TOLERANCE_FRAMES = 3
CLIP_LENGTH = FPS * 2
NUM_CLIPS = 8


@dataclass
class DanceMetrics:
    beat_alignment_score: float
    motion_diversity: float
    summary: str


def _kinematic_energy(motion: np.ndarray) -> np.ndarray:
    """Per-frame scalar energy from root velocity and joint motion."""
    root_vel = np.linalg.norm(np.diff(motion[:, :3], axis=0, prepend=motion[:1, :3]), axis=1)
    joint_motion = np.linalg.norm(np.diff(motion, axis=0, prepend=motion[:1]), axis=1)
    return 0.6 * root_vel + 0.4 * joint_motion


def _detect_kinematic_beats(energy: np.ndarray) -> np.ndarray:
    envelope = librosa.onset.onset_strength(onset_envelope=energy, sr=SAMPLE_RATE, hop_length=1)
    peaks = librosa.onset.onset_detect(onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=1)
    return np.asarray(peaks, dtype=np.int64)


def beat_alignment_score(motion: np.ndarray, metadata: SongMetadata) -> float:
    energy = _kinematic_energy(motion)
    dance_beats = _detect_kinematic_beats(energy)
    music_beats = metadata.beat_frames

    if len(dance_beats) == 0 or len(music_beats) == 0:
        return 0.0

    aligned = 0
    for db in dance_beats:
        if np.min(np.abs(music_beats - db)) <= BEAT_TOLERANCE_FRAMES:
            aligned += 1
    return float(aligned / len(dance_beats))


def motion_diversity(motion: np.ndarray) -> float:
    length = motion.shape[0]
    if length < CLIP_LENGTH + 1:
        return 0.0

    max_start = length - CLIP_LENGTH
    starts = np.linspace(0, max_start, NUM_CLIPS, dtype=int)
    clips = [motion[s : s + CLIP_LENGTH] for s in starts]
    features = [clip.mean(axis=0) for clip in clips]

    distances: list[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distances.append(float(np.linalg.norm(features[i] - features[j])))
    return float(np.mean(distances)) if distances else 0.0


def compute_metrics(
    motion: np.ndarray,
    metadata: SongMetadata,
    model_name: str,
    generation_notes: str,
) -> DanceMetrics:
    bas = beat_alignment_score(motion, metadata)
    diversity = motion_diversity(motion)
    summary = (
        f"{model_name.upper()} generated {motion.shape[0]} frames "
        f"({motion.shape[0] / FPS:.1f}s) at 30 FPS. "
        f"{generation_notes} "
        f"BAS={bas:.3f}, diversity={diversity:.3f}."
    )
    return DanceMetrics(
        beat_alignment_score=bas,
        motion_diversity=diversity,
        summary=summary,
    )
