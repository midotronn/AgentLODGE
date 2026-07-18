"""Music structure analysis: parse a song into sections + an energy arc.

This module gives the choreography pipeline an explicit, training-free view of a song's *form*:

  * ``sections`` -- contiguous spans (intro / verse / chorus / bridge / drop / outro) with
    repetition labels (same label == musically similar section, e.g. the two choruses),
  * ``energy_curve`` -- a per-frame normalized intensity used to shape the dance's energy arc
    (build -> climax -> resolution),
  * ``climax_index`` -- the peak-energy section.

Boundaries come from librosa's agglomerative (Laplacian-style) segmentation over stacked
chroma+MFCC features; repetition labels from a self-similarity comparison of section means; the
energy curve from RMS + spectral flux. Everything is snapped to musical downbeats and constrained
to a minimum section length. If librosa is unavailable or segmentation fails, a robust fallback
produces downbeat/uniform sections so the caller never hard-fails.

librosa is imported lazily so importing this module (and unit-testing the pure-numpy helpers)
does not require the audio stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from agentlodge.config import FPS, HOP_LENGTH, SAMPLE_RATE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentlodge.audio.preprocess import SongMetadata

logger = logging.getLogger(__name__)

_HI, _LO = 0.6, 0.4  # energy thresholds for role inference (on the [0,1] section energy)
_ROLES = ("intro", "verse", "chorus", "bridge", "drop", "outro")


@dataclass
class Section:
    """One contiguous musical section, in motion-frame units (30 FPS)."""

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    label: int          # repetition group id; same label => musically similar section
    role: str           # one of _ROLES
    energy: float       # normalized [0, 1] mean intensity
    repeat_of: int | None = None  # earliest earlier section with the same label, else None

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "label": self.label,
            "role": self.role,
            "energy": round(self.energy, 3),
            "repeat_of": self.repeat_of,
        }


@dataclass
class MusicStructure:
    """Detected musical form + energy arc for a song."""

    sections: list[Section]
    energy_curve: np.ndarray            # (total_frames,) normalized [0, 1]
    recurrence: np.ndarray              # (n_sections, n_sections) section similarity
    climax_index: int
    tempo: float
    total_frames: int
    used_fallback: bool = False

    def boundaries(self) -> list[int]:
        """Frame boundaries ``[0, ..., total_frames]`` (len == n_sections + 1)."""
        if not self.sections:
            return [0, self.total_frames]
        return [self.sections[0].start_frame] + [s.end_frame for s in self.sections]

    def to_dict(self) -> dict:
        # Down-sample the energy curve so it stays small in pipeline_log.json.
        n = min(48, len(self.energy_curve)) or 1
        idx = np.linspace(0, max(len(self.energy_curve) - 1, 0), n).astype(int)
        return {
            "n_sections": len(self.sections),
            "climax_index": self.climax_index,
            "tempo": round(self.tempo, 1),
            "used_fallback": self.used_fallback,
            "sections": [s.to_dict() for s in self.sections],
            "energy_curve_ds": [round(float(v), 3) for v in self.energy_curve[idx]],
        }


# --------------------------------------------------------------------------- pure-numpy helpers
def _znorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-8) if sd > 0 else np.zeros_like(x)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win <= 1 or x.size < win:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def _resample_to(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(n)
    if x.size == n:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)


def _estimate_k(duration_seconds: float) -> int:
    """Heuristic number of sections from song length (~18s per section, clamped 3..10)."""
    return int(np.clip(round(duration_seconds / 18.0), 3, 10))


def _snap_to_downbeats(bounds: list[int], downbeats: np.ndarray, tol_frames: int) -> list[int]:
    """Snap each interior boundary to the nearest downbeat within ``tol_frames``."""
    if downbeats.size == 0:
        return bounds
    snapped = [bounds[0]]
    for b in bounds[1:-1]:
        j = int(np.argmin(np.abs(downbeats - b)))
        snapped.append(int(downbeats[j]) if abs(int(downbeats[j]) - b) <= tol_frames else b)
    snapped.append(bounds[-1])
    return sorted(set(snapped))


def _merge_short(bounds: list[int], min_frames: int) -> list[int]:
    """Merge sections shorter than ``min_frames`` into a neighbour."""
    if len(bounds) <= 2:
        return bounds
    out = [bounds[0]]
    for b in bounds[1:]:
        if b - out[-1] < min_frames and b != bounds[-1]:
            continue  # drop this boundary -> merge forward
        out.append(b)
    # ensure the final section is long enough by absorbing it backwards
    if len(out) > 2 and out[-1] - out[-2] < min_frames:
        out.pop(-2)
    return out


def _label_sections(section_feats: np.ndarray, sim_thresh: float = 0.82) -> np.ndarray:
    """Greedy repetition labels from section mean-feature cosine similarity."""
    n = section_feats.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    norms = np.linalg.norm(section_feats, axis=1, keepdims=True) + 1e-8
    unit = section_feats / norms
    sim = unit @ unit.T
    labels = -np.ones(n, dtype=int)
    next_label = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = next_label
        for j in range(i + 1, n):
            if labels[j] < 0 and sim[i, j] >= sim_thresh:
                labels[j] = next_label
        next_label += 1
    return labels


def _recurrence(section_feats: np.ndarray) -> np.ndarray:
    if section_feats.shape[0] == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(section_feats, axis=1, keepdims=True) + 1e-8
    unit = section_feats / norms
    return (unit @ unit.T).astype(np.float32)


def _repeat_of_from_labels(labels: np.ndarray) -> list[int | None]:
    """For each section, the earliest earlier section sharing its label (else None)."""
    out: list[int | None] = []
    first_seen: dict[int, int] = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab in first_seen:
            out.append(first_seen[lab])
        else:
            out.append(None)
            first_seen[lab] = i
    return out


def _kmeans_np(X: np.ndarray, k: int, *, iters: int = 50, seed: int = 0) -> np.ndarray:
    """Tiny deterministic k-means (k-means++ init) for spectral clustering; no sklearn dependency."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = [int(rng.integers(n))]
    for _ in range(1, k):
        d2 = np.min(np.linalg.norm(X[:, None, :] - X[idx][None, :, :], axis=2) ** 2, axis=1)
        s = float(d2.sum())
        idx.append(int(rng.integers(n)) if s <= 1e-12 else int(rng.choice(n, p=d2 / s)))
    C = X[idx].astype(np.float64).copy()
    labels = -np.ones(n, dtype=int)
    for _ in range(iters):
        new = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2).argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                C[c] = X[m].mean(axis=0)
    return labels


def _spectral_labels(section_feats: np.ndarray, sim_thresh: float = 0.82,
                     max_k: int | None = None) -> np.ndarray:
    """Repetition labels via normalized-Laplacian spectral clustering (McFee & Ellis, ISMIR 2014).

    Builds a thresholded cosine-similarity affinity over section mean-features, forms the symmetric
    normalized Laplacian, picks k from the largest eigengap, and clusters the spectral embedding.
    Captures GLOBAL relationships (a returning chorus groups with the first even without local
    novelty between them). Deterministic; falls back to the greedy labeler for <= 2 sections.
    """
    n = int(section_feats.shape[0])
    if n <= 2:
        return _label_sections(section_feats, sim_thresh)
    norms = np.linalg.norm(section_feats, axis=1, keepdims=True) + 1e-8
    unit = section_feats / norms
    sim = unit @ unit.T
    A = np.where(sim >= sim_thresh, sim, 0.0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    if not np.any(A):
        return np.arange(n, dtype=int)  # nothing repeats -> all distinct
    dinv = 1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8))
    L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    evals = np.clip(np.linalg.eigvalsh(L), 0.0, None)
    upper = min(n, max_k or n)
    gaps = np.diff(evals[:upper])
    k = int(np.clip(int(np.argmax(gaps)) + 1 if gaps.size else 1, 1, n))
    if k <= 1:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    _, evecs = np.linalg.eigh(L)
    embed = evecs[:, :k]
    embed = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    return _kmeans_np(embed, k).astype(int)


def _infer_roles(energies: np.ndarray, labels: np.ndarray) -> tuple[list[str], int]:
    """Assign a role to each section from energy rank, position and recurrence."""
    e = np.asarray(energies, dtype=np.float64)
    n = len(e)
    if n == 0:
        return [], 0
    climax = int(np.argmax(e))
    counts = {int(l): int(np.sum(labels == l)) for l in np.unique(labels)}
    med = float(np.median(e))
    roles: list[str] = []
    for i in range(n):
        recurring = counts[int(labels[i])] > 1
        if i == climax:
            roles.append("chorus" if recurring else "drop")
        elif i == 0 and e[i] <= med:
            roles.append("intro")
        elif i == n - 1 and e[i] <= med:
            roles.append("outro")
        elif e[i] >= _HI:
            roles.append("chorus" if recurring else "drop")
        elif e[i] <= _LO:
            roles.append("intro" if i < n / 2 else "outro")
        else:
            roles.append("verse" if recurring else "bridge")
    return roles, climax


def _build_sections(bounds: list[int], energy_curve: np.ndarray,
                    section_feats: np.ndarray, tempo: float,
                    total_frames: int, used_fallback: bool, *,
                    spectral: bool = False) -> MusicStructure:
    n = len(bounds) - 1
    energies = np.array([
        float(np.mean(energy_curve[bounds[i]:bounds[i + 1]]))
        if bounds[i + 1] > bounds[i] else 0.0
        for i in range(n)
    ])
    energies_norm = _minmax(energies) if n > 1 else np.array([0.5] * n)
    if section_feats.shape[0] == n:
        labels = _spectral_labels(section_feats) if spectral else _label_sections(section_feats)
    else:
        labels = np.arange(n, dtype=int)
    roles, climax = _infer_roles(energies_norm, labels)
    repeat_of = _repeat_of_from_labels(labels)
    sections = [
        Section(
            start_frame=int(bounds[i]), end_frame=int(bounds[i + 1]),
            start_sec=bounds[i] / FPS, end_sec=bounds[i + 1] / FPS,
            label=int(labels[i]), role=roles[i], energy=float(energies_norm[i]),
            repeat_of=repeat_of[i],
        )
        for i in range(n)
    ]
    return MusicStructure(
        sections=sections, energy_curve=energy_curve.astype(np.float32),
        recurrence=_recurrence(section_feats), climax_index=climax,
        tempo=float(tempo), total_frames=int(total_frames), used_fallback=used_fallback,
    )


# --------------------------------------------------------------------------- fallback
def _fallback_structure(metadata: "SongMetadata", total_frames: int,
                        min_section_seconds: float,
                        energy_curve: np.ndarray | None = None) -> MusicStructure:
    """Downbeat/uniform sections when librosa segmentation is unavailable or fails."""
    from agentlodge.dance.hybrid import segment_boundaries

    bounds = segment_boundaries(metadata, total_frames, min_seg_seconds=min_section_seconds)
    if energy_curve is None or energy_curve.size != total_frames:
        energy_curve = np.full(total_frames, 0.5, dtype=np.float32)
    n = len(bounds) - 1
    # Position-only pseudo features so same-label detection stays inert in the fallback.
    feats = np.eye(max(n, 1), dtype=np.float32)[:n]
    struct = _build_sections(bounds, energy_curve, feats, float(getattr(metadata, "bpm", 0.0)),
                             total_frames, used_fallback=True)
    logger.info("Structure analysis fallback: %d downbeat/uniform sections", n)
    return struct


# --------------------------------------------------------------------------- public API
def analyze_structure(wav_path: str | Path, metadata: "SongMetadata", total_frames: int,
                      *, min_section_seconds: float = 8.0, k: int | None = None,
                      spectral: bool = False) -> MusicStructure:
    """Analyze a song into a :class:`MusicStructure` (sections + energy arc).

    ``total_frames`` is the target motion length (30 FPS); boundaries/energy are expressed in that
    frame space. ``spectral`` uses normalized-Laplacian spectral clustering (McFee & Ellis) for
    section-type labels instead of the greedy cosine labeler. Falls back to downbeat/uniform
    sections on any failure.
    """
    min_frames = max(int(min_section_seconds * FPS), 1)
    beats = np.asarray(getattr(metadata, "beat_frames", []), dtype=np.int64)
    beats = beats[(beats > 0) & (beats < total_frames)]
    downbeats = beats[::4] if beats.size else np.array([], dtype=np.int64)

    try:
        import librosa

        y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=13)
        feat = np.vstack([
            chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8),
            (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8),
        ])
        feat_len = feat.shape[1]
        if feat_len < 4:
            raise ValueError("audio too short for structure analysis")

        # Energy arc: RMS + spectral flux, smoothed, normalized, resampled to motion frames.
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        raw_energy = 0.5 * _znorm(rms) + 0.5 * _znorm(_resample_to(flux, rms.size))
        energy_feat = _smooth(raw_energy, win=FPS)  # ~1s smoothing at feature rate (==FPS)
        energy_curve = _minmax(_resample_to(energy_feat, total_frames)).astype(np.float32)

        n_sections = k or _estimate_k(getattr(metadata, "duration_seconds", feat_len / FPS))
        n_sections = int(np.clip(n_sections, 2, max(2, feat_len // max(min_frames, 1))))
        feat_bounds = librosa.segment.agglomerative(feat, n_sections)
        # feature frame -> motion frame (both ~30 FPS but lengths may differ slightly)
        scale = total_frames / feat_len
        bounds = sorted({0, total_frames} | {int(round(b * scale)) for b in feat_bounds})
        bounds = [b for b in bounds if 0 <= b <= total_frames]
        if bounds[0] != 0:
            bounds = [0] + bounds
        if bounds[-1] != total_frames:
            bounds = bounds + [total_frames]

        bounds = _snap_to_downbeats(bounds, downbeats, tol_frames=FPS // 2)
        bounds = _merge_short(bounds, min_frames)
        if len(bounds) < 2:
            raise ValueError("segmentation produced no usable sections")

        # Per-section mean feature vector (in feature space) for repetition labels.
        section_feats = np.array([
            feat[:, int(bounds[i] / scale):max(int(bounds[i] / scale) + 1, int(bounds[i + 1] / scale))].mean(axis=1)
            for i in range(len(bounds) - 1)
        ])
        tempo = float(getattr(metadata, "bpm", 0.0))
        struct = _build_sections(bounds, energy_curve, section_feats, tempo,
                                 total_frames, used_fallback=False, spectral=spectral)
        logger.info("Structure analysis: %d sections, climax@%d, roles=%s",
                    len(struct.sections), struct.climax_index,
                    [s.role for s in struct.sections])
        return struct
    except Exception as exc:  # noqa: BLE001 - robust fallback on any analysis failure
        logger.warning("Structure analysis failed (%s); using downbeat/uniform fallback", exc)
        energy_curve = None
        try:
            energy_curve = locals().get("energy_curve")  # reuse if we got that far
        except Exception:
            energy_curve = None
        return _fallback_structure(metadata, total_frames, min_section_seconds, energy_curve)
