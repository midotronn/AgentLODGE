"""Audio preprocessing for Lodge (librosa) and EDGE (Jukebox)."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from agentlodge.config import FPS, Settings
from agentlodge.env_paths import lodge_import_paths, use_code_paths
from agentlodge.subprocess_runner import (
    run_edge_inference_subprocess,
    run_jukebox_extraction,
    run_lodge_inference_subprocess,
)

logger = logging.getLogger(__name__)

HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH


@dataclass
class SongMetadata:
    duration_seconds: float
    bpm: float
    beat_frames: np.ndarray
    wav_path: Path


@dataclass
class PreprocessedAudio:
    wav_path: Path
    metadata: SongMetadata
    lodge_features: np.ndarray
    edge_feature_slices: list[np.ndarray] | None = None


def ensure_wav(audio_path: str | Path) -> Path:
    """Convert mp3/other formats to wav at the pipeline sample rate."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() == ".wav":
        return audio_path

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError("pydub is required to convert non-wav audio") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    segment = AudioSegment.from_file(str(audio_path))
    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    segment.export(str(out_path), format="wav")
    return out_path


def extract_song_metadata(wav_path: Path) -> SongMetadata:
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    duration = len(y) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(np.atleast_1d(tempo)[0])
    return SongMetadata(
        duration_seconds=float(duration),
        bpm=bpm,
        beat_frames=np.asarray(beat_frames, dtype=np.int64),
        wav_path=wav_path,
    )


def extract_lodge_features(wav_path: Path, lodge_code_path: Path) -> np.ndarray:
    """Extract 35-dim librosa features at 30 FPS for the full song."""
    with use_code_paths(lodge_code_path):
        from dld.data.utils.audio import extract as extract_music35

    features, _ = extract_music35(str(wav_path))
    return np.asarray(features, dtype=np.float32)


def slice_audio(wav_path: Path, stride: float, length: float, out_dir: Path) -> int:
    audio, sr = librosa.load(str(wav_path), sr=None)
    file_name = wav_path.stem
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(out_dir / f"{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def extract_edge_slices(
    wav_path: Path,
    edge_code_path: Path,
    work_dir: Path,
    *,
    max_slices: int | None = None,
) -> list[np.ndarray]:
    """Slice audio and extract Jukebox embeddings via an isolated EDGE subprocess."""
    slice_dir = (work_dir / "edge_slices").resolve()
    count = slice_audio(wav_path.resolve(), stride=2.5, length=5.0, out_dir=slice_dir)
    logger.info("Created %d EDGE audio slices in %s", count, slice_dir)

    wav_slices = sorted(slice_dir.glob("*.wav"), key=_slice_key)
    if not wav_slices:
        raise ValueError("EDGE preprocessing produced no audio slices")

    if max_slices is not None and len(wav_slices) > max_slices:
        logger.warning(
            "Limiting EDGE slices from %d to %d to reduce memory use",
            len(wav_slices),
            max_slices,
        )
        extra = slice_dir / "unused_slices"
        extra.mkdir(exist_ok=True)
        for wav_slice in wav_slices[max_slices:]:
            wav_slice.rename(extra / wav_slice.name)
        wav_slices = wav_slices[:max_slices]

    cache_dir = work_dir / "edge_juke_cache"
    pending = [
        cache_dir / f"{wav_slice.stem}.npy"
        for wav_slice in wav_slices
        if not (cache_dir / f"{wav_slice.stem}.npy").exists()
    ]
    if pending:
        run_jukebox_extraction(
            edge_code_path.resolve(), slice_dir.resolve(), cache_dir.resolve()
        )

    features: list[np.ndarray] = []
    for wav_slice in wav_slices:
        cache_path = cache_dir / f"{wav_slice.stem}.npy"
        if not cache_path.exists():
            raise RuntimeError(f"Missing Jukebox cache for {wav_slice.name}")
        features.append(np.load(cache_path).astype(np.float32))
    return features


def release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _slice_key(path: Path) -> int:
    stem = path.stem
    return int(stem.split("slice")[-1])


def preprocess_audio(
    audio_path: str | Path,
    settings: Settings,
    work_dir: Path,
    *,
    extract_lodge: bool = True,
    extract_edge: bool = True,
) -> PreprocessedAudio:
    wav_path = ensure_wav(audio_path)
    metadata = extract_song_metadata(wav_path)

    lodge_features = np.empty((0, 35), dtype=np.float32)
    if extract_lodge:
        lodge_features = extract_lodge_features(wav_path, settings.lodge_code_path)

    edge_slices = None
    if extract_edge:
        max_slices = settings.max_edge_slices
        edge_slices = extract_edge_slices(
            wav_path, settings.edge_code_path, work_dir, max_slices=max_slices
        )

    return PreprocessedAudio(
        wav_path=wav_path,
        metadata=metadata,
        lodge_features=lodge_features,
        edge_feature_slices=edge_slices,
    )
