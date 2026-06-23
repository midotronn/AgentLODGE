"""Audio preprocessing for Lodge (librosa) and EDGE (Jukebox)."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from agentlodge.config import FPS, Settings

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
    sys.path.insert(0, str(lodge_code_path))
    try:
        from dld.data.utils.audio import extract as extract_music35
    finally:
        if str(lodge_code_path) in sys.path:
            sys.path.remove(str(lodge_code_path))

    features, _ = extract_music35(str(wav_path))
    return np.asarray(features, dtype=np.float32)


def extract_edge_slices(
    wav_path: Path, edge_code_path: Path, work_dir: Path
) -> list[np.ndarray]:
    """Slice audio and extract Jukebox embeddings for EDGE long-form generation."""
    sys.path.insert(0, str(edge_code_path))
    try:
        from data.slice import slice_audio
        from data.audio_extraction.jukebox_features import extract as juke_extract
    finally:
        if str(edge_code_path) in sys.path:
            sys.path.remove(str(edge_code_path))

    slice_dir = work_dir / "edge_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    slice_audio(str(wav_path), stride=2.5, length=5.0, out_dir=str(slice_dir))

    wav_slices = sorted(slice_dir.glob("*.wav"), key=_slice_key)
    if not wav_slices:
        raise ValueError("EDGE preprocessing produced no audio slices")

    features: list[np.ndarray] = []
    for wav_slice in wav_slices:
        reps, _ = juke_extract(str(wav_slice))
        features.append(np.asarray(reps, dtype=np.float32))
    return features


def _slice_key(path: Path) -> int:
    stem = path.stem
    return int(stem.split("slice")[-1])


def preprocess_audio(
    audio_path: str | Path,
    settings: Settings,
    work_dir: Path,
    *,
    extract_edge: bool = True,
) -> PreprocessedAudio:
    wav_path = ensure_wav(audio_path)
    metadata = extract_song_metadata(wav_path)
    lodge_features = extract_lodge_features(wav_path, settings.lodge_code_path)

    edge_slices = None
    if extract_edge:
        edge_slices = extract_edge_slices(wav_path, settings.edge_code_path, work_dir)

    return PreprocessedAudio(
        wav_path=wav_path,
        metadata=metadata,
        lodge_features=lodge_features,
        edge_feature_slices=edge_slices,
    )
