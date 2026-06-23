"""End-to-end AgentLODGE pipeline orchestration."""

from __future__ import annotations

import json
import logging
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentlodge.agent.selector import SelectionResult, select_dance
from agentlodge.audio.preprocess import PreprocessedAudio, preprocess_audio
from agentlodge.config import Settings
from agentlodge.dance.edge import EdgeResult, generate_edge_dance
from agentlodge.dance.format import ensure_lodge139
from agentlodge.dance.lodge import LodgeResult, generate_lodge_dance
from agentlodge.dance.metrics import DanceMetrics, compute_metrics
from agentlodge.image.costume import generate_costume_image

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutcome:
    motion: np.ndarray | None = None
    summary: str = ""
    error: str | None = None


@dataclass
class PipelineResult:
    output_dir: Path
    selected_model: str
    selection_reasoning: str
    lodge_metrics: DanceMetrics | None = None
    edge_metrics: DanceMetrics | None = None
    song_duration_seconds: float = 0.0
    costume_description: str = ""
    errors: list[str] = field(default_factory=list)


def _run_lodge_job(
    lodge_features: np.ndarray,
    settings_dict: dict,
    work_dir: str,
) -> dict:
    settings = Settings.from_dict(settings_dict)
    try:
        result = generate_lodge_dance(
            lodge_features,
            settings,
            Path(work_dir),
        )
        return {"motion": result.motion, "summary": result.summary, "error": None}
    except Exception as exc:
        return {
            "motion": None,
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _run_edge_job(
    wav_path: str,
    edge_slices: list,
    settings_dict: dict,
    work_dir: str,
) -> dict:
    settings = Settings.from_dict(settings_dict)
    try:
        result = generate_edge_dance(
            Path(wav_path),
            edge_slices,
            settings,
            Path(work_dir),
        )
        return {"motion": result.motion, "summary": result.summary, "error": None}
    except Exception as exc:
        return {
            "motion": None,
            "summary": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _settings_to_dict(settings: Settings) -> dict:
    return {
        "anthropic_api_key": settings.anthropic_api_key,
        "openai_api_key": settings.openai_api_key,
        "gemini_api_key": settings.gemini_api_key,
        "image_backend": settings.image_backend,
        "output_dir": str(settings.output_dir),
        "lodge_code_path": str(settings.lodge_code_path),
        "edge_code_path": str(settings.edge_code_path),
        "lodge_weights_path": str(settings.lodge_weights_path),
        "lodge_global_weights_path": str(settings.lodge_global_weights_path),
        "edge_weights_path": str(settings.edge_weights_path),
        "lodge_genre": settings.lodge_genre,
    }


def run_pipeline(
    audio_path: str | Path,
    costume_description: str,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> PipelineResult:
    settings = settings or Settings.from_env()
    out_dir = Path(output_dir or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    work_root = Path(tempfile.mkdtemp(prefix="agentlodge_", dir=out_dir))

    logger.info("Preprocessing audio...")
    preprocessed = preprocess_audio(audio_path, settings, work_root)

    settings_dict = _settings_to_dict(settings)
    lodge_out = GenerationOutcome()
    edge_out = GenerationOutcome()

    logger.info("Running Lodge++ and EDGE generation in parallel...")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _run_lodge_job,
                preprocessed.lodge_features,
                settings_dict,
                str(work_root / "lodge"),
            ): "lodge",
            executor.submit(
                _run_edge_job,
                str(preprocessed.wav_path),
                preprocessed.edge_feature_slices or [],
                settings_dict,
                str(work_root / "edge"),
            ): "edge",
        }
        for future in as_completed(futures):
            name = futures[future]
            payload = future.result()
            outcome = lodge_out if name == "lodge" else edge_out
            outcome.motion = payload["motion"]
            outcome.summary = payload["summary"]
            outcome.error = payload["error"]
            if payload["error"]:
                errors.append(f"{name} generation failed: {payload['error']}")
                logger.error("%s generation failed: %s", name, payload["error"])

    if lodge_out.motion is None and edge_out.motion is None:
        raise RuntimeError(
            "Both Lodge++ and EDGE generation failed.\n"
            + "\n".join(errors)
        )

    lodge_metrics = None
    edge_metrics = None
    if lodge_out.motion is not None:
        lodge_metrics = compute_metrics(
            lodge_out.motion,
            preprocessed.metadata,
            "lodge",
            lodge_out.summary,
        )
    if edge_out.motion is not None:
        edge_metrics = compute_metrics(
            edge_out.motion,
            preprocessed.metadata,
            "edge",
            edge_out.summary,
        )

    selected_model = "lodge"
    reasoning = ""
    if lodge_metrics and edge_metrics:
        selection = select_dance(
            lodge_metrics,
            edge_metrics,
            preprocessed.metadata,
            settings.anthropic_api_key,
        )
        selected_model = selection.selected_model
        reasoning = selection.reasoning
        if selection.used_fallback:
            errors.append(f"selection agent fallback: {reasoning}")
    elif lodge_out.motion is None:
        selected_model = "edge"
        reasoning = "Lodge++ failed; automatically selected EDGE output."
    elif edge_out.motion is None:
        selected_model = "lodge"
        reasoning = "EDGE failed; automatically selected Lodge++ output."

    selected_motion = lodge_out.motion if selected_model == "lodge" else edge_out.motion
    if selected_motion is None:
        selected_motion = edge_out.motion if lodge_out.motion is None else lodge_out.motion
        selected_model = "edge" if lodge_out.motion is None else "lodge"

    selected_139 = ensure_lodge139(selected_motion)
    np.save(out_dir / "selected_dance.npy", selected_139)

    costume_path = out_dir / "costume_output.png"
    try:
        generate_costume_image(costume_description, costume_path, settings)
        logger.info("Saved costume image to %s", costume_path)
    except Exception as exc:
        msg = f"Costume image generation failed: {exc}"
        errors.append(msg)
        logger.error(msg)

    log_payload = {
        "selected_model": selected_model,
        "selection_reasoning": reasoning,
        "lodge_bas": lodge_metrics.beat_alignment_score if lodge_metrics else None,
        "edge_bas": edge_metrics.beat_alignment_score if edge_metrics else None,
        "lodge_diversity": lodge_metrics.motion_diversity if lodge_metrics else None,
        "edge_diversity": edge_metrics.motion_diversity if edge_metrics else None,
        "song_duration_seconds": preprocessed.metadata.duration_seconds,
        "costume_description": costume_description,
        "errors": errors,
    }
    (out_dir / "pipeline_log.json").write_text(json.dumps(log_payload, indent=2))

    return PipelineResult(
        output_dir=out_dir,
        selected_model=selected_model,
        selection_reasoning=reasoning,
        lodge_metrics=lodge_metrics,
        edge_metrics=edge_metrics,
        song_duration_seconds=preprocessed.metadata.duration_seconds,
        costume_description=costume_description,
        errors=errors,
    )
