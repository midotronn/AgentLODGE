"""Environment-backed configuration for the AgentLODGE pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FPS = 30
HOP_LENGTH = 512
SAMPLE_RATE = FPS * HOP_LENGTH


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def _path(value: str | None, default: str) -> Path:
    raw = value or default
    return Path(raw).expanduser().resolve()


def _opt_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    image_backend: str
    output_dir: Path
    lodge_code_path: Path
    edge_code_path: Path
    lodge_weights_path: Path
    lodge_global_weights_path: Path
    edge_weights_path: Path
    lodge_genre: str
    min_audio_seconds: float
    max_edge_slices: int | None
    render_backend: str = "stick"
    render_character: str = "smplx"
    blender_bin: Path | None = None
    smplx_model_dir: Path | None = None
    smplx_uv_path: Path | None = None
    smplx_texture_path: Path | None = None
    ybot_fbx_path: Path | None = None
    render_color: str = "0.5,0.5,0.52"
    hybrid_enabled: bool = False
    hybrid_min_seg_seconds: float = 8.5
    hybrid_blend_frames: int = 15
    hybrid_scheduler: str = "metric"

    @classmethod
    def from_env(cls) -> "Settings":
        lodge_weights = _path(
            os.getenv("LODGE_WEIGHTS_PATH"),
            "../Runs/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
        )
        lodge_root = lodge_weights.parent.parent.parent.parent
        global_default = lodge_root / "Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt"

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            image_backend=os.getenv("IMAGE_BACKEND", "openai").lower(),
            output_dir=_path(os.getenv("OUTPUT_DIR"), "./outputs"),
            lodge_code_path=_path(os.getenv("LODGE_CODE_PATH"), "../Runs/LODGE"),
            edge_code_path=_path(os.getenv("EDGE_CODE_PATH"), "../Runs/EDGE"),
            lodge_weights_path=lodge_weights,
            lodge_global_weights_path=_path(
                os.getenv("LODGE_GLOBAL_WEIGHTS_PATH"), str(global_default)
            ),
            edge_weights_path=_path(
                os.getenv("EDGE_WEIGHTS_PATH"), "../Runs/EDGE/checkpoint.pt"
            ),
            lodge_genre=os.getenv("LODGE_GENRE", "Hiphop"),
            min_audio_seconds=float(os.getenv("AGENTLODGE_MIN_AUDIO_SECONDS", "20")),
            max_edge_slices=_optional_int(os.getenv("AGENTLODGE_MAX_EDGE_SLICES", "7")),
            render_backend=os.getenv("AGENTLODGE_RENDER_BACKEND", "stick").lower(),
            render_character=os.getenv("AGENTLODGE_RENDER_CHARACTER", "smplx").lower(),
            blender_bin=_opt_path(os.getenv("BLENDER_BIN")),
            smplx_model_dir=_opt_path(os.getenv("SMPLX_MODEL_DIR")),
            smplx_uv_path=_opt_path(os.getenv("SMPLX_UV_PATH")),
            smplx_texture_path=_opt_path(os.getenv("SMPLX_TEXTURE_PATH")),
            ybot_fbx_path=_opt_path(os.getenv("YBOT_FBX_PATH")),
            render_color=os.getenv("AGENTLODGE_RENDER_COLOR", "0.5,0.5,0.52"),
            hybrid_enabled=os.getenv("AGENTLODGE_HYBRID", "0").lower()
            in {"1", "true", "yes"},
            hybrid_min_seg_seconds=float(os.getenv("AGENTLODGE_HYBRID_MIN_SEG", "8.5")),
            hybrid_blend_frames=int(os.getenv("AGENTLODGE_HYBRID_BLEND", "15")),
            hybrid_scheduler=os.getenv("AGENTLODGE_HYBRID_SCHEDULER", "metric").lower(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return cls(
            anthropic_api_key=data.get("anthropic_api_key"),
            openai_api_key=data.get("openai_api_key"),
            gemini_api_key=data.get("gemini_api_key"),
            image_backend=data.get("image_backend", "openai"),
            output_dir=_path(data.get("output_dir"), "./outputs"),
            lodge_code_path=_path(data.get("lodge_code_path"), "../Runs/LODGE"),
            edge_code_path=_path(data.get("edge_code_path"), "../Runs/EDGE"),
            lodge_weights_path=_path(
                data.get("lodge_weights_path"),
                "../Runs/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
            ),
            lodge_global_weights_path=_path(
                data.get("lodge_global_weights_path"),
                "../Runs/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
            ),
            edge_weights_path=_path(
                data.get("edge_weights_path"), "../Runs/EDGE/checkpoint.pt"
            ),
            lodge_genre=data.get("lodge_genre", "Hiphop"),
            min_audio_seconds=float(data.get("min_audio_seconds", 20)),
            max_edge_slices=data.get("max_edge_slices"),
            render_backend=data.get("render_backend", "stick"),
            render_character=data.get("render_character", "smplx"),
            blender_bin=_opt_path(data.get("blender_bin")),
            smplx_model_dir=_opt_path(data.get("smplx_model_dir")),
            smplx_uv_path=_opt_path(data.get("smplx_uv_path")),
            smplx_texture_path=_opt_path(data.get("smplx_texture_path")),
            ybot_fbx_path=_opt_path(data.get("ybot_fbx_path")),
            render_color=data.get("render_color", "0.5,0.5,0.52"),
            hybrid_enabled=bool(data.get("hybrid_enabled", False)),
            hybrid_min_seg_seconds=float(data.get("hybrid_min_seg_seconds", 8.5)),
            hybrid_blend_frames=int(data.get("hybrid_blend_frames", 15)),
            hybrid_scheduler=data.get("hybrid_scheduler", "metric"),
        )

    def validate_image_backend(self) -> None:
        if self.image_backend == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when IMAGE_BACKEND=openai")
        if self.image_backend == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when IMAGE_BACKEND=gemini")
        if self.image_backend not in {"openai", "gemini"}:
            raise ValueError("IMAGE_BACKEND must be 'openai' or 'gemini'")
