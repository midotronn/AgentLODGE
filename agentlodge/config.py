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


def _path(value: str | None, default: str) -> Path:
    raw = value or default
    return Path(raw).expanduser().resolve()


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
        )

    def validate_image_backend(self) -> None:
        if self.image_backend == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when IMAGE_BACKEND=openai")
        if self.image_backend == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when IMAGE_BACKEND=gemini")
        if self.image_backend not in {"openai", "gemini"}:
            raise ValueError("IMAGE_BACKEND must be 'openai' or 'gemini'")
