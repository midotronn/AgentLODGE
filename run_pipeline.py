#!/usr/bin/env python3
"""AgentLODGE pipeline entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from agentlodge.config import Settings
from agentlodge.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agentic dance and costume generation pipeline"
    )
    parser.add_argument("--audio", required=True, help="Path to input .wav or .mp3")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to OUTPUT_DIR env var or ./outputs)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip stick-figure video rendering",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    settings = Settings.from_env()
    try:
        result = run_pipeline(
            audio_path=args.audio,
            output_dir=args.output_dir,
            settings=settings,
            render_video=not args.skip_video,
        )
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    print(f"Selected model: {result.selected_model}")
    print(f"Reasoning: {result.selection_reasoning}")
    if result.costume_description:
        print(f"Costume (from audio): {result.costume_description}")
    print(f"Outputs saved to: {result.output_dir}")
    if result.stick_figure_video:
        print(f"Stick figure video: {result.stick_figure_video}")
    if result.errors:
        print("Warnings/errors logged in pipeline_log.json:")
        for err in result.errors:
            print(f"  - {err.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
