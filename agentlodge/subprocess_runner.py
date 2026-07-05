"""Run heavy EDGE/Jukebox and Lodge work in isolated child processes."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from agentlodge.env_paths import resolve_venv_python

logger = logging.getLogger(__name__)
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
AGENTLODGE_ROOT = SCRIPTS_DIR.parent


def _venv_python(code_path: Path, label: str) -> Path:
    python = resolve_venv_python(code_path)
    if python is None:
        raise FileNotFoundError(
            f"{label} virtualenv not found at {code_path / '.venv'}. "
            f"Create it in the {label} repo before running the pipeline."
        )
    return python


def _run(cmd: list[str], *, timeout_seconds: int | None, step: str) -> subprocess.CompletedProcess:
    logger.info("Running %s subprocess: %s", step, Path(cmd[0]).name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.debug(result.stderr.strip())
    return result


def _raise_on_failure(result: subprocess.CompletedProcess, step: str) -> None:
    if result.returncode == 0:
        return
    if result.returncode in {-9, 137}:
        raise RuntimeError(
            f"{step} was killed (likely out of memory). "
            "Close other apps, use a shorter clip, or pre-cache Jukebox features on a GPU machine."
        )
    raise RuntimeError(
        f"{step} failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def run_lodge_inference_subprocess(
    lodge_code_path: Path,
    lodge_weights_path: Path,
    lodge_global_weights_path: Path,
    lodge_genre: str,
    features_npy: Path,
    output_npy: Path,
    work_dir: Path,
    *,
    timeout_seconds: int | None = None,
) -> str:
    python = _venv_python(lodge_code_path, "LODGE")
    cmd = [
        str(python),
        str(SCRIPTS_DIR / "run_lodge_inference.py"),
        "--agentlodge-root",
        str(AGENTLODGE_ROOT),
        "--features-npy",
        str(features_npy),
        "--output-npy",
        str(output_npy),
        "--work-dir",
        str(work_dir),
        "--lodge-code-path",
        str(lodge_code_path),
        "--lodge-weights-path",
        str(lodge_weights_path),
        "--lodge-global-weights-path",
        str(lodge_global_weights_path),
        "--lodge-genre",
        lodge_genre,
    ]
    result = _run(cmd, timeout_seconds=timeout_seconds, step="LODGE")
    _raise_on_failure(result, "LODGE inference")
    summary_path = work_dir / "lodge_summary.txt"
    return summary_path.read_text() if summary_path.exists() else "LODGE generation completed."


def run_jukebox_extraction(
    edge_code_path: Path,
    slice_dir: Path,
    cache_dir: Path,
    *,
    timeout_seconds: int | None = None,
) -> None:
    python = _venv_python(edge_code_path, "EDGE")
    cmd = [
        str(python),
        str(SCRIPTS_DIR / "jukebox_extract_all.py"),
        "--edge-root",
        str(edge_code_path),
        "--slice-dir",
        str(slice_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    result = _run(cmd, timeout_seconds=timeout_seconds, step="Jukebox")
    _raise_on_failure(result, "Jukebox extraction")


def run_edge_inference_subprocess(
    edge_code_path: Path,
    checkpoint: Path,
    work_dir: Path,
    features_npy: Path,
    output_npy: Path,
    *,
    timeout_seconds: int | None = None,
) -> None:
    python = _venv_python(edge_code_path, "EDGE")
    cmd = [
        str(python),
        str(SCRIPTS_DIR / "run_edge_inference.py"),
        "--edge-root",
        str(edge_code_path),
        "--checkpoint",
        str(checkpoint),
        "--work-dir",
        str(work_dir),
        "--features-npy",
        str(features_npy),
        "--output-npy",
        str(output_npy),
    ]
    result = _run(cmd, timeout_seconds=timeout_seconds, step="EDGE")
    _raise_on_failure(result, "EDGE inference")


def run_stick_video_subprocess(
    lodge_code_path: Path,
    motion_npy: Path,
    output_mp4: Path,
    *,
    audio_path: Path | None = None,
    smpl_joint_path: Path | None = None,
    timeout_seconds: int | None = None,
) -> None:
    python = _venv_python(lodge_code_path, "LODGE")
    cmd = [
        str(python),
        str(SCRIPTS_DIR / "render_stick_video.py"),
        "--agentlodge-root",
        str(AGENTLODGE_ROOT),
        "--motion-npy",
        str(motion_npy),
        "--output-mp4",
        str(output_mp4),
        "--lodge-code-path",
        str(lodge_code_path),
    ]
    if audio_path is not None:
        cmd.extend(["--audio", str(audio_path)])
    if smpl_joint_path is not None:
        cmd.extend(["--smpl-joint-path", str(smpl_joint_path)])
    result = _run(cmd, timeout_seconds=timeout_seconds, step="stick figure video")
    _raise_on_failure(result, "Stick figure video render")


def run_blender_video_subprocess(
    lodge_code_path: Path,
    motion_npy: Path,
    output_mp4: Path,
    blender_bin: Path,
    smplx_model_dir: Path,
    *,
    audio_path: Path | None = None,
    uv_npz: Path | None = None,
    texture_png: Path | None = None,
    width: int = 900,
    height: int = 900,
    samples: int = 32,
    timeout_seconds: int | None = None,
) -> None:
    """Render the dance as a shaded SMPL-X mesh in Blender (needs the licensed model)."""
    python = _venv_python(lodge_code_path, "LODGE")
    cmd = [
        str(python),
        str(SCRIPTS_DIR / "render_blender_dance.py"),
        "--agentlodge-root", str(AGENTLODGE_ROOT),
        "--motion-npy", str(motion_npy),
        "--output-mp4", str(output_mp4),
        "--lodge-code-path", str(lodge_code_path),
        "--smplx-model-dir", str(smplx_model_dir),
        "--blender-bin", str(blender_bin),
        "--blender-script", str(SCRIPTS_DIR / "blender_render_smplx.py"),
        "--width", str(width), "--height", str(height), "--samples", str(samples),
    ]
    if audio_path is not None:
        cmd.extend(["--audio", str(audio_path)])
    if uv_npz is not None:
        cmd.extend(["--uv-npz", str(uv_npz)])
    if texture_png is not None:
        cmd.extend(["--texture", str(texture_png)])
    result = _run(cmd, timeout_seconds=timeout_seconds, step="Blender mesh video")
    _raise_on_failure(result, "Blender mesh video render")

