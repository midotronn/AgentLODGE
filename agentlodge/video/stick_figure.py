"""Render FineDance / LODGE motion arrays as stick-figure videos."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap

from agentlodge.config import FPS
from agentlodge.dance.format import to_native_finedance139
from agentlodge.env_paths import lodge_import_paths, use_code_paths

logger = logging.getLogger(__name__)

# 22 body joints from FineDance / LODGE (SMPLX body, no hands)
BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def default_smpl_joint_path(lodge_code_path: Path) -> Path:
    return lodge_code_path / "data" / "smplx_neu_J_1.npy"


def _orient_z_up(joints: np.ndarray) -> np.ndarray:
    """Reorient FK joints (L, 22, 3) so the human's vertical axis maps to matplotlib z.

    LODGE (FineDance) FK output is Y-up, while EDGE-derived motion run through the same
    FK comes out Z-up, so a fixed axis swap mis-orients one of them (the figure ends up
    lying down). Detect the vertical axis model-agnostically as the one with the largest
    median per-frame extent (a standing/dancing human is tallest along "up"), move it to
    z, and flip if the head ends up below the feet.
    """
    extent = np.median(joints.max(axis=1) - joints.min(axis=1), axis=0)
    up = int(np.argmax(extent))
    horiz = [a for a in range(3) if a != up]
    oriented = joints[..., [horiz[0], horiz[1], up]]
    head_z = float(np.mean(oriented[:, 15, 2]))  # SMPL head joint
    feet_z = float(np.mean(oriented[:, [7, 8, 10, 11], 2]))  # ankles + feet
    if head_z < feet_z:
        oriented = oriented.copy()
        oriented[..., 2] = -oriented[..., 2]
    return oriented


def joints_from_motion139(
    motion: np.ndarray,
    *,
    lodge_code_path: Path,
    smpl_joint_path: Path | None = None,
) -> np.ndarray:
    """Forward kinematics from a (L, 139) motion array to (L, 22, 3) joints (z-up)."""
    jpath = smpl_joint_path or default_smpl_joint_path(lodge_code_path)
    if not jpath.exists():
        raise FileNotFoundError(
            f"SMPL joint regressor not found at {jpath}. "
            "Download or symlink LODGE data/smplx_neu_J_1.npy."
        )

    native = to_native_finedance139(motion)
    trans = torch.from_numpy(native[:, 4:7]).float()
    rot6d = torch.from_numpy(native[:, 7:139]).float().view(-1, 22, 6)

    with use_code_paths(*lodge_import_paths(lodge_code_path)):
        from dld.data.render_joints.smplfk import SMPLX_Skeleton, ax_from_6v

        poses = ax_from_6v(rot6d).reshape(-1, 66)
        fk = SMPLX_Skeleton(device="cpu", batch=1, Jpath=str(jpath))
        joints = fk.forward(poses, trans).detach().cpu().numpy()[:, :22]
    # Model-agnostic: put the human's vertical axis on matplotlib's z regardless of
    # whether the FK output is Y-up (LODGE) or Z-up (EDGE-derived motion).
    return _orient_z_up(joints)


def _framing(joints: np.ndarray) -> tuple[float, float, float, float]:
    """Center the figure and stand it on the floor, sized to fill the frame.

    Returns (center_x, center_y, floor_z, radius). Derived from the full motion so the
    dancer stays centered and grounded regardless of the model's root translation.
    """
    xs, ys, zs = joints[..., 0], joints[..., 1], joints[..., 2]
    cx = 0.5 * (float(xs.min()) + float(xs.max()))
    cy = 0.5 * (float(ys.min()) + float(ys.max()))
    # Floor at a robust low percentile of the per-frame minimum foot height, so the
    # dancer rests on the ground for most frames without a single deep frame dropping
    # the floor far below the feet.
    per_frame_foot_min = joints[:, (7, 8, 10, 11), 2].min(axis=1)
    floor = float(np.percentile(per_frame_foot_min, 5))
    span = max(
        float(xs.max() - xs.min()),
        float(ys.max() - ys.min()),
        float(zs.max() - zs.min()),
    )
    radius = max(span, 1e-3) * 0.5 * 1.15  # cubic half-extent with a margin
    return cx, cy, floor, radius


def _set_line_data_3d(line, x: np.ndarray) -> None:
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def _set_scatter_data_3d(scat, x: np.ndarray, color: str) -> None:
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([color])


def _plot_pose(num, poses, lines, ax, scat, contact, parents) -> None:
    """Update a single animation frame without resetting the (dynamic) axis limits."""
    pose = poses[num]
    static = contact[num]
    for i, (point, idx) in enumerate(zip(scat, (7, 8, 10, 11))):
        _set_scatter_data_3d(point, pose[idx : idx + 1], "r" if static[i] else "g")
    for i, (parent, line) in enumerate(zip(parents, lines)):
        if i == 0:
            continue
        _set_line_data_3d(line, np.stack((pose[i], pose[parent]), axis=0))


def render_stick_figure_video(
    joints: np.ndarray,
    output_mp4: Path,
    *,
    lodge_code_path: Path,
    audio_path: Path | None = None,
    fps: int = FPS,
) -> Path:
    """Animate joint positions to an mp4, optionally muxed with audio."""
    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    num_steps = joints.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    cx, cy, floor, radius = _framing(joints)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(floor, floor + 2 * radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-70)

    xx, yy = np.meshgrid(
        np.linspace(cx - radius, cx + radius, 2),
        np.linspace(cy - radius, cy + radius, 2),
    )
    ax.plot_surface(xx, yy, np.full_like(xx, floor), zorder=-11, cmap=cm.twilight)

    lines = [ax.plot([], [], [], zorder=10, linewidth=1.5)[0] for _ in BODY_PARENTS]
    scat = [
        ax.scatter([], [], [], zorder=10, s=8, cmap=ListedColormap(["r", "g", "b"]))
        for _ in range(4)
    ]
    feet = joints[:, (7, 8, 10, 11)]
    feetv = np.zeros(feet.shape[:2])
    feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    contact = feetv < 0.01

    anim = animation.FuncAnimation(
        fig,
        _plot_pose,
        num_steps,
        fargs=(joints, lines, ax, scat, contact, BODY_PARENTS),
        interval=1000 // fps,
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for stick-figure video export but was not found on PATH.")
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg

    with TemporaryDirectory() as temp_dir:
        # Encode frames straight to H.264 mp4 (piped to ffmpeg) instead of writing a
        # GIF first: the GIF -> h264 path is pathologically slow / can hang.
        needs_audio = audio_path is not None and audio_path.exists()
        video_only = Path(temp_dir) / "render.mp4" if needs_audio else output_mp4
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.save(str(video_only), writer=writer)
        plt.close(fig)

        if needs_audio:
            # Mux the pre-encoded video with audio using a stream copy (fast).
            cmd = [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(audio_path),
                "-shortest",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-q:a",
                "4",
                str(output_mp4),
            ]
            logger.info("Muxing audio into %s", output_mp4.name)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg audio mux failed (exit {result.returncode}).\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )

    return output_mp4


def render_motion_npy_to_video(
    motion_npy: Path,
    output_mp4: Path,
    *,
    lodge_code_path: Path,
    audio_path: Path | None = None,
    smpl_joint_path: Path | None = None,
    fps: int = FPS,
) -> Path:
    """Load a (L, 139) motion .npy and write a stick-figure mp4."""
    motion = np.load(motion_npy)
    joints = joints_from_motion139(
        motion,
        lodge_code_path=lodge_code_path,
        smpl_joint_path=smpl_joint_path,
    )
    logger.info("Rendering stick figure video (%d frames)", joints.shape[0])
    return render_stick_figure_video(
        joints,
        output_mp4,
        lodge_code_path=lodge_code_path,
        audio_path=audio_path,
        fps=fps,
    )
