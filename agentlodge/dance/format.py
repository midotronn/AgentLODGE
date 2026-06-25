"""Motion format helpers."""

from __future__ import annotations

import numpy as np


def edge_to_lodge139(motion: np.ndarray) -> np.ndarray:
    """Convert EDGE (L, 151) representation to Lodge (L, 139)."""
    if motion.shape[-1] != 151:
        raise ValueError(f"Expected EDGE motion with 151 dims, got {motion.shape[-1]}")
    trans = motion[:, :3]
    rot22 = motion[:, 3 : 3 + 22 * 6]
    contact = motion[:, 147:151]
    return np.concatenate([trans, rot22, contact], axis=-1).astype(np.float32)


def ensure_lodge139(motion: np.ndarray) -> np.ndarray:
    if motion.shape[-1] == 139:
        return motion.astype(np.float32)
    if motion.shape[-1] == 151:
        return edge_to_lodge139(motion)
    raise ValueError(f"Unsupported motion dimension: {motion.shape[-1]}")


def _looks_like_contact(values: np.ndarray) -> bool:
    return float(np.mean((values >= -0.01) & (values <= 1.01))) > 0.9


def to_native_finedance139(motion: np.ndarray) -> np.ndarray:
    """Convert AgentLODGE layout to native FineDance 139-dim layout for FK/rendering.

    Native layout: contact (4) + root translation (3) + 22-joint 6D rotation (132).
    AgentLODGE layout: root translation (3) + rotation (132) + contact (4).
    """
    if motion.shape[-1] != 139:
        raise ValueError(f"Expected motion with 139 dims, got {motion.shape[-1]}")
    motion = motion.astype(np.float32)
    start_contact = _looks_like_contact(motion[:, :4])
    end_contact = _looks_like_contact(motion[:, 135:139])
    if end_contact and not start_contact:
        return np.concatenate(
            [motion[:, 135:139], motion[:, :3], motion[:, 3:135]],
            axis=1,
        )
    return motion
