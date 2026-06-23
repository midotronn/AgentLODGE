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
