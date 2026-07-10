"""Training-free motion transitions for the hybrid LODGE+EDGE pipeline.

The hybrid stitches time-segments taken from an independently generated LODGE dance and EDGE
dance. Because the two generators live in different motion distributions, we cannot make one
model's diffusion *continue* the other's pose (proven infeasible without fine-tuning). Instead
we impose a smooth handoff at each generator switch using the game-animation standard:

  root position + facing (yaw) alignment  +  inertialized blending (Bollo 2016).

At a switch the incoming segment is rotated/translated so its first frame's root matches the
outgoing motion's end, then the first ``blend_frames`` frames are eased from the outgoing end
pose into the incoming motion with a smoothstep-decayed per-joint quaternion offset. This yields
a C0/near-C1 continuous seam that (measured) is smoother than native same-generator stitching.

All motion here is the AgentLODGE 139-dim layout ``[trans(3) | 22*6D rot(132) | contact(4)]``.
Everything is normalised to a Z-up frame (EDGE-native) so a mixed sequence renders consistently.

pytorch3d is imported lazily so importing this module never hard-fails in a pytorch3d-less env.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 22
_ROOT = slice(3, 9)          # root joint 6D in the 139 layout
_ROT = slice(3, 3 + NUM_JOINTS * 6)
_TRANS = slice(0, 3)
_CONTACT = slice(3 + NUM_JOINTS * 6, 139)

# Rx(+90): Y-up -> Z-up, (x, y, z) -> (x, -z, y)
_RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=np.float32)


def _t():
    import torch  # noqa: F401
    return torch


def _pt3d():
    from pytorch3d.transforms import (  # noqa: F401
        matrix_to_quaternion,
        matrix_to_rotation_6d,
        quaternion_to_matrix,
        rotation_6d_to_matrix,
    )
    return (rotation_6d_to_matrix, matrix_to_rotation_6d,
            matrix_to_quaternion, quaternion_to_matrix)


def to_zup(motion139: np.ndarray) -> np.ndarray:
    """Convert a Y-up (LODGE/FineDance) 139 motion to the Z-up (EDGE) frame.

    Input MUST be in AgentLODGE layout ``[trans(3) | rot(132) | contact(4)]`` (contact last);
    pass LODGE-native ``[contact | trans | rot]`` arrays through ``to_agentlodge139`` first or the
    root orientation will be corrupted (contact channels get swizzled as translation).

    Translation is swizzled (x, y, z) -> (x, -z, y); the root joint's global orientation is
    pre-multiplied by Rx(+90). Local joint rotations (frame-independent) are unchanged.
    """
    torch = _t()
    rot6d_to_mat, mat_to_6d, *_ = _pt3d()
    m = motion139.astype(np.float32).copy()
    s = m.shape[0]
    trans = m[:, _TRANS]
    m[:, _TRANS] = np.stack([trans[:, 0], -trans[:, 2], trans[:, 1]], axis=1)
    R = rot6d_to_mat(torch.from_numpy(m[:, _ROT].reshape(s, NUM_JOINTS, 6).copy()).float())
    R[:, 0] = torch.einsum("ij,sjk->sik", torch.from_numpy(_RX90), R[:, 0])
    m[:, _ROT] = mat_to_6d(R).reshape(s, NUM_JOINTS * 6).numpy()
    return m


def rotate_root_yaw(motion139: np.ndarray, delta_rad: float, up: str = "z") -> np.ndarray:
    """Rotate a whole 139-dim motion about its vertical axis by ``delta_rad`` (global facing).

    Applies R(delta) about the ``up`` axis to the root joint's global orientation and to the root
    translation so the dancer faces a new direction without otherwise altering the choreography.
    ``up`` is "z" for the AgentLODGE/EDGE Z-up frame and "y" for native LODGE/FineDance (Y-up):
    rotating a Y-up motion about +Y maps to a +Z rotation of the rendered (re-oriented) result.
    Used to align a motion's facing to a reference (e.g. EDGE, which faces the camera).
    """
    if abs(delta_rad) < 1e-9:
        return motion139
    torch = _t()
    rot6d_to_mat, mat_to_6d, *_ = _pt3d()
    m = motion139.astype(np.float32).copy()
    s = m.shape[0]
    c, sn = np.cos(delta_rad), np.sin(delta_rad)
    if up == "y":
        R = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=np.float32)
    else:  # "z"
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    m[:, _TRANS] = m[:, _TRANS] @ R.T
    Rmat = rot6d_to_mat(torch.from_numpy(m[:, _ROT].reshape(s, NUM_JOINTS, 6).copy()).float())
    Rmat[:, 0] = torch.einsum("ij,sjk->sik", torch.from_numpy(R), Rmat[:, 0])
    m[:, _ROT] = mat_to_6d(Rmat).reshape(s, NUM_JOINTS * 6).numpy()
    return m
def _quat_from_6d(r6):
    rot6d_to_mat, _, mat_to_quat, _ = _pt3d()
    return mat_to_quat(rot6d_to_mat(r6))


def _6d_from_quat(q):
    _, mat_to_6d, _, quat_to_mat = _pt3d()
    return mat_to_6d(quat_to_mat(q))


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    torch = _t()
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(a):
    c = a.clone()
    c[..., 1:] *= -1
    return c / (a * a).sum(-1, keepdim=True)


def _slerp(a, b, w):
    torch = _t()
    d = (a * b).sum(-1, keepdim=True)
    b = torch.where(d < 0, -b, b)
    d = d.abs().clamp(max=1.0)
    ang = torch.arccos(d)
    s = torch.sin(ang)
    w = w.unsqueeze(-1)
    lin = d > 0.9995
    out = torch.where(lin, a + w * (b - a),
                      (torch.sin((1 - w) * ang) * a + torch.sin(w * ang) * b) / (s + 1e-8))
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


def _yaw_z(rootq):
    """Yaw about +Z from a root quaternion (Z-up)."""
    _, _, _, quat_to_mat = _pt3d()
    torch = _t()
    R = quat_to_mat(rootq)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def root_yaw(frame139: np.ndarray) -> float:
    """Yaw (about +Z) of a single 139-dim frame's root orientation, in the Z-up frame."""
    torch = _t()
    q = _quat_from_6d(torch.from_numpy(frame139[_ROOT].reshape(1, 6).astype(np.float32)).float())
    return float(_yaw_z(q[0]))


def blend_onto(prev_tail139: np.ndarray, seg139: np.ndarray, blend_frames: int = 15,
               canonical_yaw: float | None = None, align_facing: bool = True) -> np.ndarray:
    """Align ``seg139`` to the end of ``prev_tail139`` and inertialize the handoff.

    Position is always chained (seg starts where prev ended) so there is no teleport. When
    ``align_facing`` is True, the segment is rotated about +Z to ``canonical_yaw`` (kills
    cross-segment drift) or the previous segment's end facing. When False, the segment KEEPS its
    own natural facing (only position is chained) -- use this when both generators already face a
    consistent direction (e.g. the camera), so re-anchoring would rotate a whole segment away; the
    small seam orientation difference is still smoothed by the inertialized root blend below.

    Both inputs must already be in the same (Z-up) frame. Returns the aligned + inertialized
    segment (same length as ``seg139``), ready to concatenate after the previous motion.
    """
    torch = _t()
    seg = torch.from_numpy(seg139.astype(np.float32)).clone()
    prev_end = torch.from_numpy(prev_tail139[-1].astype(np.float32))
    n = seg.shape[0]
    blend_frames = int(max(0, min(blend_frames, n)))

    seg_tr = seg[:, _TRANS]
    seg_r6 = seg[:, _ROT].reshape(n, NUM_JOINTS, 6)
    seg_q = _quat_from_6d(seg_r6)                       # (n, 22, 4)
    prev_r6 = prev_end[_ROT].reshape(NUM_JOINTS, 6)
    prev_q = _quat_from_6d(prev_r6)                     # (22, 4)
    prev_tr = prev_end[_TRANS]

    # 1. facing alignment: rotate the segment about +Z. Target = a fixed canonical yaw (kills
    #    cross-segment drift) when provided, else the previous segment's end facing. When
    #    align_facing is False, keep the segment's own facing (dyaw = 0) and only chain position.
    _, _, mat_to_quat, quat_to_mat = _pt3d()
    if align_facing:
        target_yaw = (torch.tensor(float(canonical_yaw)) if canonical_yaw is not None
                      else _yaw_z(prev_q[0]))
        dyaw = target_yaw - _yaw_z(seg_q[0, 0])
    else:
        dyaw = torch.tensor(0.0)
    cz, sz = torch.cos(dyaw), torch.sin(dyaw)
    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)]),
        torch.stack([sz, cz, torch.zeros_like(cz)]),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
    ])
    segR0 = quat_to_mat(seg_q[:, 0])
    seg_q[:, 0] = mat_to_quat(torch.einsum("ij,sjk->sik", Rz, segR0))
    seg_tr = torch.einsum("ij,sj->si", Rz, seg_tr - seg_tr[0]) + prev_tr

    # 2. inertialize: first blend_frames start at prev end pose, ease into segment.
    if blend_frames > 0:
        qoff = _qmul(prev_q, _qinv(seg_q[0]))          # (22, 4)
        troff = prev_tr - seg_tr[0]                    # (3,)
        k = torch.arange(blend_frames).float() / max(1, blend_frames - 1)
        w = 1.0 - (3 * k ** 2 - 2 * k ** 3)            # smoothstep 1 -> 0
        for f in range(blend_frames):
            shifted = _qmul(qoff, seg_q[f])
            seg_q[f] = _slerp(seg_q[f], shifted, w[f].expand(NUM_JOINTS))
            seg_tr[f] = seg_tr[f] + w[f] * troff

    seg[:, _TRANS] = seg_tr
    seg[:, _ROT] = _6d_from_quat(seg_q).reshape(n, NUM_JOINTS * 6)
    return seg.numpy().astype(np.float32)
