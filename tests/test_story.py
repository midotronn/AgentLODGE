"""Unit tests for the structure-aware ("story") choreography stage.

Covers the pure-numpy pieces that run without the heavy rotation/audio backends:
motion transforms (mirror/retime/amplitude_scale), music-structure helpers, the storyboard
agent's rule-based fallback + JSON parsing, and per-section source selection. The librosa
segmentation path and torch-based inertial assembly are validated separately (on-GPU).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import transition as T
from agentlodge.audio import structure as S
from agentlodge.agent import storyboard as SB
from agentlodge.dance import story as ST


def _valid_motion(n: int, scale: float = 0.05, seed: int = 0) -> np.ndarray:
    """A random but structurally valid AgentLODGE 139-dim motion (orthonormal 6D rotations)."""
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * scale, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def _structure(seed: int = 0) -> S.MusicStructure:
    bounds = [0, 30, 90, 150, 210, 270, 300]
    ec = np.interp(np.arange(300), [0, 150, 299], [0.0, 1.0, 0.0]).astype(np.float32)
    feats = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, .05], [0, 0, 1], [1, 0, .05]], float)
    return S._build_sections(bounds, ec, feats, 120.0, 300, used_fallback=False)


# --------------------------------------------------------------------------- motion transforms
def test_mirror_is_involution():
    m = _valid_motion(40, seed=1)
    assert np.allclose(T.mirror(T.mirror(m)), m, atol=1e-4)


def test_mirror_swaps_contacts_and_keeps_valid_rotations():
    m = _valid_motion(30, seed=2)
    mm = T.mirror(m)
    assert np.allclose(mm[:, 135:139], m[:, [137, 138, 135, 136]])
    R = T._sixd_to_matrix(mm[:, 3:135].reshape(30, 22, 6))
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_retime_changes_length_binary_contacts_and_is_identity_at_same_length():
    m = _valid_motion(40, seed=3)
    r = T.retime(m, 25)
    assert r.shape == (25, 139)
    assert set(np.unique(r[:, 135:139]).tolist()) <= {0.0, 1.0}
    assert np.array_equal(T.retime(m, 40), m)


def test_amplitude_scale_preserves_valid_rotations_and_clamps():
    m = _valid_motion(40, seed=4)
    a = T.amplitude_scale(m, 5.0)  # clamped to 1.4
    R = T._sixd_to_matrix(a[:, 3:135].reshape(40, 22, 6))
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)
    assert np.array_equal(T.amplitude_scale(m, 1.0), m)


# --------------------------------------------------------------------------- structure helpers
def test_merge_short_and_snap():
    assert S._merge_short([0, 10, 12, 60, 120], 15) == [0, 60, 120]
    assert S._snap_to_downbeats([0, 50, 120], np.array([0, 48, 96, 144]), 5) == [0, 48, 120]


def test_label_sections_detects_repetition():
    feats = np.array([[1, 0, 0], [0, 1, 0], [1, 0, .05], [0, 1, .05], [0, 0, 1]], float)
    assert list(S._label_sections(feats)) == [0, 1, 0, 1, 2]


def test_build_sections_covers_full_span():
    ms = _structure()
    assert ms.boundaries() == [0, 30, 90, 150, 210, 270, 300]
    assert ms.sections[0].start_frame == 0 and ms.sections[-1].end_frame == 300
    assert 0 <= ms.climax_index < len(ms.sections)
    for s in ms.sections:
        assert 0.0 <= s.energy <= 1.0
        assert s.role in S._ROLES


# --------------------------------------------------------------------------- storyboard agent
def test_rule_based_storyboard_reuses_earlier_same_label():
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    assert len(board.plans) == len(ms.sections)
    assert board.used_fallback
    assert board.plans[0].reuse_of is None
    for p in board.plans:
        if p.reuse_of is not None:
            assert p.reuse_of < p.section_index
            assert ms.sections[p.reuse_of].label == ms.sections[p.section_index].label


def test_storyboard_parse_validates_and_rejects_bad_reuse():
    ms = _structure()
    payload = {"arc": "x", "reasoning": "y", "plans": [
        {"section_index": i, "role": s.role, "target_intensity": float(s.energy),
         "vocabulary": "explosive_fast", "generator_bias": "edge",
         "reuse_of": None, "variation": {"mirror": False, "retime": 1.0, "amplitude": 1.1}}
        for i, s in enumerate(ms.sections)]}
    board = SB._parse_response(json.dumps(payload), ms)
    assert not board.used_fallback and len(board.plans) == len(ms.sections)
    # reuse of a section with a different label must be rejected
    bad = {"section_index": 1, "role": "verse", "target_intensity": 0.5,
           "vocabulary": "explosive_fast", "generator_bias": "edge", "reuse_of": 0}
    assert SB._coerce_plan(bad, 1, ms).reuse_of is None


def test_storyboard_parse_rejects_wrong_plan_count():
    ms = _structure()
    try:
        SB._parse_response(json.dumps({"arc": "x", "plans": []}), ms)
    except ValueError:
        return
    raise AssertionError("expected ValueError on plan-count mismatch")


# --------------------------------------------------------------------------- source selection
def test_select_sources_is_gap_free_and_covers_full_length():
    lodge = _valid_motion(300, scale=0.02, seed=5)
    edge = _valid_motion(300, scale=0.12, seed=6)
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    dec = ST.select_sources(lodge, edge, ms, board, motif_reuse=True)
    assert dec[0]["a"] == 0
    assert dec[-1]["b"] == 300
    assert all(dec[i]["b"] == dec[i + 1]["a"] for i in range(len(dec) - 1))
    for d in dec:
        assert d["source"] in {"lodge", "edge"} or d["source"].startswith("reuse:")


def test_select_sources_generates_motif_candidate_when_planned():
    lodge = _valid_motion(300, scale=0.05, seed=7)
    edge = _valid_motion(300, scale=0.05, seed=8)
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    dec = ST.select_sources(lodge, edge, ms, board, motif_reuse=True)
    # sections 3 and 4 repeat sections 1 and 2 -> a reuse candidate must be scored
    assert any(k.startswith("reuse") for k in dec[3]["costs"])
    assert any(k.startswith("reuse") for k in dec[4]["costs"])
