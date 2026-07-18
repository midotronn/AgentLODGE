"""Unit tests for Phase-3 segment captions + plan-realization alignment (verifier)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.agent import segment_caption as C
from agentlodge.agent.storyboard import SectionPlan
from agentlodge.dance import transition as T


def _motion(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * 0.05, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def test_segment_features_keys_and_ranges():
    f = C.segment_features(_motion(60, 1))
    assert {"mean_energy", "cov", "trend", "directionality", "contact_rate",
            "regularity", "n_motion_beats"} <= set(f)
    assert 0.0 <= f["contact_rate"] <= 1.0
    assert 0.0 <= f["directionality"] <= 1.0 + 1e-6
    assert 0.0 <= f["regularity"] <= 1.0


def test_caption_is_readable_and_uses_energy_level():
    cap_hi = C.caption_segment(_motion(60, 2), energy_norm=0.9)
    cap_lo = C.caption_segment(_motion(60, 2), energy_norm=0.05)
    assert cap_hi.startswith("A ") and cap_hi.endswith(".") and "phrase" in cap_hi
    assert "explosive" in cap_hi
    assert "calm" in cap_lo
    # without energy_norm, no absolute level word forced
    assert C.caption_segment(_motion(60, 2)).startswith("A ")


def test_vocabulary_match():
    assert abs(C.vocabulary_match(0.90, "explosive_fast") - 1.0) < 1e-6
    assert C.vocabulary_match(0.10, "explosive_fast") < 0.3
    assert C.vocabulary_match(0.12, "grounded_minimal") > 0.9
    assert C.vocabulary_match(0.5, "") == 1.0  # empty vocab -> neutral


def test_plan_realization_alignment():
    plan = SectionPlan(section_index=0, role="drop", target_intensity=0.9,
                       vocabulary="explosive_fast", generator_bias="edge")
    good = C.plan_realization_alignment(plan, 0.9)
    bad = C.plan_realization_alignment(plan, 0.1)
    assert abs(good - 1.0) < 1e-6
    assert bad < good
    # a low TMR critic score drags the aligned case down
    blended = C.plan_realization_alignment(plan, 0.9, tmr_score=0.0)
    assert abs(blended - 0.6) < 1e-6


def test_story_decisions_include_caption_and_alignment():
    # captions/alignment are surfaced per section by the assembler's select_sources
    from agentlodge.audio import structure as S
    from agentlodge.agent import storyboard as SB
    from agentlodge.dance import story as ST

    bounds = [0, 30, 60, 90]
    ec = np.linspace(0, 1, 90).astype(np.float32)
    feats = np.eye(3, dtype=float)
    ms = S._build_sections(bounds, ec, feats, 120.0, 90, used_fallback=False)
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    dec = ST.select_sources(_motion(90, 3), _motion(90, 4), ms, board, motif_reuse=True)
    assert all("caption" in d and "phrase" in d["caption"] for d in dec)
    assert all(0.0 <= d["plan_alignment"] <= 1.0 for d in dec)
