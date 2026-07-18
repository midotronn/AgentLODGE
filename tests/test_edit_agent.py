"""Unit tests for the Phase-4 natural-language editing agent (offline / rule-based parse)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.audio import structure as S
from agentlodge.agent import storyboard as SB
from agentlodge.agent import edit_agent as EA
from agentlodge.dance import transition as T


def _valid_motion(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * 0.05, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def _structure() -> S.MusicStructure:
    bounds = [0, 30, 60, 90]
    ec = np.interp(np.arange(90), [0, 45, 89], [0.0, 1.0, 0.2]).astype(np.float32)
    return S._build_sections(bounds, ec, np.eye(3, dtype=float), 120.0, 90, used_fallback=False)


def _agent(**kw):
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    return EA.EditAgent(ms, board, _valid_motion(90, 3), _valid_motion(90, 4),
                        music_beat_frames=np.arange(0, 90, 5), **kw)


# --------------------------------------------------------------------------- resolve + parse
def test_resolve_section():
    ms = _structure()  # roles: intro, drop, outro
    assert EA.resolve_section("the intro", ms) == 0
    assert EA.resolve_section("the drop", ms) == 1
    assert EA.resolve_section("at the end", ms) == 2
    assert EA.resolve_section(2, ms) == 2
    # time resolution on a realistically-scaled structure (10s sections)
    big = S._build_sections([0, 300, 600, 900], np.linspace(0, 1, 900).astype(np.float32),
                            np.eye(3, dtype=float), 120.0, 900, used_fallback=False)
    assert EA.resolve_section("0:15", big) == 1        # 15s -> section 1 (10-20s)


def test_rule_parse_actions():
    ms = _structure()
    assert EA.parse_instruction("reuse the intro and mirror it at the end", ms).action == "recapitulate"
    assert EA.parse_instruction("make the drop more energetic", ms).action == "set_intensity"
    assert EA.parse_instruction("make the intro calmer", ms).params["direction"] == -1
    assert EA.parse_instruction("reverse the intro", ms).action == "retrograde"
    assert EA.parse_instruction("mirror the drop", ms).action == "mirror"
    assert EA.parse_instruction("tighten the beat on the drop", ms).action == "beat"


# --------------------------------------------------------------------------- edits
def test_recapitulation_edit():
    ag = _agent()
    res = ag.edit("reuse the opening and mirror it at the end")
    assert res.ok
    assert res.decisions[-1]["source"] == "reuse:0"
    assert ag.recapitulate is True


def test_make_section_more_energetic():
    ag = _agent()
    res = ag.edit("make the drop more energetic")
    assert res.op.action == "set_intensity"
    d = res.decisions[1]
    le = {k: v for k, v in d["energies"].items() if k in ("lodge", "edge")}
    assert d["source"] == max(le, key=le.get)   # now uses the higher-energy source
    assert res.ok


def test_reverse_intro_applies_retrograde():
    ag = _agent()
    res = ag.edit("reverse the intro")
    assert res.op.action == "retrograde" and res.op.section == 0
    assert "retrograde" in res.decisions[0]["post_variation"]
    assert res.ok


def test_mirror_section_applies_post_variation():
    ag = _agent()
    res = ag.edit("mirror the drop")
    assert "mirror" in res.decisions[1]["post_variation"]
    assert res.ok


def test_beat_edit_sets_a_generator_and_returns_result():
    ag = _agent()
    res = ag.edit("tighten the beat on the drop")
    assert res.op.action == "beat"
    assert ag.storyboard.plans[1].generator_bias in ("lodge", "edge")
    assert res.iterations <= ag.max_iters
    assert len(res.decisions) == 3


def test_edit_always_returns_decisions_even_on_partial():
    ag = _agent(max_iters=2)
    res = ag.edit("make the drop more energetic")
    assert isinstance(res, EA.EditResult)
    assert res.decisions and res.iterations <= 2
