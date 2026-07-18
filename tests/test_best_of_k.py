"""Unit tests for Phase-2 best-of-K seed selection (generator-agnostic; synthetic candidates)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import best_of_k as BK

_IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _pulsing_motion(length: int, period: int, phase: int = 0) -> np.ndarray:
    """Valid 139-dim motion whose kinematic speed dips to ~0 every `period` frames, offset by
    `phase` frames -> motion beats at {phase, phase+period, ...}."""
    t = np.arange(length)
    speed = np.abs(np.sin(np.pi * (t - phase) / period))
    trans = np.zeros((length, 3), dtype=np.float32)
    trans[:, 0] = np.cumsum(speed)
    rot = np.tile(_IDENTITY_6D, (length, 22)).reshape(length, 132).astype(np.float32)
    contact = np.ones((length, 4), dtype=np.float32)
    return np.concatenate([trans, rot, contact], axis=1).astype(np.float32)


def _music():
    return np.arange(0, 151, 15)


def test_aligned_candidate_scores_higher_bas():
    aligned = _pulsing_motion(150, 15, phase=0)
    off = _pulsing_motion(150, 15, phase=7)  # ~half a beat out of phase
    scores = BK.score_candidates([off, aligned], _music())
    assert scores[1]["bas"] > scores[0]["bas"]


def test_select_best_picks_the_beat_aligned_candidate():
    aligned = _pulsing_motion(150, 15, phase=0)
    off = _pulsing_motion(150, 15, phase=7)
    best, scores = BK.select_best([off, aligned], _music())
    assert best == 1
    assert len(scores) == 2


def test_best_of_k_selects_winning_seed():
    def generate_fn(seed):
        return _pulsing_motion(150, 15, phase=0) if seed == 42 else _pulsing_motion(150, 15, phase=7)

    motion, seed, report = BK.best_of_k(generate_fn, [1, 42, 3], _music())
    assert seed == 42
    assert report["winner_seed"] == 42
    assert report["k"] == 3
    assert report["scores"][report["winner_index"]]["bas"] == report["winner_bas"]


def test_best_of_k_skips_invalid_and_raises_when_all_fail():
    def bad(seed):
        return None if seed % 2 else np.zeros((1, 139))  # None or too-short

    with pytest.raises(ValueError):
        BK.best_of_k(bad, [1, 2, 3], _music())


def test_best_of_k_survives_a_throwing_seed():
    def flaky(seed):
        if seed == 2:
            raise RuntimeError("bad seed")
        return _pulsing_motion(150, 15, phase=0 if seed == 1 else 7)

    motion, seed, report = BK.best_of_k(flaky, [1, 2, 3], _music())
    assert 2 not in report["seeds"]          # the throwing seed was skipped
    assert seed == 1                          # aligned candidate wins


def test_target_intensity_affects_energy_match():
    calm = _pulsing_motion(150, 15, phase=0)
    calm[:, :3] *= 0.3                          # smaller movement -> lower energy
    lively = _pulsing_motion(150, 15, phase=0)
    lively[:, :3] *= 3.0                         # larger movement -> higher energy
    hi = BK.score_candidates([calm, lively], _music(), target_intensity=1.0)
    assert hi[1]["energy_match"] >= hi[0]["energy_match"]  # target=high favors the livelier one
