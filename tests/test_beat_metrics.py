"""Unit tests for the Phase-0 beat-alignment metrics and the retrograde primitive.

Pure-numpy + scipy, no torch / audio / FK backends -- runnable anywhere like test_story.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import beat_metrics as BM
from agentlodge.dance import transition as T


_IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _pulsing_motion(length: int, period: int) -> np.ndarray:
    """A valid 139-dim motion whose kinematic speed dips to ~0 every `period` frames.

    Rotations are constant (identity 6D) so the kinematic speed equals |d/dt trans|; translation
    integrates a |sin| speed profile whose zeros fall on multiples of `period` -> motion beats
    there.
    """
    t = np.arange(length)
    speed = np.abs(np.sin(np.pi * t / period))
    trans = np.zeros((length, 3), dtype=np.float32)
    trans[:, 0] = np.cumsum(speed)
    rot = np.tile(_IDENTITY_6D, (length, 22)).reshape(length, 132).astype(np.float32)
    contact = np.ones((length, 4), dtype=np.float32)
    return np.concatenate([trans, rot, contact], axis=1).astype(np.float32)


def _random_motion(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * 0.05, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- retrograde
def test_retrograde_is_involution():
    m = _random_motion(40, seed=1)
    assert np.allclose(T.retrograde(T.retrograde(m)), m, atol=1e-6)


def test_retrograde_reverses_frames_and_contacts():
    m = _random_motion(25, seed=2)
    r = T.retrograde(m)
    assert r.shape == m.shape
    assert np.allclose(r[0], m[-1])
    assert np.allclose(r[-1], m[0])
    assert np.allclose(r[:, 135:139], m[::-1, 135:139])


# --------------------------------------------------------------------------- BAS
def test_bas_perfect_when_motion_beats_on_music_beats():
    mb = np.array([10, 20, 30, 40])
    bas = BM.beat_alignment_score(np.zeros((50, 139)), mb, sigma_frames=3.0, motion_beats=mb)
    assert abs(bas - 1.0) < 1e-6


def test_bas_lower_when_misaligned_and_in_unit_range():
    mb = np.array([20])
    aligned = BM.beat_alignment_score(np.zeros((50, 139)), np.array([20]),
                                      sigma_frames=3.0, motion_beats=mb)
    off3 = BM.beat_alignment_score(np.zeros((50, 139)), np.array([23]),
                                   sigma_frames=3.0, motion_beats=mb)  # dist 3, sigma 3
    assert abs(aligned - 1.0) < 1e-6
    assert abs(off3 - np.exp(-0.5)) < 1e-6  # exp(-3^2 / (2*3^2))
    assert 0.0 <= off3 < aligned <= 1.0


def test_bas_zero_on_empty_beats():
    assert BM.beat_alignment_score(np.zeros((10, 139)), [], motion_beats=np.array([1, 2])) == 0.0
    assert BM.beat_alignment_score(np.zeros((10, 139)), [1, 2], motion_beats=np.array([])) == 0.0


# --------------------------------------------------------------------------- kinematic beats
def test_kinematic_beats_detects_periodic_stops():
    period = 15
    m = _pulsing_motion(150, period)
    beats = BM.kinematic_beats(m)
    assert beats.size >= 5
    # every detected beat is near a multiple of the period
    nearest = np.abs(beats[:, None] - np.arange(0, 151, period)[None, :]).min(axis=1)
    assert np.all(nearest <= 4)
    # and BAS of detected beats against the true grid is high
    grid = np.arange(0, 151, period)
    assert BM.beat_alignment_score(m, grid, sigma_frames=3.0, motion_beats=beats) > 0.5


# --------------------------------------------------------------------------- coverage + feet
def test_beat_coverage_in_unit_range():
    m = _pulsing_motion(150, 15)
    cov = BM.beat_coverage(m, np.arange(0, 151, 15))
    assert 0.0 <= cov <= 1.0
    assert cov > 0.4  # most music beats have a nearby motion beat


def test_foot_contact_consistency_perfect_when_stationary_contact():
    m = _pulsing_motion(60, 15)
    m[:, :3] = 0.0            # no horizontal movement
    m[:, 135:139] = 1.0       # feet always in contact
    assert BM.foot_contact_consistency(m) == 1.0


def test_compute_beat_metrics_has_expected_keys():
    m = _pulsing_motion(120, 15)
    out = BM.compute_beat_metrics(m, np.arange(0, 121, 15))
    assert set(out) == {"beat_alignment", "beat_coverage", "foot_contact_consistency", "n_motion_beats"}
    assert 0.0 <= out["beat_alignment"] <= 1.0
    assert out["n_motion_beats"] >= 1
