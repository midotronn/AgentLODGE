"""Rebuild the STORY plan for a song and dump the storyboard + per-section selection costs.

Shows exactly what the story planner decided and WHY:
  * the storyboard (arc + reasoning + per-section role/intensity/bias/vocab/reuse),
  * per-section source choice with the full per-candidate cost breakdown,
  * structure ("story") quality metrics for the assembled dance.

Runs the same code path as the pipeline's story stage. Writes JSON to story_<sid>.json and
prints a readable summary. INFO logging is enabled so story.py / storyboard.py logs show live.

Usage (on the pod):  python dump_story.py 150 [min_section_seconds]
Paths default to the pod layout under $AGENTLODGE_DUMP_BASE (default /workspace); override with:
  AGENTLODGE_DUMP_BASE=/workspace  LODGE_DIR=/workspace/LODGE  OAI_KEY_FILE=/workspace/.oai_key
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal as _sps

if not hasattr(_sps, "hann"):  # older/newer scipy shim used elsewhere in this repo
    from scipy.signal.windows import hann as _hann
    _sps.hann = _hann

BASE = Path(os.getenv("AGENTLODGE_DUMP_BASE", "/workspace"))
LODGE = Path(os.getenv("LODGE_DIR", str(BASE / "LODGE")))
sys.path.insert(0, str(BASE / "AgentLODGE"))

import librosa  # noqa: E402

from agentlodge.audio.preprocess import SongMetadata  # noqa: E402
from agentlodge.audio.structure import analyze_structure  # noqa: E402
from agentlodge.agent.storyboard import author_storyboard  # noqa: E402
from agentlodge.dance.story import build_story_dance  # noqa: E402
from agentlodge.dance.story_metrics import compute_story_metrics  # noqa: E402
from agentlodge.config import FPS, SAMPLE_RATE, HOP_LENGTH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    sid = sys.argv[1] if len(sys.argv) > 1 else "150"
    min_section = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    wav = Path(f"{LODGE}/data/finedance/music_wav/{sid}.wav")
    key_file = Path(os.getenv("OAI_KEY_FILE", str(BASE / ".oai_key")))
    api_key = key_file.read_text().strip() if key_file.exists() else os.getenv("OPENAI_API_KEY")

    lodge_raw = np.load(f"{BASE}/lodge_fd_{sid}_full.npy")
    edge_raw = np.load(f"{BASE}/edge_fd_{sid}_full.npy")
    y, sr = librosa.load(str(wav), sr=SAMPLE_RATE)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    n0 = int(min(len(lodge_raw), len(edge_raw)))
    beats = np.asarray(beats)[np.asarray(beats) < n0]
    meta = SongMetadata(len(y) / sr, float(np.atleast_1d(tempo)[0]), beats, wav)

    structure = analyze_structure(wav, meta, n0, min_section_seconds=min_section)
    storyboard = author_storyboard(structure, meta, None, api_key, motif_reuse=True)
    story = build_story_dance(lodge_raw, edge_raw, structure, storyboard, meta,
                              blend_frames=15, motif_reuse=True, energy_shaping=False)
    metrics = compute_story_metrics(story.motion, structure)

    print("\n" + "=" * 78)
    print(f"STORYBOARD (song {sid})")
    print("=" * 78)
    print(storyboard.describe())

    print("\n" + "=" * 78)
    print("PER-SECTION SELECTION (why each section chose its source)")
    print("=" * 78)
    for s in story.section_scores:
        mark = "= honored" if s["matched_bias"] else ("~ auto" if s["plan_bias"] == "auto"
                                                       else "! overrode")
        print(f"[{s['a'] / FPS:5.1f}-{s['b'] / FPS:5.1f}s] {s['role']:<8} "
              f"chose {s['source']:<8} (cost {s['chosen_cost']:.3f})  "
              f"plan bias={s['plan_bias']:<5} target_E={s['target_intensity']:.2f}  {mark}")
        print(f"            costs={s['costs']}  energies={s['energies']}")

    print("\n" + "=" * 78)
    print("STORY METRICS")
    print("=" * 78)
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print(f"\nreasoning: {story.reasoning}")

    out = {
        "song": sid,
        "storyboard": storyboard.to_dict(),
        "structure": structure.to_dict(),
        "story_schedule": [[int(a), int(b), str(src), str(role)] for a, b, src, role in story.schedule],
        "story_section_scores": story.section_scores,
        "story_metrics": metrics,
        "reasoning": story.reasoning,
    }
    out_path = Path(f"{BASE}/story_{sid}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print("DUMP_DONE")


if __name__ == "__main__":
    main()
