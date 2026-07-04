#!/usr/bin/env bash
# Run full AgentLODGE e2e on RunPod and log output.
set -euo pipefail

WORK="${WORKSPACE:-/workspace}"
cd "$WORK/AgentLODGE"
source .venv/bin/activate

AUDIO="${1:-test_audio_30s.wav}"
OUT="${2:-outputs/runpod_e2e}"
mkdir -p "$(dirname "$OUT")" "$OUT"

LOG="$OUT/run.log"
echo "Starting e2e at $(date)" | tee "$LOG"

AGENTLODGE_PARALLEL=0 \
python run_pipeline.py \
  --audio "$AUDIO" \
  --output_dir "$OUT" 2>&1 | tee -a "$LOG"

echo "--- Results ---" | tee -a "$LOG"
ls -la "$OUT" | tee -a "$LOG"
python3 - <<PY
import json, numpy as np
from pathlib import Path
out = Path("$OUT")
log = json.loads((out / "pipeline_log.json").read_text())
print(json.dumps(log, indent=2))
d = np.load(out / "selected_dance.npy")
print("selected_dance.npy shape:", d.shape)
assert d.shape[1] == 139
img = out / "costume_output.png"
print("costume_output.png:", "OK" if img.exists() and img.stat().st_size > 1000 else "MISSING (API key?)")
video = out / "dance_stick_figure.mp4"
print("dance_stick_figure.mp4:", "OK" if video.exists() and video.stat().st_size > 1000 else "MISSING")
PY

echo "Done at $(date)" | tee -a "$LOG"
