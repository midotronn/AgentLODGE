# AgentLODGE

End-to-end pipeline that accepts a song, generates dances with **LODGE** and **EDGE** in parallel, selects the best result with an LLM agent, and generates a costume illustration derived entirely from the audio.

## Pipeline

1. **Audio preprocessing** — Librosa 35-dim features (LODGE) and Jukebox embeddings (EDGE) at 30 FPS
2. **Parallel dance generation** — LODGE global+PDDM and EDGE long-form (5s clips, 2.5s overlap)
3. **Dance selection agent** — an LLM reasons over per-window **long-term coherence** signals (seam smoothness, jitter, foot stability, sustained beat-sync trend, variety), scores each dance on a coherence rubric, and picks the more coherent long-form dance; beat alignment and diversity are secondary tie-breakers
4. **Stick-figure video** — SMPL forward kinematics + matplotlib animation, muxed with input audio
5. **Costume description agent** — an LLM turns the song's acoustic features (tempo, energy, timbre, key/mode, rhythmic density) plus the `LODGE_GENRE` hint into a costume description
6. **Costume image generation** — OpenAI (`gpt-image-1`) or Gemini Imagen renders the audio-derived description

## Requirements

- Python 3.10+
- [LODGE](https://li-ronghui.github.io/lodgepp) codebase and pretrained weights
- [EDGE](https://edge-dance.github.io) codebase and checkpoint
- `ffmpeg` on PATH (stick-figure video export)
- API key for OpenAI (selection agent, **costume description agent**, and costume image when `IMAGE_BACKEND=openai`). Because the costume description always uses OpenAI chat, `OPENAI_API_KEY` is required for costume generation even when `IMAGE_BACKEND=gemini`.

## Setup

```bash
cd AgentLODGE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and model paths
```

Ensure LODGE and EDGE are cloned and configured separately (see their READMEs). Point `LODGE_CODE_PATH`, `EDGE_CODE_PATH`, and weight paths in `.env` to your local installs.

Recommended defaults if you use the sibling `Runs/` layout in this workspace:

```env
LODGE_CODE_PATH=../Runs/LODGE
EDGE_CODE_PATH=../Runs/EDGE
LODGE_WEIGHTS_PATH=../Runs/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt
LODGE_GLOBAL_WEIGHTS_PATH=../Runs/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt
EDGE_WEIGHTS_PATH=../Runs/EDGE/checkpoint.pt
```

Run LODGE and EDGE inference from their own virtual environments if dependency sets differ; the pipeline subprocesses inherit the active Python environment.

## Memory and Apple Silicon notes

The pipeline runs each heavy model in a **separate subprocess** and frees memory between steps to avoid system crashes:

1. LODGE (isolated subprocess)
2. Jukebox feature extraction (EDGE venv subprocess)
3. EDGE inference (EDGE venv subprocess)

On Macs with limited unified memory, **Jukebox may be killed (OOM)** during step 2. The pipeline will still finish using the LODGE dance and log the failure in `pipeline_log.json`.

Recommendations:

- Use at least **20 seconds** of audio (`AGENTLODGE_MIN_AUDIO_SECONDS=20`) for LODGE fine diffusion.
- Limit EDGE slices on memory-constrained machines: `AGENTLODGE_MAX_EDGE_SLICES=7` (default).
- Keep `AGENTLODGE_PARALLEL=0` on Apple Silicon (default when no CUDA GPU).
- Pre-extract Jukebox `.npy` features on a GPU machine and copy them into the work dir `edge_juke_cache/` to skip extraction.
- Close other memory-heavy apps before running EDGE/Jukebox.

## Usage

```bash
python run_pipeline.py \
  --audio path/to/song.wav \
  --output_dir ./outputs
```

The costume is derived automatically from the input audio — there is no costume text
argument. The generated description is printed and saved to `pipeline_log.json`.

## Outputs

Written to `output_dir`:

| File | Description |
|------|-------------|
| `selected_dance.npy` | Selected motion array `(L, 139)` in SMPL format |
| `dance_stick_figure.mp4` | Stick-figure animation of the selected dance, with input audio |
| `costume_output.png` | Generated costume illustration |
| `pipeline_log.json` | Selection reasoning, metrics, and errors |

You can also render a saved motion file directly:

```bash
python scripts/render_stick_video.py \
  --agentlodge-root . \
  --motion-npy outputs/run/selected_dance.npy \
  --output-mp4 outputs/run/dance_stick_figure.mp4 \
  --lodge-code-path ../Runs/LODGE \
  --audio path/to/song.wav
```

Stick-figure rendering needs `LODGE/data/smplx_neu_J_1.npy` (same file LODGE uses for FK).

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Selection agent, costume description agent, and costume image (when `IMAGE_BACKEND=openai`). Required for costume generation even under `IMAGE_BACKEND=gemini`. |
| `OPENAI_CHAT_MODEL` | Chat model for the selection and costume description agents (default: `gpt-4o-mini`) |
| `OPENAI_IMAGE_MODEL` | Image model for costume (default: `gpt-image-1`) |
| `GEMINI_API_KEY` | Required if `IMAGE_BACKEND=gemini` |
| `IMAGE_BACKEND` | `openai` or `gemini` |
| `OUTPUT_DIR` | Default output directory |
| `LODGE_CODE_PATH` | Path to LODGE repo |
| `EDGE_CODE_PATH` | Path to EDGE repo |
| `LODGE_WEIGHTS_PATH` | Local (PDDM) checkpoint |
| `LODGE_GLOBAL_WEIGHTS_PATH` | Global choreography checkpoint |
| `EDGE_WEIGHTS_PATH` | EDGE model checkpoint |
| `LODGE_GENRE` | FineDance genre label, also used as a costume style hint (default: `Hiphop`) |

## Error handling

- LODGE failure → falls back to EDGE
- EDGE failure → falls back to LODGE
- Both fail → `RuntimeError`
- Costume description agent failure (e.g. missing `OPENAI_API_KEY`) → logged; costume image skipped, dance outputs still saved
- Image generation failure → logged; dance output still saved
- Stick figure video failure → logged; motion and costume outputs still saved
- Selection agent failure → defaults to LODGE

## License

Prototype integration code. LODGE and EDGE are subject to their respective project licenses.
