# AgentLODGE

End-to-end pipeline that accepts a song and a costume description, generates dances with **Lodge++** and **EDGE** in parallel, selects the best result with an LLM agent, and generates a costume illustration.

## Pipeline

1. **Audio preprocessing** — Librosa 35-dim features (Lodge++) and Jukebox embeddings (EDGE) at 30 FPS
2. **Parallel dance generation** — Lodge++ global+PDDM and EDGE long-form (5s clips, 2.5s overlap)
3. **Dance selection agent** — Claude compares beat alignment, motion diversity, and song metadata
4. **Costume image generation** — DALL-E 3 or Gemini Imagen

## Requirements

- Python 3.10+
- [Lodge++](https://li-ronghui.github.io/lodgepp) codebase and pretrained weights
- [EDGE](https://edge-dance.github.io) codebase and checkpoint
- API keys for Anthropic (selection) and OpenAI or Gemini (costume image)

## Setup

```bash
cd AgentLODGE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and model paths
```

Ensure Lodge++ and EDGE are cloned and configured separately (see their READMEs). Point `LODGE_CODE_PATH`, `EDGE_CODE_PATH`, and weight paths in `.env` to your local installs.

Recommended defaults if you use the sibling `Runs/` layout in this workspace:

```env
LODGE_CODE_PATH=../Runs/LODGE
EDGE_CODE_PATH=../Runs/EDGE
LODGE_WEIGHTS_PATH=../Runs/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt
LODGE_GLOBAL_WEIGHTS_PATH=../Runs/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt
EDGE_WEIGHTS_PATH=../Runs/EDGE/checkpoint.pt
```

Run Lodge++ and EDGE inference from their own virtual environments if dependency sets differ; the pipeline subprocesses inherit the active Python environment.

## Usage

```bash
python run_pipeline.py \
  --audio path/to/song.wav \
  --costume "a flowing red ballgown with silver embroidery" \
  --output_dir ./outputs
```

## Outputs

Written to `output_dir`:

| File | Description |
|------|-------------|
| `selected_dance.npy` | Selected motion array `(L, 139)` in SMPL format |
| `costume_output.png` | Generated costume illustration |
| `pipeline_log.json` | Selection reasoning, metrics, and errors |

## Configuration

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude selection agent |
| `OPENAI_API_KEY` | Required if `IMAGE_BACKEND=openai` |
| `GEMINI_API_KEY` | Required if `IMAGE_BACKEND=gemini` |
| `IMAGE_BACKEND` | `openai` or `gemini` |
| `OUTPUT_DIR` | Default output directory |
| `LODGE_CODE_PATH` | Path to Lodge++ repo |
| `EDGE_CODE_PATH` | Path to EDGE repo |
| `LODGE_WEIGHTS_PATH` | Local (PDDM) checkpoint |
| `LODGE_GLOBAL_WEIGHTS_PATH` | Global choreography checkpoint |
| `EDGE_WEIGHTS_PATH` | EDGE model checkpoint |
| `LODGE_GENRE` | FineDance genre label (default: `Hiphop`) |

## Error handling

- Lodge++ failure → falls back to EDGE
- EDGE failure → falls back to Lodge++
- Both fail → `RuntimeError`
- Image generation failure → logged; dance output still saved
- Selection agent failure → defaults to Lodge++

## License

Prototype integration code. Lodge++ and EDGE are subject to their respective project licenses.
