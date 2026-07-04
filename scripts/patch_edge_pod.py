#!/usr/bin/env python3
"""One-off patches for upstream EDGE on RunPod (PyTorch 2.6 weights_only default).

PyTorch 2.6 changed ``torch.load``'s default to ``weights_only=True``. EDGE's
checkpoint stores a ``dataset.preprocess.Normalizer`` object, so it must be loaded
with ``weights_only=False``.
"""
from __future__ import annotations

import os
from pathlib import Path

EDGE = Path(os.environ.get("EDGE_CODE_PATH", "/workspace/EDGE"))

edge_py = EDGE / "EDGE.py"
if edge_py.exists():
    body = edge_py.read_text()
    needle = "checkpoint_path, map_location=self.accelerator.device"
    patched = needle + ", weights_only=False"
    if needle in body and patched not in body:
        edge_py.write_text(body.replace(needle, patched))
        print("patched EDGE.py torch.load weights_only")
    else:
        print("EDGE.py already patched or pattern missing")
else:
    print(f"EDGE.py not found at {edge_py}")
