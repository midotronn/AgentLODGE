#!/usr/bin/env python3
"""One-off patches for upstream LODGE on RunPod (PyTorch 2.6 + tensor types)."""
from __future__ import annotations

import os
import re
from pathlib import Path

LODGE = Path(os.environ.get("LODGE_CODE_PATH", "/workspace/LODGE"))

# Local_Module: normalizer accepts Tensor or ndarray
local = LODGE / "dld/models/modeltype/Local_Module.py"
text = local.read_text()
old = (
    "                orikey[idx_] = self.normalizer.normalize("
    "torch.from_numpy(orikey[idx_])).detach().cpu().numpy()"
)
new = (
    "                item = orikey[idx_]\n"
    "                if not isinstance(item, torch.Tensor):\n"
    "                    item = torch.from_numpy(item)\n"
    "                orikey[idx_] = self.normalizer.normalize(item.float()).detach().cpu().numpy()"
)
if old in text:
    local.write_text(text.replace(old, new))
    print("patched Local_Module normalizer loop")
else:
    print("Local_Module already patched or pattern missing")

# Global/Local modules: weights_only=False for normalizer pickle (kwarg must go INSIDE torch.load)
for name in ("Global_Module.py", "Local_Module.py"):
    path = LODGE / "dld/models/modeltype" / name
    body = path.read_text()
    # Repair a previously mis-applied patch that placed the kwarg outside torch.load(),
    # which produced a SyntaxError (`torch.load(...)), weights_only=False`).
    broken = 'normalizer")), weights_only=False'
    if broken in body:
        path.write_text(body.replace(broken, 'normalizer"), weights_only=False)'))
        print(f"repaired {name} torch.load kwarg placement")
        continue
    needle = 'torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))'
    repl = 'torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"), weights_only=False)'
    if needle in body and repl not in body:
        path.write_text(body.replace(needle, repl))
        print(f"patched {name} torch.load")

# Replace the upstream author's hardcoded absolute SMPLX joint path with the local
# relative one (LODGE inference runs with cwd = LODGE root, so 'data/...' resolves).
ABS_JPATH = "/data2/lrh/project/dance/Lodge/lodge_pub/data/smplx_neu_J_1.npy"
REL_JPATH = "data/smplx_neu_J_1.npy"
for py in LODGE.rglob("*.py"):
    body = py.read_text()
    if ABS_JPATH in body:
        py.write_text(body.replace(ABS_JPATH, REL_JPATH))
        print(f"patched absolute Jpath in {py.relative_to(LODGE)}")

# smplfk / fk_vis: stdlib pickle instead of pickle5
for rel in (
    "dld/data/render_joints/smplfk.py",
    "dld/data/utils/fk_vis.py",
):
    path = LODGE / rel
    if not path.exists():
        continue
    body = path.read_text()
    if "import pickle5 as pickle" in body:
        path.write_text(body.replace("import pickle5 as pickle", "import pickle"))
        print(f"patched {rel} pickle import")

# numpy 2.x: deprecated np.float alias
for py in LODGE.rglob("*.py"):
    body = py.read_text()
    if "np.float" not in body:
        continue
    new_body = re.sub(r"\bnp\.float\b", "np.float64", body)
    if new_body != body:
        py.write_text(new_body)
        print(f"patched np.float in {py.relative_to(LODGE)}")
