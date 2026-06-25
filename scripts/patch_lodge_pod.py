#!/usr/bin/env python3
"""One-off patches for upstream LODGE on RunPod (PyTorch 2.6 + tensor types)."""
from pathlib import Path

LODGE = Path("/workspace/LODGE")

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

# Global/Local modules: weights_only=False for normalizer pickle
for name in ("Global_Module.py", "Local_Module.py"):
    path = LODGE / "dld/models/modeltype" / name
    body = path.read_text()
    needle = 'torch.load(eval(f"cfg.DATASET.{dataname.upper()}.normalizer"))'
    repl = needle + ", weights_only=False"
    if needle in body and repl not in body:
        path.write_text(body.replace(needle, repl))
        print(f"patched {name} torch.load")
