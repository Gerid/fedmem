from __future__ import annotations

import torch
from fedprotrack.models.torch_feature_adapter import _AdapterBlock

block = _AdapterBlock(64, 16)
x = torch.randn(5, 64)
out = block(x)
diff = (out - x).abs().max().item()
print(f"Max |adapter(x) - x| = {diff:.6f}  (should be 0.0)")
print(f"Up weight norm = {block.up.weight.data.norm().item():.6f}")
print(f"Up bias norm = {block.up.bias.data.norm().item():.6f}")
