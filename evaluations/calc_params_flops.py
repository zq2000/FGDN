#!/usr/bin/env python3
"""Compute the parameter count and FLOPs of FGDN (Table 3 of the paper).

Usage (from the repository root):

    python evaluations/calc_params_flops.py

The architecture is instantiated with the default arguments, which equal the
`network_g` section of `options/*.yml`. FLOPs are measured with fvcore on a
1x3x256x256 input, following the paper protocol. Expected output:
~21.96M parameters and ~158.0G FLOPs.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from basicsr.archs.FGDN_arch import FGDN  # noqa: E402


def main() -> None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        sys.exit("fvcore is not installed. Install it first:  pip install fvcore")

    model = FGDN()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  ({n_params / 1e6:.2f}M)")

    dummy = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    print(f"FLOPs (256x256): {total:,}  ({total / 1e9:.1f}G)")


if __name__ == "__main__":
    main()
