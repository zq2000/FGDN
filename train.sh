#!/usr/bin/env bash
# Single-GPU FGDN training on Rain200H (paper Sec. 4.1.3 protocol).
# For the other benchmarks, swap the option file, e.g. options/Rain200L.yml.
cd "$(dirname "$0")"
PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/Rain200H.yml
