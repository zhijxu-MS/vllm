#!/usr/bin/bash

reset
/root/miniconda3/bin/torchrun  --nproc-per-node 8  ptm.py