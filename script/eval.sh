#!/usr/bin/env bash
# Evaluate the released Lapras checkpoints on TSR (inference only); other datasets: replace tsr in both paths.
# Download the checkpoints into ckpt/ and the datasets into dataset/ first (README). Run from the repo root.
set -euo pipefail

# ChatTS-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29500 evaluation/evaluate.py \
  --ckpt ckpt/chatts/lapras_tsr \
  --test_file dataset/tsr/test_with_ts_tags.jsonl \
  --batch_size 8 \
  --max_new_tokens 512

# OpenTSLM-1B (Flamingo)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29500 evaluation/evaluate.py \
  --ckpt ckpt/opentslm/lapras_tsr \
  --test_file dataset/tsr/test_with_ts_tags.jsonl \
  --batch_size 8 \
  --max_new_tokens 512

# SLIP-1B
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29500 evaluation/evaluate.py \
  --ckpt ckpt/slip/lapras_tsr \
  --test_file dataset/tsr/test.jsonl \
  --batch_size 8 \
  --max_new_tokens 512
