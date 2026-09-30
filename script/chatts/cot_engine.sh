#!/usr/bin/env bash
# ChatTS-8B | CoT | Engine: train, then evaluate on the test set.
# Run from the repo root.
set -euo pipefail

deepspeed --include localhost:0,1,2,3 --master_port 29500 train.py \
  --deepspeed ds_config/ds_z2_bf16.json \
  --model_name_or_path ckpt/base/chatts \
  --template chatts \
  --dataset engine_ts_tags \
  --output_dir output/chatts/cot_engine \
  --do_train \
  --finetuning_type full \
  --dataset_dir dataset \
  --cutoff_len 10000 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.02 \
  --bf16 \
  --add_special_tokens "<|bot|>,<|eot|>" \
  --resize_vocab True \
  --trust_remote_code True \
  --disable_gradient_checkpointing True \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 2 \
  --logging_steps 10 \
  --save_strategy no \
  --save_safetensors False \
  --plot_loss \
  --report_to wandb \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --weight_decay 1e-2 \
  --ts_patch_size 60 \
  --stage sft \
  --use_cot True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29500 evaluation/evaluate.py \
  --ckpt output/chatts/cot_engine \
  --test_file dataset/engine/test_with_ts_tags.jsonl \
  --batch_size 8 \
  --max_new_tokens 512
