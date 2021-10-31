#!/bin/bash
#SBATCH --job-name=train_rescore_lm
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=exp/train_rescore_lm-%A-%a.out
#SBATCH --error=exp/train_rescore_lm-%A-%a.err
#SBATCH --exclude=gqxx-01-[011,016,020,033,083,093,105,110,116,125,139,145,146,152,167,174]
#SBATCH --array=0


python train_rescore_lm.py \
  --train_text "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --dev_text   "${HOME}/Dataset/mr_fisher_swbd/test.txt" \
  \
  --embed_size 128 \
  --hidden_size 128 \
  --n_layers 2 \
  --dropout 0.2 \
  \
  --lr 1e-4 \
  \
  --device cuda:0 \
  --seed 1023 \
  --exp_dir exp/rescore_lm \
  --epoch 4 \
  --train_batch 32 \
  --eval_batch 64 \
  --log_every 5000
