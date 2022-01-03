#!/bin/bash
#SBATCH --job-name=train_lat_bart_ark
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=70G
#SBATCH --output=exp/train_lat_bart_ark-%A-%a.out
#SBATCH --error=exp/train_lat_bart_ark-%A-%a.err
#SBATCH --exclude=gqxx-01-[008,011,016,020,033,083,093,105,106,110,116,125,139,145,146,152,167,174]
#SBATCH --array=5,6

hostname
export PYTHONPATH="${HOME}/Projects/LatticeBART"

lat_bart_path=none
exp_dir=exp/LatBart

epoch=8
use_marginal=1
use_min_depth=1
use_max_depth=1
lattice_mask=1
forward_backward=1
homophone_replace=0.05
token_mask=0
depth_disturb=0

decoder_input=sent
use_metric=bleu

init_model=0
lr=1e-5
weight_decay=1e-6
seed=66

case ${SLURM_ARRAY_TASK_ID} in
1)
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  lr=1e-4
  exp_dir=exp/LatBart_20
  ;;
2)
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
  lr=1e-4
  exp_dir=exp/LatBart_100
  ;;
3)
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  exp_dir=exp/LatBart_20
  ;;
4)
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
  exp_dir=exp/LatBart_100
  ;;
5)
  decoder_input=lattice
  use_metric=loss
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.80.ark"
  exp_dir=exp/L2L_80
  homophone_replace=0.1
  token_mask=0.1
  ;;
6)
  decoder_input=lattice
  use_metric=loss
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.80.ark"
  exp_dir=exp/L2L_80
  homophone_replace=0.1
  token_mask=0.1
  ;;
*)
  echo "Wrong SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
  exit 1
  ;;
esac

echo "decoder_input: ${decoder_input}, init_model: ${init_model}, train_lat_ark: ${train_lat_ark}"
echo "use_marginal: ${use_marginal}, forward_backward: ${forward_backward}, homophone_replace: ${homophone_replace}"
echo "token_mask: ${token_mask}, depth_disturb: ${depth_disturb}"

python -W ignore train_lattice_bart.py \
  --lat_enc_layers 6 \
  --lat_bart_path "${lat_bart_path}" \
  --bart_path "${HOME}/Dataset/transformers/bart-base" \
  --init_model ${init_model} \
  \
  --lr ${lr} \
  --weight_decay ${weight_decay} \
  --lr_decay 0.95 \
  --grad_acc_steps 3 \
  --clip 1e-1 \
  \
  --train_lat_ark "${train_lat_ark}" \
  --train_text    "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --eval_lat_ark "${HOME}/Dataset/mr_fisher_swbd/dev.ark" \
  --eval_text    "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --test_lat_ark "${HOME}/Dataset/mr_fisher_swbd/eval2000.b4.ark" "${HOME}/Dataset/mr_fisher_swbd/rt03.b4.ark" \
  --test_text    "${HOME}/Dataset/mr_fisher_swbd/test.txt" \
  \
  --lm_scale 1.0 \
  --ac_scale 1.0 \
  --min_nodes 1 \
  --max_nodes 200 \
  --use_marginal ${use_marginal} \
  --use_min_depth ${use_min_depth} \
  --use_max_depth ${use_max_depth} \
  --use_forward_backward ${forward_backward} \
  --use_lattice_mask ${lattice_mask} \
  --homophone_replace ${homophone_replace} \
  --token_mask ${token_mask} \
  --depth_disturb ${depth_disturb} \
  \
  --device cuda:0 \
  --parallel DDP \
  --seed ${seed} \
  --optim adam \
  --exp_dir ${exp_dir} \
  --epoch ${epoch} \
  --train_batch 16 \
  --eval_batch 4 \
  --num_workers 12 \
  --log_every 1000 \
  --patience 3 \
  --use_metric ${use_metric} \
   \
  --decoder_input ${decoder_input} \
  --always_save 1 \
  \
  --num_beams 4 \
  --min_len 2 \
  --max_len 256
