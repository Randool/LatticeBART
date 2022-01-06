#!/bin/bash
#SBATCH --job-name=train_LatticeBART_fisher_swbd
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=70G
#SBATCH --output=/mnt/lustre/sjtu/home/lfd98/Projects/LatticeBART/exp/train_LatticeBART_fisher_swbd-%A-%a.log
#SBATCH --error=/mnt/lustre/sjtu/home/lfd98/Projects/LatticeBART/exp/train_LatticeBART_fisher_swbd-%A-%a.log
#SBATCH --exclude=gqxx-01-[008,011,016,020,033,083,093,105,106,110,116,125,139,145,146,152,167,174]
#SBATCH --dependency=29568034
#SBATCH --array=7,8,9

hostname

lat_bart_path=none

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

export PYTHONPATH="${HOME}/Projects/LatticeBART"
exp_root="${PYTHONPATH}/exp"
exp_dir="${exp_root}/fisher_swbd"

data_root="${HOME}/Dataset/ASR/mr_fisher_swbd"

case ${SLURM_ARRAY_TASK_ID} in
1)
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${data_root}/train.20.ark"
  lr=1e-4
  exp_dir="${exp_dir}/LatBart_20"
  ;;
2)
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${data_root}/train.ark"
  lr=1e-4
  exp_dir="${exp_dir}/LatBart_100"
  ;;
3)
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${data_root}/train.20.ark"
  exp_dir="${exp_dir}/LatBart_20"
  ;;
4)
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${data_root}/train.ark"
  exp_dir="${exp_dir}/LatBart_100"
  ;;
5)
  decoder_input=lattice
  use_metric=loss
  init_model=0
  train_lat_ark="${data_root}/train.80.ark"
  exp_dir="${exp_dir}/L2L_80"
  homophone_replace=0.1
  token_mask=0.1
  ;;
6)
  decoder_input=lattice
  use_metric=loss
  init_model=1
  train_lat_ark="${data_root}/train.80.ark"
  exp_dir="${exp_dir}/L2L_80"
  homophone_replace=0.1
  token_mask=0.1
  ;;
7)
  # The below experiments use ckpt inheriting BART parameters and trained on Librispeech
  lat_bart_path="${exp_root}/Librispeech/L2L/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0"
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${data_root}/train.1.ark"
  exp_dir="${exp_dir}/L2S_1"
  homophone_replace=0.1
  token_mask=0.1
  ;;
8)
  # Use ckpt trained on Librispeech
  lat_bart_path="${exp_root}/Librispeech/L2L/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0"
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${data_root}/train.2.ark"
  exp_dir="${exp_dir}/L2S_2"
  homophone_replace=0.1
  token_mask=0.1
  ;;
9)
  # Use ckpt trained on Librispeech
  lat_bart_path="${exp_root}/Librispeech/L2L/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0"
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${data_root}/train.20.ark"
  exp_dir="${exp_dir}/L2S_20"
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
  --train_text    "${data_root}/train.txt" \
  --eval_lat_ark "${data_root}/dev.ark" \
  --eval_text    "${data_root}/train.txt" \
  --test_lat_ark "${data_root}/eval2000.b4.ark" "${data_root}/rt03.b4.ark" \
  --test_text    "${data_root}/test.txt" \
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
  --exp_dir "${exp_dir}" \
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
