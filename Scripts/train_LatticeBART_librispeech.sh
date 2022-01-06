#!/bin/bash
#SBATCH --job-name=train_LatticeBART_librispeech
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=70G
#SBATCH --output=/mnt/lustre/sjtu/home/lfd98/Projects/LatticeBART/exp/train_LatticeBART_librispeech-%A-%a.log
#SBATCH --error=/mnt/lustre/sjtu/home/lfd98/Projects/LatticeBART/exp/train_LatticeBART_librispeech-%A-%a.log
#SBATCH --exclude=gqxx-01-[008,011,016,020,033,083,093,105,106,110,116,125,139,145,146,152,167,174]
#SBATCH --dependency=29568034
#SBATCH --array=5,6

hostname

lat_bart_path=none

epoch=8
use_marginal=1
use_min_depth=1
use_max_depth=1
lattice_mask=1
forward_backward=1
homophone_replace=0
token_mask=0
depth_disturb=0

decoder_input=sent
use_metric=bleu

init_model=0
lr=1e-5
weight_decay=1e-6
seed=66

export PYTHONPATH="${HOME}/Projects/LatticeBART"
exp_root="${PYTHONPATH}/exp/Librispeech"

librispeech_dir="${HOME}/kaldi/egs/librispeech/s5"
train_lat_ark="${librispeech_dir}/exp/tri6b/LatticeBART_train.ark"
eval_lat_ark="${librispeech_dir}/exp/tri6b/LatticeBART_dev.ark"
test_lat_ark="${librispeech_dir}/exp/tri6b/LatticeBART_test.ark"

train_text="${librispeech_dir}/data/train_text"
eval_text="${librispeech_dir}/data/dev_text"
test_text="${librispeech_dir}/data/test_text"

case ${SLURM_ARRAY_TASK_ID} in
1)
  decoder_input=sent
  use_metric=bleu
  init_model=1
  lr=1e-4
  exp_dir="${exp_root}/L2S_init"
  ;;
2)
  decoder_input=sent
  use_metric=bleu
  exp_dir="${exp_root}/L2S"
  ;;
3)
  decoder_input=lattice
  use_metric=loss
  init_model=1
  lr=1e-4
  exp_dir="${exp_root}/L2L_init"
  ;;
4)
  decoder_input=lattice
  use_metric=loss
  exp_dir="${exp_root}/L2L"
  ;;
5)
  # Use L2L checkpoint*
  lat_bart_path="${exp_root}/Librispeech/L2L/SZ-base,init,D-L,nlayer-6,lr-0.0001,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0"
  decoder_input=sent
  use_metric=bleu
  exp_dir="${exp_root}/L2L_init_L2S"
  ;;
6)
  # Use L2L checkpoint
  lat_bart_path="${exp_root}/Librispeech/L2L/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0"
  decoder_input=sent
  use_metric=bleu
  exp_dir="${exp_root}/L2L_L2S"
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
  --train_text    "${train_text}" \
  --eval_lat_ark "${eval_lat_ark}" \
  --eval_text    "${eval_text}" \
  --test_lat_ark "${test_lat_ark}" \
  --test_text    "${test_text}" \
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
