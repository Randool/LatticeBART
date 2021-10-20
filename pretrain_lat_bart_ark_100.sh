#!/bin/bash
#SBATCH --job-name=pretrain_L2L
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=80G
#SBATCH --output=exp/pretrain_L2L-%A-%a.out
#SBATCH --error=exp/pretrain_L2L-%A-%a.err
#SBATCH --exclude=gqxx-01-[008,011,016,020,033,083,093,105,110,116,118,125,139,145,146,152,167,174]
#SBATCH --array=9,99,999

#lat_bart_path=none
exp_dir=exp/lat_bart_ark

init_model=0

marginal=0.9
use_min_depth=1
use_max_depth=1
lattice_mask=1
forward_backward=0.9
homophone_replace=0.05
token_mask=0.1
depth_disturb=10

seed=${SLURM_ARRAY_TASK_ID}

echo "decoder_input: ${decoder_input}, init_model: ${init_model}, train_lat_ark: ${train_lat_ark}"
echo "marginal: ${marginal}, forward_backward: ${forward_backward}, homophone_replace: ${homophone_replace}"
echo "token_mask: ${token_mask}, depth_disturb: ${depth_disturb}"

#  --bart_path "${HOME}/Dataset/transformers/bart-base" \

python -W ignore train_lattice_bart.py \
  --lat_enc_layers 6 \
  --lat_bart_path exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-666,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0 \
  --init_model ${init_model} \
  \
  --lr 1e-5 \
  --weight_decay 1e-6 \
  --lr_decay 0.95 \
  --grad_acc_steps 3 \
  --clip 1e-1 \
  \
  --train_lat_ark "${HOME}/Dataset/mr_fisher_swbd/train.ark" \
  --train_text    "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --eval_lat_ark  "${HOME}/Dataset/mr_fisher_swbd/dev.ark" \
  --eval_text    "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --test_lat_ark "${HOME}/Dataset/mr_fisher_swbd/eval2000.ark" "${HOME}/Dataset/mr_fisher_swbd/rt03.ark" \
  --test_text    "${HOME}/Dataset/mr_fisher_swbd/test.txt"     "${HOME}/Dataset/mr_fisher_swbd/test.txt" \
  \
  --lm_scale 1.0 \
  --ac_scale 1.0 \
  --min_nodes 1 \
  --max_nodes 200 \
  --use_marginal ${marginal} \
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
  --epoch 8 \
  --train_batch 16 \
  --eval_batch 4 \
  --num_workers 12 \
  --log_every 1000 \
  --patience 3 \
  --use_metric loss \
   \
  --decoder_input lattice \
  --always_save 1 \
  \
  --num_beams 4 \
  --min_len 2 \
  --max_len 256
