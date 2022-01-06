#!/bin/bash
#SBATCH --job-name=pretrain_lat_bart_ark
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=70G
#SBATCH --output=exp/pretrain_lat_bart_ark-%A-%a.out
#SBATCH --error=exp/pretrain_lat_bart_ark-%A-%a.err
#SBATCH --exclude=gqxx-01-[008,011,016,020,033,083,093,105,110,116,125,139,145,146,152,167,174]
#SBATCH --array=1,2,3,4

hostname
export PYTHONPATH="${HOME}/Projects/LatticeBART"

lat_bart_path=none
exp_dir=exp/lat_bart_ark

seed=66
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
use_metric=loss

init_model=0
lr=1e-5
weight_decay=1e-6

pass_eval=0

case ${SLURM_ARRAY_TASK_ID} in
1)
  # L2S + init + 20%
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
#  train_lat_ark=none
  lr=1e-4
  seed=99
  ;;
2)
  # L2S + init + 100%
  decoder_input=sent
  use_metric=bleu
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
#  train_lat_ark=none
  lr=1e-4
  ;;
3)
  # L2S + 20%
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
#  train_lat_ark=none
  seed=666
  ;;
4)
  # L2S + 100%
  decoder_input=sent
  use_metric=bleu
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
#  train_lat_ark=none
  ;; ################################################################
5)
  # L2L + init + 80% (--> L2S + 20%)
  decoder_input=lattice
  use_metric=loss
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.80.ark"
  seed=6666
  lr=1e-4
  use_marginal=0.9
  forward_backward=0.9
  homophone_replace=0.05
  token_mask=0.1
  depth_disturb=10
  ;;
6)
  # L2L + 80% (--> L2S + 20%)
  decoder_input=lattice
  use_metric=loss
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.80.ark"
  seed=8888
  use_marginal=1
  forward_backward=1
  homophone_replace=0.05
  token_mask=0.05
  ;;
7)
  # L2L + init + 100% (--> L2S + 100%)
  decoder_input=lattice
  use_metric=loss
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
  lr=1e-4
  use_marginal=0.9
  forward_backward=0.9
  homophone_replace=0.05
  token_mask=0.1
  depth_disturb=10
  ;;
8)
  # L2L + 100% (--> L2S + 100%), ## 还是炸了
  decoder_input=lattice
  use_metric=loss
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.ark"
  seed=88
  use_marginal=0.9
  forward_backward=0.9
  homophone_replace=0.05
  token_mask=0.1
  depth_disturb=10
  ;;
9)
  # L2L + init + 20%
  decoder_input=lattice
  use_metric=loss
  init_model=1
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  ;;
10)
  # L2L + 20%
  decoder_input=lattice
  use_metric=loss
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  ;;
11)
  # (L2L + init + 80% -->) L2S + 20%
  decoder_input=sent
  use_metric=wer
  init_model=0
  lr=1e-4
#  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  train_lat_ark=none
  seed=661
  lat_bart_path=exp/lat_bart_ark/SZ-base,init,D-L,nlayer-6,lr-0.0001,optim-adam,pall-DDP,seed-6666,batch-48,gpu-4,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;
12)
  # (L2L + 80% -->) L2S + 20%
  decoder_input=sent
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  # train_lat_ark=none
  seed=660
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-8888,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.05,Tmask-0.10,Dd-0
  exp_dir=exp/lat_bart_ark_1
  ;;
19)
  # (L2L + init + 20% -->) L2S + 20%
  decoder_input=sent
  use_metric=wer
  init_model=0
  lr=1e-4
#  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  train_lat_ark=none
  seed=663
  lat_bart_path=exp/lat_bart_ark/SZ-base,init,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-666,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0
  exp_dir=exp/lat_bart_ark_1
  ;;
20)
  # (L2L + 20% -->) L2S + 20%, !! 因为80%上的训练似乎过拟合了，所以紧急之下用20%代替一下
  decoder_input=sent
  use_metric=wer
  init_model=0
#  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  train_lat_ark=none
  seed=662
#  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-666,batch-48,gpu-4,m-1.00,mD-1.00,MD-1.00,fb-1.00,MSK-1.00,homo-0.00,Tmask-0.00,Dd-0
  exp_dir=exp/lat_bart_ark_1
  ;;
21)
  # [7](L2L + init + 100%) --> L2S + 20%
  decoder_input=sent
  use_metric=wer
  init_model=0
  lr=1e-4
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=60
  lat_bart_path=exp/lat_bart_ark/SZ-base,init,D-L,nlayer-6,lr-0.0001,optim-adam,pall-DDP,seed-66,batch-48,gpu-4,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;
22)
  # (L2L + 100%) --> L2S + 20%
  decoder_input=sent
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-999,batch-48,gpu-8,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;


51)
  ####### 消融实验 #######
  # 1. 无 marginal score
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  use_marginal=0
  ;;
52)
  # 2. 无forward backward score
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  forward_backward=0
  ;;
53)
  # 3. 无positional
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  use_min_depth=0
  use_max_depth=0
  ;;
54)
  # 4. 无mask
  use_metric=wer
  init_model=0
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.20.ark"
  seed=666
  lattice_mask=0
  ;;


101)
  ##################################################
  ###### 无前置训练的 小规模数据实验 start at 100 ######
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.1.ark"
  seed=101
  ;;
102)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.2.ark"
  seed=102
  ;;
104)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.4.ark"
  seed=104
  ;;
108)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.8.ark"
  seed=108
  ;;
301)
  ######################################################
  ###### 做了L2L前置训练的 小规模数据实验 start at 300 ######
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.1.ark"
  seed=301
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-999,batch-48,gpu-8,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;
302)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.2.ark"
  seed=302
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-999,batch-48,gpu-8,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;
304)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.4.ark"
  seed=304
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-999,batch-48,gpu-8,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
  ;;
308)
  decoder_input=sent
  use_metric=wer
  init_model=0
  pass_eval=1
  epoch=3
  train_lat_ark="${HOME}/Dataset/mr_fisher_swbd/train.8.ark"
  seed=308
  lat_bart_path=exp/lat_bart_ark/SZ-base,D-L,nlayer-6,lr-1e-05,optim-adam,pall-DDP,seed-999,batch-48,gpu-8,m-0.90,mD-1.00,MD-1.00,fb-0.90,MSK-1.00,homo-0.05,Tmask-0.10,Dd-10
  exp_dir=exp/lat_bart_ark_1
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
  --pass_eval ${pass_eval} \
   \
  --decoder_input ${decoder_input} \
  --always_save 1 \
  \
  --num_beams 4 \
  --min_len 2 \
  --max_len 256
