#!/bin/bash

## Fisher + SWBD 数据集

exp_path=/mnt/lustre/sjtu/home/hs627/kaldi/egs/swbd/s5c

lattice_path="${exp_path}/exp/nnet3_ali/lat.*.gz"
text_path="${exp_path}/data/train_nodup_sp_hires/split4/*/text"

# 词表
vocab=${exp_path}/data/lang/words.txt


# ----

_=/mnt/lustre/sjtu/home/zkz01/newkaldi/egs/aishell/s5/exp/chain/tdnn_1a_sp/align_train_sp
# 这个是带speed-perturb的数据集，如果你只要原始的 把所有带sp的去掉
