#!/bin/bash
#SBATCH --job-name=process_data_for_lattice_bart
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=exp/process_data_for_lattice_bart-%A-%a.out
#SBATCH --error=exp/process_data_for_lattice_bart-%A-%a.err
#SBATCH --mail-type=end
#SBATCH --mail-user=dlf43@qq.com
#SBATCH --exclude=gqxx-01-[003,006,011,016,020,033,083,093,105,110,125,139,145,146,152,167,174]
#SBATCH --array=1

marginal=1
lattice_mask=1
forward_backward=1

HOME=/mnt/lustre/sjtu/home/lfd98
KALDI_EGS=${HOME}/kaldi/egs/librispeech/s5

python process_data.py \
  --cache_dir exp/data_cache \
  --train_lats "${KALDI_EGS}/exp/tri6b/decode_fglarge_train_960/pruned.lat.*.m200.b*.l15.w0.5.txt" \
  --eval_lats "${KALDI_EGS}/exp/tri6b/decode_fglarge_dev_*/pruned.lat.*.txt" \
  --test_lats "${KALDI_EGS}/exp/tri6b/decode_fglarge_test_*/pruned.lat.*.txt" \
  --train_text  ${KALDI_EGS}/data/train_text \
  --eval_text   ${KALDI_EGS}/data/dev_text \
  --test_text   ${KALDI_EGS}/data/test_text \
  \
  --sort_by_len 1 \
  --lm_scale 1.0 \
  --ac_scale 1.0 \
  --min_nodes 1 \
  --max_nodes 200 \
  --use_marginal ${marginal} \
  --use_min_depth 1 \
  --use_max_depth 1 \
  --use_forward_backward ${forward_backward} \
  --use_lattice_mask ${lattice_mask} \
