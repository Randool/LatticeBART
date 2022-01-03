#!/bin/bash

set -e
export PYTHONPATH="${HOME}/Projects/LatticeBART"

eval2000=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/decode_eval2000
rt03=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/decode_rt03

KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

for lm_scale in 10 11 12 13 14 15 16; do
  for n in 10 20; do
    for D in ${eval2000} ${rt03}; do
      echo "====lm_scale: ${lm_scale}====n: ${n}===="
      best_file=${D}/${n}best_l${lm_scale}.txt
      for lat in "${D}"/lat.*.gz; do
        bn=$(dirname "${lat}")/$(basename "${lat}" .gz)
        text="${bn}.text.n${n}"
        rescore="${bn}.rescore.n${n}"
        ac_lm="${rescore}.ac_lm.n_${n}"
        best="${bn}.best.n${n}"

        python ./tools/select_nbest_best.py \
          --text_file "${text}" \
          --score_file "${ac_lm}" \
          --lm_scale ${lm_scale} \
          --write "${best}"

        cat "${best}" >> ${best_file}
      done
    done
  done
done