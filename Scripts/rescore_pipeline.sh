#!/bin/bash

set -e

eval2000=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/decode_eval2000
rt03=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/decode_rt03
vocab=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/vocab
lm_scale=7

KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

int2sym=${KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl
lattice_to_nbest=${KALDI_ROOT}/src/latbin/lattice-to-nbest
nbest_to_linear=${KALDI_ROOT}/src/latbin/nbest-to-linear
# Usage: nbest-to-linear [options] <nbest-rspecifier> <alignments-wspecifier> [<transcriptions-wspecifier> [<lm-cost-wspecifier> [<ac-cost-wspecifier>]]]

for n in 10 20; do
  for D in ${eval2000} ${rt03}; do
    echo "=========================="
    best_file=${D}/${n}best.txt
    for lat in "${D}"/lat.*.gz; do
      bn=$(dirname "${lat}")/$(basename "${lat}" .gz)
      ali="${bn}.ali"
      text="${bn}.text.n${n}"
      lmwt="${bn}.lmwt.n${n}"
      acwt="${bn}.acwt.n${n}"
      rescore="${bn}.rescore.n${n}"
      ac_lm="${rescore}.ac_lm.n_${n}"
      best="${bn}.best.n${n}"

      ${lattice_to_nbest} --acoustic-scale=1.0 --lm-scale=1.0 --n=${n} "ark:gunzip -c ${lat} |" ark:- | \
      ${nbest_to_linear} ark:- ark:"${ali}" \
        "ark,t:|${int2sym} -f 2- ${vocab} > ${text}" \
        ark,t:"${lmwt}" \
        ark,t:"${acwt}"

      echo "Rescoring on ${text} -> ${rescore}"
      # LSTM rescore
      python -W ignore lm_rescore.py \
        --ckpt /mnt/lustre/sjtu/home/lfd98/remote/post_process/exp/rescore_lm/epoch_1 \
        --vocab /mnt/lustre/sjtu/home/lfd98/remote/post_process/exp/rescore_lm \
        --sent_file "${text}" \
        --write_file "${rescore}"

      paste "${acwt}" "${rescore}" > "${ac_lm}"
      python ./tools/select_nbest_best.py \
        --text_file "${text}" \
        --score_file "${ac_lm}" \
        --lm_scale ${lm_scale} \
        --write "${best}"

      cat "${best}" >> ${best_file}

      # Do clean-up
      rm "${ali}" "${lmwt}" "${acwt}" "${rescore}" "${best}"
    done
  done
done
