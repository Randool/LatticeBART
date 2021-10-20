#!/bin/bash

set -e

usage() {
  echo "Usage: $0 -d <lattice_dir> -v <vocab> -t <refer-text> [-a <min_acwt>] [-A <max_acwt>] [-s <ac_step>]" 1>&2;
  exit 1;
}

lattice_dir=
vocab=
refer=

min_acwt=1
max_acwt=1
ac_step=1

while getopts "d:v:t:a:A:s:" opt ; do
  case ${opt} in
    d) lattice_dir=${OPTARG} ;;
    v) vocab=${OPTARG} ;;
    t) refer=${OPTARG} ;;
    a) min_acwt=${OPTARG} ;;
    A) max_acwt=${OPTARG} ;;
    s) ac_step=${OPTARG} ;;
    *) usage ;;
  esac
done

if [ -z "${lattice_dir}" ] || [ ! -d "${lattice_dir}" ] || [ -z "${vocab}" ] || \
   [ ! -f "${vocab}" ] || [ -z "${refer}" ] || [ ! -f "${refer}" ]; then
  usage
fi

echo "acwt=${min_acwt}:${max_acwt}:${ac_step}"

#### KALDI ####
KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

lattice_scale=${KALDI_ROOT}/src/latbin/lattice-scale
lattice_best_path=${KALDI_ROOT}/src/latbin/lattice-best-path
int2sym=${KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl
compute_wer=${KALDI_ROOT}/src/bin/compute-wer

#for log_file in /tmp/wer.*.log; do
#  if [ -e ${log_file} ]; then
#    rm ${log_file}
#  fi
#done

calculate_wer() {
  acwt=$1
  best_file="${lattice_dir}/acwt_${acwt}.best"

  [[ -e ${best_file} ]] && rm ${best_file}

  for lat in "${lattice_dir}"/lat.*.gz; do
    (
      ${lattice_scale} --acoustic-scale="${acwt}" "ark:gunzip -c ${lat} |" ark:- | \
      ${lattice_best_path} ark:- ark,t:- | \
      ${int2sym} -f 2- "${vocab}" | \
      sed -r 's/\[[^]]*]/ /g' | \
      sed 's/[ ][ ]*/ /g' >> ${best_file}
    ) 2>/dev/null
  done

  sort ${best_file} > ${best_file}.tmp
  mv ${best_file}.tmp ${best_file}

  wer=`${compute_wer} ark,t:${best_file} ark,t:"${refer}" | grep WER`
  echo "acwt=${acwt}, ${wer}"

  # %WER xxx [ xxx / xxx, 95 ins, 74 del, 1814 sub ]
#  res=`grep WER "${log_file}" | awk 'BEGIN{a=0; b=0} {a+=$4; b+=$6} END{printf("%d / %d, %.3f%", a, b, 100 * a / b)}'`
#  echo "acwt=${acwt}, ${res}"
}

#### LOOP ####

for acwt in $(seq "${min_acwt}" "${ac_step}" "${max_acwt}"); do
  calculate_wer ${acwt} &
done
wait

# 大小写转换：
# echo $dept | sed 's/ .*/\L&/g' | sed 's/(.*)/ /g' | sed 's/[ ][ ]*/ /g'

# ./find_best_weight.sh -d /mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/decode_eval2000 -v /mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/vocab -t /mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/eval2000.txt -a 0.5 -A 0.8 -s 0.05
# >> acwt=0.7 --> 36.133%
