#!/bin/bash

set -e

usage() {
  echo "Usage: $0 -p <path> [-v <vocab>] [-a <ACWT>] [-l <LMWT>]" 1>&2;
  exit 1;
}

#### CONFIG ####

path=
vocab=
#BEAM=10
ACWT=1.0
LMWT=1.0

while getopts "p:v:b:a:l:" opt ; do
  case ${opt} in
    p) path=${OPTARG} ;;
    v) vocab=${OPTARG} ;;
#    b) BEAM=${OPTARG} ;;
    a) ACWT=${OPTARG} ;;
    l) LMWT=${OPTARG} ;;
    *) usage ;;
  esac
done

if [ -z "${path}" ]; then
  usage
fi

echo "path=${path}, vocab=${vocab}, ACWT=${ACWT}, LMWT=${LMWT}" 1>&2;

#### KALDI ####
KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

lattice_best_path=${KALDI_ROOT}/src/latbin/lattice-best-path
int2sym=${KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl

#### LOOP ####
tgt_file="${path}/1best.a${ACWT}.l${LMWT}.txt"

[[ -f "${tgt_file}" ]] && rm "${tgt_file}"

for lat in "${path}"/lat.*.gz; do
  echo " ${lat} >> ${tgt_file}"

  (if [ -z "${vocab}" ]; then
    ${lattice_best_path} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" "ark:gunzip -c ${lat} |" ark,t:"${tgt_file}"
  else
    # 'ark,t:|int2sym.pl -f 2- words.txt > text'
    ${lattice_best_path} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" \
      "ark:gunzip -c ${lat} |" \
      "ark,t:|${int2sym} -f 2- ${vocab} >> ${tgt_file}"
  fi) 2>>"${path}/best.log.txt"
done

tmp="${path}/tmp"
sort "${tgt_file}" > "${tmp}"
mv "${tmp}" "${tgt_file}"
