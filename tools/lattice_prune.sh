#!/bin/bash

set -e

usage() {
  echo "Usage: $0 -p <pattern> [-v <vocab>] [-m <max_arc>] [-b <beam>] [-a <ACWT>] [-l <LMWT>] [-w <WIP>]" 1>&2;
  exit 1;
}

#### CONFIG ####
pattern=
vocab=
MAX_ARCS=200
BEAM=10
ACWT=1
LMWT=1
WIP=0

while getopts "p:v:m:b:a:l:w:" opt ; do
  case ${opt} in
    p) pattern=${OPTARG} ;;
    v) vocab=${OPTARG} ;;
    m) MAX_ARCS=${OPTARG} ;;
    b) BEAM=${OPTARG} ;;
    a) ACWT=${OPTARG} ;;
    l) LMWT=${OPTARG} ;;
    w) WIP=${OPTARG} ;;
    *) usage ;;
  esac
done

if [ -z "${pattern}" ]; then
  usage
fi

echo "pattern=${pattern}, vocab=${vocab}, MAX_ARCS=${MAX_ARCS}, BEAM=${BEAM}, ACWT=${ACWT}, LMWT=${LMWT}, WIP=${WIP}" 1>&2;

#### KALDI ####
KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

lattice_scale=${KALDI_ROOT}/src/latbin/lattice-scale
lattice_add_penalty=${KALDI_ROOT}/src/latbin/lattice-add-penalty
lattice_prune=${KALDI_ROOT}/src/latbin/lattice-determinize-pruned
lattice_minimize=${KALDI_ROOT}/src/latbin/lattice-minimize
int2sym=${KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl

#### LOOP ####
for lat in ${pattern}; do {
  dir=$(dirname "${lat}")
  name=$(basename "${lat}" '.gz')
  tgt_file="${dir}/pruned.${name}.m${MAX_ARCS}.b${BEAM}.a${ACWT}.l${LMWT}.w${WIP}.txt"
  log="${dir}/pruned.m${MAX_ARCS}.b${BEAM}.l${LMWT}.w${WIP}.log"

  echo " ${lat} >> ${tgt_file}"

  ( if [ "$(echo "${WIP}" | awk '{print($1 != 0)}')" -eq 0 ]; then
      if [ -z "${vocab}" ]; then
        ${lattice_scale} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" "ark:gunzip -c ${lat} |" ark:- | \
        ${lattice_prune} --max-arcs="${MAX_ARCS}" --beam="${BEAM}" ark:- ark:- | \
        ${lattice_minimize} ark:- "ark,t:${tgt_file}"
      else
        ${lattice_scale} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" "ark:gunzip -c ${lat} |" ark:- | \
        ${lattice_prune} --max-arcs="${MAX_ARCS}" --beam="${BEAM}" ark:- ark:- | \
        ${lattice_minimize} ark:- ark,t:- | \
        ${int2sym} -f 3 "${vocab}" > "${tgt_file}"
      fi
    else
      if [ -z "${vocab}" ]; then
        ${lattice_scale} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" "ark:gunzip -c ${lat} |" ark:- | \
        ${lattice_add_penalty} --word-ins-penalty="${WIP}" ark:- ark:- | \
        ${lattice_prune} --max-arcs="${MAX_ARCS}" --beam="${BEAM}" ark:- ark:- | \
        ${lattice_minimize} ark:- "ark,t:${tgt_file}"
      else
        ${lattice_scale} --acoustic-scale="${ACWT}" --lm-scale="${LMWT}" "ark:gunzip -c ${lat} |" ark:- | \
        ${lattice_add_penalty} --word-ins-penalty="${WIP}" ark:- ark:- | \
        ${lattice_prune} --max-arcs="${MAX_ARCS}" --beam="${BEAM}" ark:- ark:- | \
        ${lattice_minimize} ark:- ark,t:- | \
        ${int2sym} -f 3 "${vocab}" > "${tgt_file}"
      fi
    fi
  ) 2>>"${log}"

} & done
wait
