#!/bin/bash

set -e

usage="Usage: $0 <lattice-pattern> <vocab> <text>"

if [ $# -ne 3 ]; then
  echo "${usage}"
  echo $#
  exit 1
fi

pattern=$1
vocab=$2
text=$3

#### KALDI ####
KALDI_ROOT=/home/randool/Applications/kaldi
if [ ! -d ${KALDI_ROOT} ]; then
  KALDI_ROOT=/mnt/lustre/sjtu/home/lfd98/kaldi
fi

lattice_oracle=${KALDI_ROOT}/src/latbin/lattice-oracle
sym2int=${KALDI_ROOT}/egs/wsj/s5/utils/sym2int.pl
int2sym=${KALDI_ROOT}/egs/wsj/s5/utils/int2sym.pl

#### OOV ID ####
OOV_id=$(awk 'BEGIN{id=-1}{if($1=="<UNK>" || $1 == "<unk>")id=$2}END{print(id)}' < "${vocab}")

#### PIPELINE ####
pipeline() {
  lat=$1
  vocab=$2
  text=$3
  OOV_id=$4

  oracle=$(dirname "${lat}")/$(basename "${lat}" .gz).oracle
  log=$(dirname "${lat}")/$(basename "${lat}" .gz).log

  echo "${lat} >> ${oracle}"

  (
    ${sym2int} --map-oov "${OOV_id}" -f 2- "${vocab}" < "${text}" |
    ${lattice_oracle} "ark:gunzip -c ${lat}|" ark:- ark,t:- |
    ${int2sym} -f 2- "${vocab}" > "${oracle}"
  ) 2>>"${log}"
}

#### LOOP ####
for lat in ${pattern}; do {
  pipeline "${lat}" "${vocab}" "${text}" "${OOV_id}"
} & done
wait

#### Collate ####
for lat in ${pattern}; do
  oracle=$(dirname "${lat}")/$(basename "${lat}" .gz).oracle
  final=$(dirname "${lat}")/oracle.txt
  cat "${oracle}" >> "${final}"
  rm "${oracle}"
done
