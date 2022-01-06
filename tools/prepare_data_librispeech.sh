#!/bin/bash

set -e

MAX_ARCS=200
MAX_BEAM=5
BEAM=4

egs=/mnt/lustre/sjtu/home/lfd98/kaldi/egs/librispeech/s5
exp=${egs}/exp/tri6b

vocab=${egs}/data/lang/words.txt

# dev data
echo ============= DEV =============
./lattice_prune.sh -p "${exp}/decode_fglarge_dev_clean/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -l 15 -w 0.5
./lattice_prune.sh -p "${exp}/decode_fglarge_dev_other/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -l 15 -w 0.5

python prepare_lattice_id_ark.py -f "${exp}/decode_fglarge_dev_*/pruned.lat.*.m200.b*.l15.w0.5.txt" -o "${exp}/LatticeBART_dev.ark"

# test data
echo ============= TEST =============
./lattice_prune.sh -p "${exp}/decode_fglarge_test_clean/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -l 15 -w 0.5
./lattice_prune.sh -p "${exp}/decode_fglarge_test_other/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -l 15 -w 0.5

python prepare_lattice_id_ark.py -f "${exp}/decode_fglarge_test_*/pruned.lat.*.m200.b*.l15.w0.5.txt" -o "${exp}/LatticeBART_test.ark"

# train data
echo ============= TRAIN =============
for beam in $(seq 1 ${MAX_BEAM}) ; do
  ./lattice_prune.sh -p "${exp}/decode_fglarge_train_960/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b "${beam}" -l 15 -w 0.5
done
#./lattice_prune.sh "${exp}/decode_fglarge_train_2kshort/lat.*.gz" ${vocab}

python prepare_lattice_id_ark.py -f "${exp}/decode_fglarge_train_960/pruned.lat.*.m200.b*.l15.w0.5.txt" -o "${exp}/LatticeBART_train.ark"
