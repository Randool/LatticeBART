#!/bin/bash
#SBATCH --job-name=prepare_data_fisher_swbd
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=exp/prepare_data_fisher_swbd-%A-%a.out
#SBATCH --error=exp/prepare_data_fisher_swbd-%A-%a.err
#SBATCH --exclude=gqxx-01-[003,006,011,016,020,033,083,093,105,110,125,139,145,146,152,167,174]
#SBATCH --array=0

set -e

stage=2

MAX_ARCS=200
MIN_BEAM=2
MAX_BEAM=6
BEAM=4
acwt=0.8
WIP=0

data_path=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd
vocab=${data_path}/vocab

if [ $stage -le 0 ]; then
  start=${SECONDS}

  # test data
  echo '============= EVAL2000 & RT03 ============='
  ./lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz"     -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -a ${acwt}
  ./lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b ${BEAM} -a ${acwt}

  # train data
  echo '============= TRAIN ============='
  for beam in $(seq ${MIN_BEAM} ${MAX_BEAM}) ; do
    ./lattice_prune.sh -p "${data_path}/data_train_new/swbd_lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b "${beam}" -a ${acwt}
    ./lattice_prune.sh -p "${data_path}/data_train_new/fisher_lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b "${beam}" -a ${acwt}
  done

  # Time cost
  duration=$((SECONDS-start))
  echo "Cost ${duration} s"  # about 4.7h
fi


if [ $stage -le 1 ]; then
  start=${SECONDS}

  # cat "${data_path}/rt03.ark" "${data_path}/eval2000.ark" > "${data_path}/test.ark"
  cat "${data_path}/rt03.txt" "${data_path}/eval2000.txt" > "${data_path}/test.txt"

  echo '============= TRAIN & TEST ark ============='
  python prepare_lattice_id_ark.py -f "${data_path}/data_train_new/pruned.*.b*.a${acwt}.l1.w${WIP}.txt" -o "${data_path}/train.ark"

  # Split train and dev data
  grep -v 'fe_03_11487-B-003109-023406' "${data_path}/train.ark" | shuf > "${data_path}/tmp.ark"

  total_lines=$(wc -l "${data_path}/tmp.ark" | awk '{print $1}')
  train_lines=$(( total_lines - 10000 ))
  echo "${total_lines}", ${train_lines}, 10000

  head -n ${train_lines} "${data_path}/tmp.ark" | sort > "${data_path}/train.ark"
  tail -n 10000 "${data_path}/tmp.ark" | sort > "${data_path}/dev.ark"

  t_lines=$(echo "${total_lines}" | awk '{printf("%d",$1*0.2)}')
  head -n "${t_lines}" "${data_path}/train.ark" > "${data_path}/train.20.ark"

  rm "${data_path}/tmp.ark"
  cat "${data_path}/fisher.txt" "${data_path}/swbd.txt" > "${data_path}/train.txt"

  # Time cost
  duration=$((SECONDS-start))
  echo "Cost ${duration} s"
fi


if [ $stage -le 2 ]; then
  start=${SECONDS}

  echo '============= EVAL2000 & RT03 ark ============='
  python prepare_lattice_id_ark.py -f "${data_path}/decode_rt03/pruned.lat.*.b${BEAM}.a${acwt}.l1.w${WIP}.txt" -o "${data_path}/rt03.b${BEAM}.ark"
  python prepare_lattice_id_ark.py -f "${data_path}/decode_eval2000/pruned.lat.*.b${BEAM}.a${acwt}.l1.w${WIP}.txt" -o "${data_path}/eval2000.b${BEAM}.ark"

  # Time cost
  duration=$((SECONDS-start))
  echo "Cost ${duration} s"
fi