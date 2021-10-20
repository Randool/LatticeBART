#!/bin/bash

set -e

MAX_ARCS=200

data_path=/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd
vocab=${data_path}/vocab

../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 3 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 3 -l 0.8
#../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz"     -v ${vocab} -m ${MAX_ARCS} -b 4 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 4 -l 0.8
../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 5 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_rt03/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 5 -l 0.8

../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 3 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 3 -l 0.8
#../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 4 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 4 -l 0.8
../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 5 -a 0.8
../lattice_prune.sh -p "${data_path}/decode_eval2000/lat.*.gz" -v ${vocab} -m ${MAX_ARCS} -b 5 -l 0.8

python ../prepare_lattice_id_ark.py -f "${data_path}/decode_rt03/pruned.lat.*.b3.*.txt" -o rt03_beam3.ark
python ../prepare_lattice_id_ark.py -f "${data_path}/decode_rt03/pruned.lat.*.b4.*.txt" -o rt03_beam4.ark
python ../prepare_lattice_id_ark.py -f "${data_path}/decode_rt03/pruned.lat.*.b5.*.txt" -o rt03_beam5.ark
python ../prepare_lattice_id_ark.py -f "${data_path}/decode_eval2000/pruned.lat.*.b3.*.txt" -o eval2000_beam3.ark
python ../prepare_lattice_id_ark.py -f "${data_path}/decode_eval2000/pruned.lat.*.b4.*.txt" -o eval2000_beam4.ark
python ../prepare_lattice_id_ark.py -f "${data_path}/decode_eval2000/pruned.lat.*.b5.*.txt" -o eval2000_beam5.ark

cat rt03_beam3.ark rt03_beam4.ark rt03_beam5.ark eval2000_beam3.ark eval2000_beam4.ark eval2000_beam5.ark > beam345.ark
rm rt03_beam*.ark eval2000_beam*.ark

python ./select_lattice.py # batch_[123].txt

for i in 0 1 2; do
  if [ -f ./selected_${i}.ark ]; then rm ./selected_${i}.ark; fi
  while read line; do
    if [ -z "$line" ]; then continue; fi
    lattice_id=$(echo "${line}" | awk '{print $1}')
    grep "${lattice_id}" ./beam345.ark | \
    awk '{printf("%s-%d",$1,NR); for(i=2;i<=NF;i++){printf(" %s",$i)}; printf("\n")}' >> ./selected_${i}.ark
  done <batch_${i}.txt
done
