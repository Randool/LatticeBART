#!/bin/bash

egs=/mnt/lustre/sjtu/home/lfd98/kaldi/egs/librispeech/s5
exp=${egs}/exp/tri6b

vocab=${egs}/data/lang/words.txt

dev_text=${egs}/data/dev_text
test_text=${egs}/data/test_text
train_text=${egs}/data/train_text

# dev data
echo ============= DEV =============
./lattice_oracle.sh "${exp}/decode_fglarge_dev_clean/lat.*.gz" ${vocab} ${dev_text}
./lattice_oracle.sh "${exp}/decode_fglarge_dev_other/lat.*.gz" ${vocab} ${dev_text}

# test data
echo ============= TEST =============
./lattice_oracle.sh "${exp}/decode_fglarge_test_clean/lat.*.gz" ${vocab} ${test_text}
./lattice_oracle.sh "${exp}/decode_fglarge_test_other/lat.*.gz" ${vocab} ${test_text}

# train data
echo ============= TRAIN =============
./lattice_oracle.sh "${exp}/decode_fglarge_train_960/lat.[1-6].gz" ${vocab} ${train_text}
