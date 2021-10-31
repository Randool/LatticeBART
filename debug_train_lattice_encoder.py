import re

from train_lattice_encoder import main

if __name__ == "__main__":
    cmd = """
      --lat_enc_layers 6
      --bart_path /home/randool/Dataset/transformers/bart-base
      --init_model 0
      --lr 1e-5
      --weight_decay 1e-6
      --lr_decay 0.95
      --grad_acc_steps 2
      --clip 1e-1
      --cache_dir exp/encoder_data_cache
      --train_lat_ark debug/data/librispeech/train.ark
      --eval_lat_ark  debug/data/librispeech/train.ark
      --test_lat_ark  debug/data/librispeech/test1.ark debug/data/librispeech/test2.ark debug/data/librispeech/test3.ark
      --lm_scale 1.0
      --ac_scale 1.0
      --min_nodes 1
      --max_nodes 200
      --use_marginal  0.9
      --use_min_depth 0.9
      --use_max_depth 0.9
      --use_forward_backward 0.8
      --use_lattice_mask 1
      --homophone_replace 0.2
      --seed 1024
      --device cuda:0
      --optim lamb
      --exp_dir debug/exp/lattice_encoder_ark_another
      --epoch 8
      --train_batch 16
      --eval_batch 32
      --num_workers 3
      --log_every 1
      --patience 3
      --use_metric loss
      --contrast_method mask
      --pooling_method weight
      --token_id_weight debug/token_freq_weight.json
    """
    args = re.split("\\s+", cmd.strip())
    main(args)
