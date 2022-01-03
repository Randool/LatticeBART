import re
import socket
from train_lattice_bart import main

config = {
    "bart_path": "/home/randool/Dataset/transformers/bart-base",
    "init_model": 0,
    "seed": 2048,
    "exp_dir": "debug/exp/lattice_bart",
}

if socket.gethostname() != "randool-manjaro":
    config["bart_path"] = "/mnt/xlancefs/home/lfd98/Dataset/transformers/bart-base"

train_cmd = """
--lat_enc_layers 6
--bart_path {bart_path}
--init_model {init_model}

--lr 1e-5
--weight_decay 1e-6
--lr_decay 0.95
--grad_acc_steps 2
--clip 1e-1

--train_lat_ark debug/data/swbd/train.ark
--eval_lat_ark  debug/data/swbd/dev.ark
--test_lat_ark  debug/data/swbd/test.ark
--train_text    debug/data/swbd/text
--eval_text     debug/data/swbd/text
--test_text     debug/data/swbd/text

--lm_scale 1.0
--ac_scale 1.0
--min_nodes 1
--max_nodes 200
--use_marginal  1
--use_min_depth 1
--use_max_depth 1
--use_forward_backward 1
--use_lattice_mask 1
--homophone_replace 0.05
--token_mask 0.05
--depth_disturb 5

--device cuda:0
--parallel DDP
--seed {seed}
--optim adam
--exp_dir {exp_dir}
--epoch 7
--train_batch 2
--eval_batch 8
--num_workers 4
--log_every 10
--patience 3
--use_metric loss

--decoder_input lattice
--always_save 1

--num_beams 4
--min_len 2
--max_len 256
"""

if __name__ == "__main__":
    for seed in (66,):
        config["seed"] = seed
        cmd = train_cmd.format(**config)
        # do_train
        args = re.split("\\s+", cmd.strip())
        main(args)
