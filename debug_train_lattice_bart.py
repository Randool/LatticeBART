import re
import socket

from train_lattice_bart_huggingface import main

config = {
    "bart_path": "/home/randool/Dataset/transformers/bart-base",
    "init_model": 0,
    "seed": 42,
}

if socket.gethostname() != "randool-manjaro":
    config["bart_path"] = "/mnt/xlancefs/home/lfd98/Dataset/transformers/bart-base"

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1

train_cmd = """
--bart_path {bart_path}
--init_model {init_model}

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
--homophone_replace 0.00
--token_mask 0.00
--depth_disturb 0
--decoder_input lattice

--output_dir ./debug/exp/LatticeBART
--do_train   True
--do_eval    True
--do_predict True
--evaluation_strategy epoch

--per_device_train_batch_size 4
--per_device_eval_batch_size  8
--gradient_accumulation_steps 2

--learning_rate 1e-5
--weight_decay  1e-6
--label_smoothing_factor 0.0
--lr_scheduler_type cosine

--num_train_epochs 8
--warmup_ratio     0.01

--disable_tqdm  True
--logging_steps 100
--save_strategy epoch
--save_steps    500

--dataloader_drop_last   True
--dataloader_num_workers 4
--dataloader_pin_memory  False

--metric_for_best_model  loss
--greater_is_better      False
--load_best_model_at_end True

--predict_with_generate True
--generation_max_length 128
--generation_num_beams  4
"""

if __name__ == "__main__":
    cmd = train_cmd.format(**config)
    # do_train
    args = re.split("\\s+", cmd.strip())
    main(args)
