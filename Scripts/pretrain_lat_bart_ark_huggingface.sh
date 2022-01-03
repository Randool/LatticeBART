#!/bin/bash
#SBATCH --job-name=train_LatBART_ark_HF
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
#SBATCH --output=exp/train_LatBART_ark_HF-%A-%a.out
#SBATCH --error=exp/train_LatBART_ark_HF-%A-%a.err
#SBATCH --exclude=gqxx-01-[027,175]
#SBATCH --array=0

hostname
export PYTHONPATH="${HOME}/Projects/LatticeBART"

python \
  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank="${SLURM_ARRAY_TASK_ID}" \
train_lattice_bart_huggingface.py \
  --output_dir  exp/LatBART_hf \
  --do_train    true \
  --do_eval     true \
  --do_predict  true \
  --evaluation_strategy     epoch \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size  8 \
  --gradient_accumulation_steps 2 \
  --learning_rate           1e-5 \
  --weight_decay            0 \
  --num_train_epochs        8 \
  --lr_scheduler_type       cosine \
  --warmup_steps            100 \
  --logging_steps           500 \
  --save_strategy           epoch \
  --save_total_limit        4 \
  --seed                    42 \
  --dataloader_num_workers  4 \
  --disable_tqdm            true \
  --metric_for_best_model   loss \
  --greater_is_better       false \
  --label_smoothing_factor  0 \
  --dataloader_pin_memory   false \
  \
  --bart_path     "${HOME}/Dataset/transformers/bart-base" \
  --init_model    0 \
  --train_lat_ark "${HOME}/Dataset/mr_fisher_swbd/train.ark" \
  --eval_lat_ark  "${HOME}/Dataset/mr_fisher_swbd/dev.ark" \
  --test_lat_ark  "${HOME}/Dataset/mr_fisher_swbd/eval2000.b4.ark" "${HOME}/Dataset/mr_fisher_swbd/rt03.b4.ark" \
  --train_text    "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --eval_text     "${HOME}/Dataset/mr_fisher_swbd/train.txt" \
  --test_text     "${HOME}/Dataset/mr_fisher_swbd/test.txt" \
  --lm_scale      1.0 \
  --ac_scale      1.0 \
  --min_nodes     1 \
  --max_nodes     200 \
  --use_marginal          1.0 \
  --use_min_depth         1.0 \
  --use_max_depth         1.0 \
  --use_forward_backward  1.0 \
  --use_lattice_mask      1.0 \
  --homophone_replace     0.0 \
  --token_mask            0.0 \
  --depth_disturb         0 \
  --decoder_input         sent
