#!/bin/bash
# 激活 pt 环境
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate pt

export WANDB_PROJECT=pt-tiny
echo "Start running..."
echo "Slurm job id: $SLURM_JOB_ID"

# # improvement: huge
# export LCKV_FLASH_ATTN=1
# # improvement: significant
# export LCKV_FUSED_RMSNORM=1
# # improvement: none
# export LCKV_FUSED_CROSSENTROPY=1
# # improvement: none
# export LCKV_FUSED_ROTARY=1
# # improvement: slightly
# export LCKV_FUSED_SWIGLU=1

## pretrain code for llama-tiny
#  - to pretrain a tinyllama, change the config to `TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T`
#  - to intialize the model with a pretrained model, add `--model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T`
#  - to use the minipile dataset, use `--dataset_name JeanKaddour/minipile`, with proper `--preprocessing_num_workers`
#  - to enable wandb, use `--report_to wandb`
python run_clm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/llama_tiny.json \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --auto_find_batch_size \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --num_train_epochs 0.6 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --run_name llama-tiny-wiki103-ep1-usesquare \
    --overwrite_output_dir \
    --output_dir /tmp/test-clm-$RANDOM-`date +"%m-%d--%H-%M-%S"`
    # --output_dir outputs/pt-tiny-wiki103-ep1-usesquare



# accelerate launch run_clm.py \
#     --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
#     --config_name configs/pt_medium.json \
#     --dataset_name JeanKaddour/minipile \
#     --preprocessing_num_workers 96 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --auto_find_batch_size \
#     --gradient_accumulation_steps 1 \
#     --block_size 1024 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.015 \
#     --learning_rate 3e-4 \
#     --weight_decay 1e-1 \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --do_train \
#     --do_eval \
#     --num_train_epochs 1 \
#     --save_total_limit 1 \
#     --save_strategy steps \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --logging_steps 50 \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --report_to none \
#     --run_name pt-medium-minipile \
#     --overwrite_output_dir \
#     --output_dir outputs/pt-medium-minipile