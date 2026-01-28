#!/bin/bash
export WANDB_PROJECT=pt-single-test

python run_clm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/llama_124M.json \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --auto_find_batch_size \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --run_name llama-124M \
    --overwrite_output_dir \
    --output_dir outputs/pt-single-test/llama-124M
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