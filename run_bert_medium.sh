#!/bin/bash
export WANDB_PROJECT=pt-single-test

python run_bert.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/bert_medium.json \
    --dataset_name ./local_datasets/minipile \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.15 \
    --report_to wandb \
    --run_name bert-medium-run-4e-4-0.15 \
    --output_dir outputs/bert-medium-4e-4-0.15 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --target_total_batch_size 128 \
    --learning_rate 0.0004 \
    --num_train_epochs 1 \
    --max_seq_length 1024 \
    --bf16
