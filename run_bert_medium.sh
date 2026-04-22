#!/bin/bash
export WANDB_PROJECT=pt-single-test

python run_bert.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/bert_medium.json \
    --dataset_name ./local_datasets/minipile \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --logging_steps 25 \
    --save_strategy steps \
    --save_steps 800 \
    --eval_strategy steps \
    --eval_steps 800 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.15 \
    --report_to wandb \
    --run_name bert-medium-run-untie-0.000275 \
    --output_dir outputs/bert-medium-untie-0.000275 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --target_total_batch_size 128 \
    --learning_rate 0.000275 \
    --num_train_epochs 1 \
    --max_seq_length 1024 \
    --bf16
