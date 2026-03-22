#!/bin/bash
export WANDB_PROJECT=pt-single-test

python run_bert.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/bert_mini.json \
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
    --warmup_ratio 0.2 \
    --report_to wandb \
    --run_name bert-mini-run-5e-4-0.2 \
    --output_dir outputs/bert-mini-5e-4-0.2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --target_total_batch_size 128 \
    --learning_rate 5e-4 \
    --num_train_epochs 1 \
    --max_seq_length 1024 \
    --bf16
