#!/bin/bash
export WANDB_PROJECT=pt-single-test

# Resume from: outputs/pt-sweep-run/yyztqil6
# (请根据实际检查点编号调整)

python run_mlm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/pt_256.json \
    --dataset_name ./local_datasets/minipile \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 800 \
    --evaluation_strategy steps \
    --eval_steps 800 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --run_name pt-256-ece \
    --output_dir outputs/pt-single-test/pt-256-ece \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --target_total_batch_size 128 \
    --num_train_epochs 1 \
    --learning_rate 0.18 \
    --weight_decay 0.002 \
    --decay_factor 0.5 \