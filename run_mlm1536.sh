#!/bin/bash
export WANDB_PROJECT=pt-single-test

# Resume from: outputs/pt-sweep-run/yyztqil6
# (请根据实际检查点编号调整)

python run_mlm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/pt_1536_2.json \
    --dataset_name ./local_datasets/minipile \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --logging_steps 15 \
    --save_strategy steps \
    --save_steps 1500 \
    --evaluation_strategy steps \
    --eval_steps 1500 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --run_name pt-resume-1536-260127-64 \
    --output_dir outputs/pt-single-test/1536-260127-64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --target_total_batch_size 64 \
    --num_train_epochs 1.5 \
    --learning_rate 0.076203 \