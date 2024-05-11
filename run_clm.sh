
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
accelerate launch run_clm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/pt_medium.json \
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
    --num_train_epochs 3 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --run_name pt-medium-test \
    --overwrite_output_dir \
    --output_dir outputs/pt-medium-test

