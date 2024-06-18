# python run_mlm.py \
#     --model_name_or_path roberta-base \
#     --dataset_name wikitext \
#     --dataset_config_name wikitext-2-raw-v1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --report_to none \
#     --output_dir outputs/test-mlm

# ***** eval metrics *****
#   epoch                   =        3.0
#   eval_accuracy           =     0.7275
#   eval_loss               =     1.2685
#   eval_runtime            = 0:00:05.67
#   eval_samples            =        496
#   eval_samples_per_second =     87.459
#   eval_steps_per_second   =     10.932
#   perplexity              =     3.5556

python run_mlm.py \
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --config_name configs/pt_tiny.json \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --overwrite_output_dir \
    --output_dir outputs/test-mlm