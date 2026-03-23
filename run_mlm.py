#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from fractions import Fraction
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
import numpy
try:
    # 修复 PyTorch 2.6+ 加载 checkpoint 时的 weights_only 报错
    safe_globals = [
        numpy.ndarray,
        numpy._core.multiarray._reconstruct,
        numpy.dtype
    ]
    # 处理 numpy.dtypes.UInt32DType
    if hasattr(numpy, "dtypes") and hasattr(numpy.dtypes, "UInt32DType"):
        safe_globals.append(numpy.dtypes.UInt32DType)

    torch.serialization.add_safe_globals(safe_globals)
except (AttributeError, TypeError, ImportError):
    pass

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import models


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.35.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def compute_pt_parameter_count(dim_z: int, dim_g: int, num_channels: int, vocab_size: int, ternary_rank: int,
                               tie_word_embeddings: bool = False) -> int:
    """Approximate trainable parameter count for the PtForMaskedLM architecture."""
    embedding_params = vocab_size * dim_z
    ternary_params = 2 * num_channels * ternary_rank * dim_z
    binary_params = dim_g * dim_z
    mlm_dense_params = dim_z * dim_z + dim_z  # weight + bias
    layernorm_params = 2 * dim_z
    decoder_weight_params = 0 if tie_word_embeddings else dim_z * vocab_size
    decoder_bias_params = vocab_size
    return (
        embedding_params
        + ternary_params
        + binary_params
        + mlm_dense_params
        + layernorm_params
        + decoder_weight_params
        + decoder_bias_params
    )


def solve_dims_and_channels(
    ratio: float,
    vocab_size: int,
    ternary_rank: int,
    tie_word_embeddings: bool,
    target_total: int,
    num_channels_min: int = 12,
    num_channels_max: int = 128,
):
    """Search even num_channels and even (dim_z, dim_g) to match target params under a ratio.

    Returns (num_channels, dim_z, dim_g, best_total, best_diff).
    """
    best = None
    for ch in range(max(2, num_channels_min + num_channels_min % 2), num_channels_max + 1, 2):
        # Solve quadratic: total ≈ (1+ratio)*dim_z^2 + (2*vocab + 2*ch*ternary_rank + 3)*dim_z + vocab
        a = 1.0 + ratio
        b = 2 * vocab_size + 2 * ch * ternary_rank + 3
        c = vocab_size - target_total
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            continue
        approx_z = (-b + math.sqrt(discriminant)) / (2 * a)
        if approx_z <= 0:
            continue
        
        # Search around approx_z for even dim_z
        base_even = int(round(approx_z / 2)) * 2
        for radius in [50, 100, 200]:
            start = max(2, base_even - radius)
            end = base_even + radius
            for dim_z in range(start, end + 1, 2):
                dim_g_float = ratio * dim_z
                dim_g = int(round(dim_g_float / 2)) * 2
                if dim_g <= 0:
                    continue
                total = compute_pt_parameter_count(dim_z, dim_g, ch, vocab_size, ternary_rank, tie_word_embeddings)
                diff = abs(total - target_total)
                if best is None or diff < best[4]:
                    best = (ch, dim_z, dim_g, total, diff)
                    if diff <= target_total * 0.005:
                        break
            if best and best[4] <= target_total * 0.005:
                break
        if best and best[4] <= target_total * 0.005:
            break
    if best is None:
        raise ValueError(
            f"Unable to find even (num_channels, dim_z, dim_g) for ratio={ratio} within channels [{num_channels_min}, {num_channels_max}]."
        )
    return best


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    # UT Sweep Parameters (added to allow command line passing)
    act_epsilon: Optional[float] = field(default=None, metadata={"help": "ACT epsilon"})
    ponder_weight: Optional[float] = field(default=None, metadata={"help": "Ponder cost weight"})
    act_bias_init: Optional[float] = field(default=None, metadata={"help": "ACT bias initialization"})
    hidden_dropout_prob: Optional[float] = field(default=None, metadata={"help": "Hidden dropout prob"})
    attention_probs_dropout_prob: Optional[float] = field(default=None, metadata={"help": "Attention dropout prob"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    target_total_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Target total batch size. If set, will auto-calculate gradient_accumulation_steps based on n_gpu and per_device_batch_size."}
    )
    decay_factor: float = field(
        default=0.0,
        metadata={"help": "Exponent factor for scaling weight decay based on model size (dim_z)."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")



class ParameterMonitorCallback(TrainerCallback):
    """
    Log parameter statistics (mean, std, min, max) to a file to monitor training dynamics.
    Appends to 'parameter_monitor.txt' in output_dir.
    """
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "parameter_monitor.txt")
        self.last_eval_loss = None
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Cache eval loss
        if metrics and "eval_loss" in metrics:
            self.last_eval_loss = metrics["eval_loss"]

    def on_save(self, args, state, control, model=None, **kwargs):
        self._log_stats(state.global_step, model, args, phase="SAVE")

    def _write_log(self, content):
        # Only log on main process
        if is_torch_xla_available():
             pass
        elif torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        try:
            with open(self.output_file, "a") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to log to parameter_monitor: {e}")

    def _log_stats(self, step, model, training_args, phase=""):
        if model is None:
            return
        
        # Check process rank via _write_log logic
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        # Prepare stat string
        lines = [f"=== {phase} Step {step} ===\n"]

        # 1. Eval Loss
        if self.last_eval_loss is not None:
             lines.append(f"Last Eval Loss: {self.last_eval_loss}\n")
        
        # 2. Hyperparameters being tuned
        if hasattr(model, "config"):
            c = model.config
            lines.append("Hyperparameters:\n")
            lines.append(f"  dim_z: {getattr(c, 'dim_z', 'N/A')}\n")
            lines.append(f"  binary_factor_scaling: {getattr(c, 'binary_factor_scaling', 'N/A')}\n")
            lines.append(f"  ternary_factor_scaling: {getattr(c, 'ternary_factor_scaling', 'N/A')}\n")
            lines.append(f"  classifier_amplifier: {getattr(c, 'classifier_amplifier', 'N/A')}\n")
            lines.append(f"  regularize_z: {getattr(c, 'regularize_z', 'N/A')}\n")
            lines.append(f"  regularize_g: {getattr(c, 'regularize_g', 'N/A')}\n")
            lines.append(f"  regularize_h: {getattr(c, 'regularize_h', 'N/A')}\n")
            lines.append(f"  learning_rate: {training_args.learning_rate}\n")
            lines.append("\n")

        # 3. Parameter Stats
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Use float32 for stats to avoid overflow
            data = param.data.float()
            mean = data.mean().item()
            std = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()
            
            lines.append(f"{name}:\n")
            lines.append(f"  Mean: {mean:.4e} | Std: {std:.4e} | Min: {min_val:.4e} | Max: {max_val:.4e}\n")
            
            if torch.isnan(data).any():
                lines.append(f"  [WARNING] NaN detected!\n")
            if torch.isinf(data).any():
                lines.append(f"  [WARNING] Inf detected!\n")
        lines.append("\n")
        
        self._write_log("".join(lines))


class ActivationMonitorCallback(TrainerCallback):
    """
    Log max/min/mean/var of activations and logits every 30 steps.
    Also logs the numerical change of single elements compared to 30 steps ago.
    """
    def __init__(self, output_dir, model):
        self.output_file = os.path.join(output_dir, "activation_monitor.txt")
        self.model = model
        self.step_interval = 30
        self.history = {}
        self.current_activations = {}
        self.should_log_step = False
        
        # Ensure output dir exists
        if hasattr(os, "makedirs"):
            os.makedirs(output_dir, exist_ok=True)
            
        self._register_hooks()

    def _register_hooks(self):
        # Handle wrapping
        m = self.model
        if hasattr(m, "module"): m = m.module
        
        # 1. Hidden Layers (PtEncoderIterator)
        iterator = None
        if hasattr(m, "model") and hasattr(m.model, "iterator"):
            iterator = m.model.iterator
        elif hasattr(m, "iterator"):
            iterator = m.iterator
            
        if iterator is not None:
             iterator.register_forward_hook(self.hook_iterator)
        
        # 2. Logits (PtForMaskedLM -> cls)
        # PtForMaskedLM has .cls which is PtLMPredictionHead
        if hasattr(m, "cls"):
            m.cls.register_forward_hook(self.hook_logits)
            
    def hook_iterator(self, module, input, output):
        if not self.should_log_step: return
        # Filter for main device if possible to avoid clutter generally, but simple approach:
        # PtEncoderIterator returns (qz, ...)
        qz = output[0] if isinstance(output, tuple) else output
        if "hidden" not in self.current_activations:
            self.current_activations["hidden"] = []
        
        # We only want to log one batch (usually the first one that runs on this gpu)
        # But this hook runs multiple times (for iterations)
        # We need to distinguish iterations.
        # But hook doesn't know iteration index easily.
        # However, forward is called sequentially.
        self.current_activations["hidden"].append(qz.detach().cpu())

    def hook_logits(self, module, input, output):
        if not self.should_log_step: return
        self.current_activations["logits"] = output.detach().cpu()

    def on_step_begin(self, args, state, control, **kwargs):
        self.current_activations = {}
        # Log at steps 1, 31, 61... 
        # state.global_step is 0 at start of step 1.
        if state.global_step % self.step_interval == 0:
            self.should_log_step = True
        else:
            self.should_log_step = False

    def on_step_end(self, args, state, control, **kwargs):
        # After step 1 finished, state.global_step is 1.
        # (1-1) % 30 == 0 -> True.
        if (state.global_step - 1) % self.step_interval == 0:
            self._log_and_rotate(state.global_step)
            
    def _log_and_rotate(self, step_num):
        # Only log on main process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
            
        lines = [f"=== Activation Stats Step {step_num} ===\n"]
        
        def compute_stats(name, current_tensor, prev_tensor=None):
            # Input: tensor of shape (batch, seq, dim)
            f = current_tensor.float()
            mean = f.mean().item()
            var = f.var().item()
            min_val = f.min().item()
            max_val = f.max().item()
            
            stat_str = f"{name}: Mean: {mean:.4e} | Var: {var:.4e} | Min: {min_val:.4e} | Max: {max_val:.4e}\n"
            
            # Delta
            if prev_tensor is not None:
                # Check shapes
                if prev_tensor.shape == current_tensor.shape:
                    diff = (current_tensor.float() - prev_tensor.float())
                    d_mean = diff.mean().item()
                    d_var = diff.var().item()
                    d_min = diff.min().item()
                    d_max = diff.max().item()
                    stat_str += f"  Delta(30): Mean: {d_mean:.4e} | Var: {d_var:.4e} | Min: {d_min:.4e} | Max: {d_max:.4e}\n"
                else:
                    stat_str += f"  Delta(30): Shape mismatch ({prev_tensor.shape} vs {current_tensor.shape})\n"
            
            return stat_str, current_tensor

        # Process Logits
        logits = self.current_activations.get("logits", None)
        if logits is not None:
            prev_logits = self.history.get("logits", None)
            s, stored = compute_stats("Logits", logits, prev_logits)
            lines.append(s)
            self.history["logits"] = stored
        else:
            lines.append("Logits: Not captured\n")

        # Process Hidden
        hidden_list = self.current_activations.get("hidden", [])
        if hidden_list:
            if "hidden" not in self.history:
                self.history["hidden"] = []
            
            new_history_hidden = []
            for i, h in enumerate(hidden_list):
                prev_h = self.history["hidden"][i] if i < len(self.history["hidden"]) else None
                s, stored = compute_stats(f"Hidden Iter {i}", h, prev_h)
                lines.append(s)
                new_history_hidden.append(stored)
            self.history["hidden"] = new_history_hidden
        else:
             lines.append("Hidden: Not captured\n")

        lines.append("\n")
        
        try:
            with open(self.output_file, "a") as f:
                f.write("".join(lines))
        except Exception as e:
            logger.warning(f"Failed to write activations: {e}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Auto-calculate gradient_accumulation_steps if target_total_batch_size is set
    if data_args.target_total_batch_size is not None:
        # Determine world size (n_gpus)
        from transformers.training_args import ParallelMode
        if training_args.parallel_mode == ParallelMode.DISTRIBUTED:
             n_gpus = training_args.world_size
        else:
             n_gpus = training_args.n_gpu
        
        # Avoid division by zero
        n_gpus = max(1, n_gpus)
        
        per_device = training_args.per_device_train_batch_size
        target = data_args.target_total_batch_size
        
        # Calculate accumulation steps
        new_accum_steps = max(1, target // (n_gpus * per_device))
        
        # Update training_args
        # Create a logger locally since the global one might not be fully configured yet, 
        # or use print to ensure visibility.
        print(f"Auto-Scaling Info: Target BS={target} | GPUs={n_gpus} | Per-Device BS={per_device}")
        print(f"  -> Setting gradient_accumulation_steps to {new_accum_steps}")
        training_args.gradient_accumulation_steps = new_accum_steps

    # Initialize wandb early if running under wandb agent
    import wandb
    if wandb.run is None and "WANDB_RUN_ID" in os.environ:
        wandb.init(reinit=True)
    
    # Update output_dir to include wandb run_id for parallel runs
    if wandb.run is not None:
        base_output_dir = training_args.output_dir
        run_id = wandb.run.id
        training_args.output_dir = os.path.join(base_output_dir, run_id)
        logger.info(f"Updated output_dir to: {training_args.output_dir}")

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Check if it is a local dataset saved via save_to_disk
        if os.path.isdir(data_args.dataset_name) and os.path.exists(os.path.join(data_args.dataset_name, "dataset_dict.json")):
            from datasets import load_from_disk
            logger.info(f"Loading dataset from local disk: {data_args.dataset_name}")
            raw_datasets = load_from_disk(data_args.dataset_name)
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
        
        if "validation" not in raw_datasets.keys():
            # If load_from_disk was used, raw_datasets has keys from dataset_dict.json (usually train, test, validation)
            # If load_dataset was used, it might only have 'train' if no split logic was applied yet for some datasets without config
            if data_args.streaming:
                # Streaming split logic is different or not fully supported in this snippet, 
                # but user turned off streaming for local disk
                pass
            else:
                 # Standard logic from original script
                 raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                 raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        for config_str in model_args.config_overrides.replace(" ", "").split(","):
            key, value = config_str.split("=")
            if key == "hidden_size":
                config.hidden_size = int(value)
            elif key == "num_hidden_layers":
                config.num_hidden_layers = int(value)
            elif key == "num_attention_heads":
                config.num_attention_heads = int(value)
            else:
                try:
                    # try int
                    val = int(value)
                except ValueError:
                    try:
                        # try float
                        val = float(value)
                    except ValueError:
                        val = value
                setattr(config, key, val)

    # -------------------------------------------------------------------------
    # Update config from ModelArguments (for UT sweeps w/ direct args)
    # -------------------------------------------------------------------------
    if model_args.act_epsilon is not None:
        config.act_epsilon = model_args.act_epsilon
        logger.info(f"Updated config.act_epsilon from args: {config.act_epsilon}")
    if model_args.ponder_weight is not None:
        config.ponder_weight = model_args.ponder_weight
        logger.info(f"Updated config.ponder_weight from args: {config.ponder_weight}")
    if model_args.act_bias_init is not None:
        config.act_bias_init = model_args.act_bias_init
        logger.info(f"Updated config.act_bias_init from args: {config.act_bias_init}")
    if model_args.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        logger.info(f"Updated config.hidden_dropout_prob from args: {config.hidden_dropout_prob}")
    if model_args.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
        logger.info(f"Updated config.attention_probs_dropout_prob from args: {config.attention_probs_dropout_prob}")
    
    # Apply wandb sweep parameters directly to config object
    if wandb.run is not None:
        logger.info("Applying wandb sweep parameters to config...")

        int_config_keys = {"num_channels", "num_iterations", "ternary_rank"}

        def _normalize_config_value(key, value):
            # Booleans encoded as strings
            if isinstance(value, str) and value.lower() in ["true", "false"]:
                return value.lower() == "true"
            # Try integer coercion for known int keys
            if key in int_config_keys:
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        return int(float_val)
                except (TypeError, ValueError):
                    pass
            # Generic float coercion to avoid strings like "1e-05"
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            return value

        config_keys = [
            "num_channels",
            "num_iterations",
            "ternary_rank",
            "potential_func_z",
            "potential_func_g",
            "max_position_embeddings",
            "initializer_range",
            "binary_initializer_range",
            "ternary_initializer_range",
            "binary_factor_scaling",
            "ternary_factor_scaling",
            "classifier_amplifier",
            "potential_eps",
            "tie_word_embeddings",
            "rope_theta",
            "dropout_prob_z",
            "dropout_prob_h",
            "regularize_z",
            "regularize_h",
            "regularize_g",
            "hidden_act",
            "layer_norm_eps",
            "output_heads",
            "output_qzs",
        ]

        for key in config_keys:
            if key in wandb.config:
                value = _normalize_config_value(key, wandb.config[key])
                old_value = getattr(config, key, None)
                setattr(config, key, value)
                logger.info(f"  {key}: {old_value} -> {value}")

        # muP adjustments
        if "dim_z" in wandb.config:
            val = int(wandb.config["dim_z"])
            config.dim_z = val
            config.hidden_size = val # ensure consistency
            logger.info(f"  dim_z set to {val}")

        if "dim_ratio" in wandb.config:
            ratio = float(wandb.config["dim_ratio"])
            config.dim_g = int(config.dim_z * ratio)
            logger.info(f"  dim_g set to {config.dim_g} (ratio {ratio})")
        elif "dim_g" in wandb.config:
            config.dim_g = int(wandb.config["dim_g"])

        if "ternary_rank_ratio" in wandb.config:
            ratio = float(wandb.config["ternary_rank_ratio"])
            # ternary_rank = dim_z * ratio
            # Must be even for RoPE
            raw_rank = int(config.dim_z * ratio)
            config.ternary_rank = max(2, raw_rank + (raw_rank % 2))
            logger.info(f"  ternary_rank set to {config.ternary_rank} (dim_z * {ratio}, rounded to even)")

        # Update other derived config if needed
        # ...

        training_int_keys = {"per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps"}
        for key in [
            "learning_rate",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "num_train_epochs",
        ]:
            if key in wandb.config:
                raw_value = wandb.config[key]
                value = raw_value
                if isinstance(raw_value, str):
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
                if key in training_int_keys and isinstance(value, float) and value.is_integer():
                    value = int(value)
                old_value = getattr(training_args, key, None)
                setattr(training_args, key, value)
                logger.info(f"  training_args.{key}: {old_value} -> {value}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Avoid increasing vocab size when possible: reuse existing tokens first.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Reusing eos_token as pad_token to avoid resizing embeddings.")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            logger.info("Added pad_token '<pad>' (vocab will grow).")

    if tokenizer.mask_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.mask_token = tokenizer.unk_token
            logger.info("Reusing unk_token as mask_token to avoid resizing embeddings. MLM masking will use unk id.")
        else:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})
            logger.info("Added mask_token '<mask>' (vocab will grow).")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code, torch_dtype=torch_dtype)

    if wandb.run is not None:
        param_count = model.num_parameters() if hasattr(model, "num_parameters") else sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameter count: {param_count:,}")
        if getattr(training_args, "process_index", 0) == 0:
            wandb.config.update({"model_parameter_count_actual": int(param_count)}, allow_val_change=True)
            wandb.run.summary["model_parameter_count"] = int(param_count)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    callbacks = []
    
    if training_args.local_rank in [-1, 0]:
        callbacks.append(ParameterMonitorCallback(training_args.output_dir))
        # callbacks.append(ActivationMonitorCallback(training_args.output_dir, model))

    # Custom Optimizer for muP
    from torch.optim import AdamW
    
    optimizer_grouped_parameters = []
    base_lr = training_args.learning_rate
    # If training_args.learning_rate is default (5e-5), hopefully user set it in sweep or args.
    
    dim_z = config.dim_z

    # Scale max_grad_norm by dim_z / 256
    if training_args.max_grad_norm is not None:
        old_norm = training_args.max_grad_norm
        training_args.max_grad_norm = old_norm * (dim_z / 256.0)
        logger.info(f"muP: Scaled max_grad_norm from {old_norm} to {training_args.max_grad_norm} (dim_z={dim_z})")
    
    # Scale weight_decay
    weight_decay = training_args.weight_decay
    if data_args.decay_factor != 0.0:
        weight_decay = weight_decay * ((dim_z / 256.0) ** data_args.decay_factor)
        logger.info(f"muP: Scaled weight_decay from {training_args.weight_decay} to {weight_decay} (dim_z={dim_z}, decay_factor={data_args.decay_factor})")
        # Explicitly update training_args so it reflects in logs and checks
        training_args.weight_decay = weight_decay

    group1_params = [] # Base LR: Input/Bias/LN
    group2_params = [] # Scaled LR: Hidden/Output (Matrices)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Heuristic for Input/Bias/Scalar
        is_group1 = False
        if param.ndim < 2:
            is_group1 = True
        elif "unary_factors" in name or "embeddings" in name:
            is_group1 = True
        elif "bias" in name: 
            is_group1 = True
            
        if is_group1:
            group1_params.append(param)
        else:
            group2_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": group1_params, "weight_decay": weight_decay, "lr": base_lr},
            {"params": group2_params, "weight_decay": weight_decay, "lr": base_lr / dim_z},
        ],
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # Define custom Trainer to fix DataParallel loss summing issue
    class PtTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            # Manual gradient accumulation handling to ensure loss scaling/logging is correct
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps

            if self.deepspeed:
                self.deepspeed.backward(loss)
            else:
                self.accelerator.backward(loss)

            return loss.detach()

    trainer = PtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        callbacks=callbacks,
        optimizers=(optimizer, None), # Pass custom optimizer
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()