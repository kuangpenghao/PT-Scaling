# Probabilistic Transformer

This is a huggingface implementation of the Probabilistic Transformer project.

## Installation

```bash
pip install -r requirements.txt
```

The current implementation does not support flash attention yet.

## Usage

See the configuration file `models/configuration_pt.py` for the hyperparameters.

The example bash script is `run_mlm.sh.template`. Please rename it to `run_mlm.sh` and modify the hyperparameters accordingly.

```bash
bash run_mlm.sh
```

## Configs

| Config                   | Parameters |
| ------------------------ | ---------- |
| `configs/pt_tiny.json`   | 34M        |
| `configs/pt_medium.json` | 40M        |
| `configs/pt_base.json`   | 53M        |
| `configs/tinypt.json`    | 89M        |

## TODO

### Hyperparameter tuning

The config (see below), and the training hyperparameters (learning rate, learning rate scheduler, batch size, optimizer, etc.) need to be tuned.

```json
{
    // no need to change
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "PtForMaskedLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "model_type": "pt",

    // model structure, will change the number of parameters, not priority
    "dim_z": 768,
    "dim_g": 3072,
    "num_iterations": 12,
    "num_channels": 12,
    "ternary_rank": 64,

    // model structure, needs to be tuned
    "potential_func_z": "square",
    "potential_func_g": "abs",

    // no need to change
    "max_position_embeddings": 512,

    // model initialization, needs to be tuned
    // you may change the codes if necessary
    "initializer_range": 0.02,
    "binary_initializer_range": 0.2,
    "ternary_initializer_range": 0.2,

    // model scaling, needs to be tuned
    "binary_factor_scaling": 1.0,
    "ternary_factor_scaling": 1.0,
    "classifier_amplifier": 768.0,

    // no need to change
    "potential_eps": 1e-6,
    "tie_word_embeddings": false,
    "rope_theta": 10000.0,
    "rope_scaling": null,

    // dropout, needs to be tuned
    "dropout_prob_z": 0.0,
    "dropout_prob_h": 0.0,
    "classifier_dropout": null,

    // regularization, needs to be tuned, important
    "regularize_z": 1.0,
    "regularize_h": 0.013,
    "regularize_g": 1.0,

    // no need to change
    "hidden_size": 768,
    "hidden_act": "silu",
    "layer_norm_eps": 1e-05,
    "output_heads": false,
    "output_qzs": false,
    "torch_dtype": "float32",
    "transformers_version": "4.31.0.dev0",
    "vocab_size": 32000
}
```
