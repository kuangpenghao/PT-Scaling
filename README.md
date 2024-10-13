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

| Config                    | Parameters |
| ------------------------- | ---------- |
| `configs/pt_tiny.json`    | 34M        |
| `configs/pt_medium.json`  | 40M        |
| `configs/pt_base.json`    | 53M        |
| `configs/tinypt.json`     | 89M        |
| `configs/llama_base.json` | 163M       |

## TODO

### Hyperparameter tuning

bert-base: the performance is incredibly bad. May need further tuning.
