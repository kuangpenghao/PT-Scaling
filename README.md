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

---

| 步长 (α) | bi_scal（二元强度） | ter_scal（三元强度） |
|----------|---------------------|----------------------|
| 0.5      | 0.24098             | 1.29852              |
| 0.6      | 0.24541             | 1.33638              |
| 0.8      | 0.25454             | 1.41544              |
| 1.2      | 0.27373             | 1.58815              |
| 1.4      | 0.28391             | 1.68282              |
| 1.6      | 0.29448             | 1.78314              |
| 1.8      | 0.30544             | 1.88944              |
| 2.0      | 0.31671             | 1.99683              |