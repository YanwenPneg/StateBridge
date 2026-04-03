<a name="readme-top"></a>

<h3 align="center">
StateBridge: Latent-State Communication for Multi-Agent Systems
</h3>

---

## Repository Structure

```
StateBridge/
│── methods/
│   ├── __init__.py          # Agent dataclass & default 4-agent pipeline
│   └── state_bridge.py      # Core StateBridge method + multi-GPU evaluator
│── models.py                # HuggingFace Transformers model wrapper
│── prompts.py               # Embedding-MAS prompt constructors
│── data.py                  # Dataset loaders (GSM8K, AIME, GPQA, ARC, MBPP+, etc.)
│── utils.py                 # Answer parsing, seed, timeout helpers
│── data/                    # Local dataset files (GPQA, MedQA)
│── requirements.txt
```


## Setup

```bash
conda create -n statebridge python=3.10 -y
conda activate statebridge
pip install -r requirements.txt
```

Set HuggingFace cache:
```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```


## Quick Start

### Single Task

```bash
python -m methods.state_bridge --model Qwen/Qwen3-4B --task gsm8k --gpus 0
```

### Multi-GPU

```bash
python -m methods.state_bridge --model Qwen/Qwen3-8B --task gpqa --gpus 0,1,2,3
```

### All Tasks

```bash
python -m methods.state_bridge --model Qwen/Qwen3-8B --run_all --gpus 0,1,2,3
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace model name |
| `--task` | `medqa` | Dataset to evaluate |
| `--gpus` | all available | Comma-separated GPU IDs |
| `--max_prefix_tokens` | `64` | Max tokens in prefix embedding |
| `--adaptive_reg` | `1e-3` | Covariance regularization λ |
| `--snap_ratio` | `0.3` | Vocabulary snapping ratio |
| `--temperature` | `0.6` | Sampling temperature |
| `--seed` | `None` | Random seed |
| `--collect_viz` | `False` | Collect alignment vectors for visualization |


