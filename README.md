<a name="readme-top"></a>

<h3 align="center">
StateBridge: Latent-State Communication for Multi-Agent Systems
</h3>

---

## Introduction

**StateBridge** enables multi-agent collaboration through **latent-state communication** — agents exchange information by passing aligned hidden states rather than generating full text.

The core alignment pipeline uses **Whitened Orthogonal Procrustes** to map hidden states into the embedding space:
1. **Center** both hidden-state and embedding distributions
2. **Whiten** via regularized covariance decomposition
3. **Rotate** using Orthogonal Procrustes (SVD-based)
4. **Reconstruct** in embedding space with norm calibration and vocabulary snapping

This achieves **superior performance** with **significantly fewer tokens** compared to text-based multi-agent systems.


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


## Citation

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```
