<div align="center">

# StateBridge

### Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems

**Yanwen Peng · Delvin Ce Zhang · Xi Wang · Nikolaos Aletras**

School of Computer Science, University of Sheffield

[![COLM 2026](https://img.shields.io/badge/COLM-2026-8B5CF6?style=flat-square)](https://colmweb.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-3B82F6?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

<!-- Uncomment once the arXiv preprint is live:
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B?style=flat-square)](https://arxiv.org/abs/XXXX.XXXXX)
-->

**Agents talk in tokens. StateBridge lets them talk in hidden states, with no training at all.**

</div>

---

## Overview

Multi-agent LLM systems communicate in text. Turning a sender's continuous hidden state into discrete tokens throws away everything token identities cannot express.

StateBridge keeps the message continuous. It takes the sender's final-layer hidden states, aligns them to the receiver's input embedding space with a closed-form orthogonal transformation, and prepends the result as a continuous prefix through `inputs_embeds`. No training, no learned projector, no changes to the transformer.

![Text communication compared with StateBridge latent communication](assets/statebridge-overview.png)

Passing hidden states directly does not work: they have the right dimensionality but sit in a different region of representation space than the input embeddings the receiver was pretrained to read. Prior work resolves this either by injecting states across every transformer layer, or by training a projector. StateBridge resolves it with alignment alone.

| | Training | Model changes | Injection point | Message memory |
|---|:---:|:---:|:---:|:---:|
| TextMAS | None | None | Discrete tokens | — |
| KV-cache transfer (LatentMAS) | None | Injection at every layer | All $L$ layers | $O(TLd)$ |
| Learned projector | Required | None | Input embedding | $O(Kd)$ |
| **StateBridge** | **None** | **None** | Input embedding | $O(Kd)$ |

Operating only at the input embedding layer buys portability. On OLMo3-7B-Think, KV-cache transfer averages 55.1% while plain text averages 73.9%: injecting across layers breaks when layer structure differs between model families. StateBridge reaches 76.7% on the same setting.

## Results

Across 26 model-task evaluations from eight benchmarks, StateBridge is best or tied-best on **22 of 26 pairs** and achieves the highest average in every model setting.

| Model setting | StateBridge | Best baseline | Δ | Best / tied |
|---|---:|---:|---:|---:|
| Qwen3-4B | **82.4** | 80.0 (LatentMAS) | **+2.4** | **5 / 5** |
| OLMo3-7B-Think | **76.7** | 73.9 (TextMAS) | **+2.8** | **4 / 5** |
| Qwen3-8B | **74.3** | 71.8 (LatentMAS) | **+2.5** | **7 / 8** |
| Qwen3-32B | **81.0** | 78.1 (TextMAS) | **+2.9** | **6 / 8** |

Accuracy for QA and mathematical reasoning, pass@1 for code generation. Gains concentrate on the harder benchmarks: GPQA +7.0 on Qwen3-8B, AIME24 +6.6, MedQA +4.0.

**Ablations (Qwen3-4B average).** Full StateBridge 82.4. Replacing Procrustes with ridge regression drops to 74.9 despite lower pointwise reconstruction error, because it distorts the pairwise geometry that encodes semantic similarity. Removing norm calibration gives 79.5, removing vocabulary anchoring 80.2. A random noise prefix collapses to 48.8, which rules out the possibility that gains come from merely prepending extra continuous vectors.

**The prefix carries more than the tokens it came from.** We asked the Critic to restate the Planner's plan on a MedQA case using only the aligned prefix. At $K{=}16$ the visible suffix is a fragment, yet the Critic recovered the diagnosis, the key clinical features, and the exclusion of alternatives, including the term *koilonychia*, which appears nowhere in the visible tokens.

![StateBridge alignment visualization](assets/statebridge-alignment-visualization.png)

PCA of message hidden states (orange), reference embeddings (blue), and aligned states (green) on 300 MedQA queries. Alignment moves the prefix into the input embedding space, and the effect holds across model scales.

## Quick Start

```bash
conda create -n statebridge python=3.10 -y
conda activate statebridge

git clone https://github.com/YanwenPneg/StateBridge.git
cd StateBridge
pip install -r requirements.txt
```

Run one task, or the full suite across GPUs:

```bash
python -m methods.state_bridge --model Qwen/Qwen3-4B --task gsm8k --gpus 0
python -m methods.state_bridge --model Qwen/Qwen3-8B --run_all --gpus 0,1,2,3
```

Use it in your own pipeline:

```python
import torch
from models import ModelWrapper
from methods.state_bridge import StateBridge

model = ModelWrapper("Qwen/Qwen3-4B", torch.device("cuda:0"))
bridge = StateBridge(model, max_prefix_tokens=64, snap_ratio=0.3)

result = bridge.run_item({"question": "..."})
print(result["prediction"], result["efficiency"]["alignment_time"])
```

`run_item` runs the full Planner → Critic → Refiner → Judger pipeline, aligning and transferring hidden states between each consecutive pair.

### Tasks and options

| Category | Task identifiers |
|---|---|
| Mathematical reasoning | `gsm8k`, `aime2024`, `aime2025` |
| Question answering | `gpqa`, `arc_challenge`, `medqa` |
| Code generation | `mbppplus`, `humanevalplus` |

Key flags: `--max_prefix_tokens` (prefix length $K$, default 64), `--snap_ratio` (vocabulary anchoring $\alpha$, default 0.3), `--gpus`, `--limit`, `--seed`. Run `--help` for the full interface.

A single configuration is tuned on MedQA with Qwen3-4B and applied unchanged to every other dataset and model. All results were produced on 2 NVIDIA A100-80G GPUs at temperature 0.6 and top-$p$ 0.95.

GPQA-Diamond and MedQA ship with the repository; the rest download from Hugging Face on first use. See [data/README.md](data/README.md) for provenance and licensing.

## How It Works

Four agents run in sequence: **Planner → Critic → Refiner → Judger**. Between every consecutive pair, StateBridge:

1. **Extracts message states.** A forward hook on the final transformer layer records the sender's hidden states during generation, along with the reference embeddings of the decoded tokens.
2. **Normalizes both spaces.** Centers and whitens the hidden states and the reference embeddings.
3. **Aligns their geometry.** Solves an orthogonal Procrustes problem in closed form via SVD. Because the solution is orthogonal, it preserves distances and angles among sender states.
4. **Calibrates compatibility.** Restores input-space statistics, matches vocabulary-embedding norms, and softly anchors each state toward a nearby vocabulary embedding.
5. **Injects a continuous prefix.** Passes the aligned states to the receiver through `inputs_embeds`.

![StateBridge alignment workflow](assets/statebridge-alignment-workflow.svg)

The receiver assigns sequential position indices over the concatenated input, so its attention and position encoding treat the prefix exactly like ordinary token embeddings. Alignment costs $O(d^3)$ for whitening and $O(KVd)$ for anchoring, both once per batch, which is cheaper than a single generation pass. The forward hook keeps extraction at $O(Kd)$ rather than the $O(TLd)$ that KV-cache transfer stores per agent.

## Repository Layout

```text
StateBridge/
├── methods/
│   ├── __init__.py           # Agent definitions
│   └── state_bridge.py       # Core alignment and multi-GPU evaluator
├── data/
│   ├── README.md             # Dataset provenance and licensing
│   ├── gpqa_diamond.json     # 198 questions, CC BY 4.0
│   └── medqa.json            # 300 questions, MIT
├── assets/                   # README figures
├── data.py                   # Dataset adapters
├── models.py                 # Hugging Face model wrapper
├── prompts.py                # Four-agent prompt construction
├── utils.py                  # Evaluation utilities
├── CONTRIBUTING.md
├── RELEASE_NOTES.md          # Version scope and release history
├── THIRD_PARTY_NOTICES.md
├── VERSION
├── requirements.txt
└── LICENSE
```

## Scope

StateBridge targets homogeneous multi-agent systems in which every agent shares the same pretrained weights. Heterogeneous sender-receiver transfer is future work. This release provides the core method, prompts, dataset adapters, and evaluator; paper baselines and one-off analysis scripts are outside it. CLI and internal interfaces may change during the `0.x` series, see [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Data and Licensing

StateBridge code is released under the [Apache License 2.0](LICENSE). Third-party datasets and model weights are not covered by it. The bundled GPQA-Diamond transformation remains under CC BY 4.0 with attribution and changes recorded; the bundled MedQA subset remains under the MIT License, redistributed with its upstream copyright and permission notice. Model weights are not distributed here. See [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

Parts of the dataset-loading, model-wrapper, prompt, utility, and agent-definition infrastructure are adapted from [LatentMAS](https://github.com/Gen-Verse/LatentMAS) under Apache-2.0. Full provenance is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{peng2026statebridge,
  title     = {StateBridge: Training-free Hidden-state Alignment for Latent
               Communication in {LLM} Multi-Agent Systems},
  author    = {Peng, Yanwen and Zhang, Delvin Ce and Wang, Xi and Aletras, Nikolaos},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

Questions about the method: `ypeng86@sheffield.ac.uk`. Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
