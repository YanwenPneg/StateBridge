# Third-Party Notices

StateBridge source code is licensed under the Apache License 2.0 except where otherwise noted. Third-party materials remain subject to their own licenses and terms.

## LatentMAS

Portions of this repository, including dataset-loading, model-wrapper, prompt, utility, and agent-definition code, are adapted from [LatentMAS](https://github.com/Gen-Verse/LatentMAS).

LatentMAS is licensed under the Apache License, Version 2.0. StateBridge has modified the adapted code except where a source file states otherwise. A copy of the Apache License 2.0 is provided in [`LICENSE`](LICENSE).

LatentMAS citation:

```bibtex
@article{zou2025latentmas,
  title   = {Latent Collaboration in Multi-Agent Systems},
  author  = {Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal = {arXiv preprint arXiv:2511.20639},
  year    = {2025}
}
```

## GPQA Diamond

[`data/gpqa_diamond.json`](data/gpqa_diamond.json) is adapted from the Diamond subset of [GPQA](https://github.com/idavidrein/gpqa).

> GPQA (c) by Irving David Rein  
> GPQA is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Changes made for this repository: selected the 198-item Diamond subset; retained the revised question text; reordered and relabeled answer choices as A-D; converted the correct answer to a letter label; and removed explanations, validation metadata, record identifiers, the canary field, and other source fields.

No endorsement by the original authors is implied.

GPQA citation:

```bibtex
@inproceedings{rein2024gpqa,
  title     = {{GPQA}: A Graduate-Level Google-Proof Q\&A Benchmark},
  author    = {David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
  booktitle = {First Conference on Language Modeling},
  year      = {2024},
  url       = {https://openreview.net/forum?id=Ti67584b98}
}
```

## Runtime Dependencies and Model Weights

Python packages listed in `requirements.txt` are installed from their respective distributions and are not vendored in this repository. Each package remains subject to its own license.

Model weights are not distributed with StateBridge. Models downloaded at runtime are governed by the license and terms on their respective model pages; the StateBridge Apache-2.0 license does not grant rights to those weights. Model names are used for identification only and do not imply endorsement.
