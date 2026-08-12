# Contributing to StateBridge

Thanks for your interest in StateBridge. This is a research release, so the most valuable contributions are reproduction reports, bug fixes, and extensions to new models or benchmarks.

## Before you start

StateBridge is in the `0.x` series. The CLI and internal interfaces may change between minor versions. Please open an issue before starting substantial work so we can confirm the change fits the release scope described in [RELEASE_NOTES.md](RELEASE_NOTES.md).

Paper baselines, one-off ablations, and analysis scripts are outside this repository's scope.

## Reporting a bug

Open an issue using the bug report template. A useful report includes:

- The exact command you ran, including all flags.
- Model name, GPU type and count, and the output of `pip list | grep -E "torch|transformers"`.
- The full traceback, or the scores you obtained alongside the scores you expected.

Reproduction gaps are bugs. If a documented command does not produce the documented number, we want to know.

## Requesting a feature

Open an issue using the feature request template. Describe the research question the feature would let you answer, not only the interface you have in mind.

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Keep the change focused. One concern per pull request.
3. Match the existing style: type hints on public functions, docstrings in the format used in `methods/state_bridge.py`.
4. Verify your change does not alter method behavior unintentionally. Run at least one benchmark before and after with a fixed `--seed` and report both numbers in the pull request.
5. Update `README.md`, `RELEASE_NOTES.md`, or `data/README.md` if your change affects the documented interface, defaults, or data handling.

## Adding a new model

`models.py` wraps Hugging Face `AutoModelForCausalLM`. A new model works without code changes if it exposes a chat template, standard input embeddings, and accepts `inputs_embeds`. If it needs special handling, isolate that handling in `models.py` rather than in the alignment code.

Please report the model, the benchmarks you ran, and the scores. Cross-family evidence is directly useful to the paper's portability claim.

## Adding a new benchmark

Add a loader to `data.py` following the existing adapters, and register the task in the CLI choices in `methods/state_bridge.py`. Answer extraction and scoring belong in `_extract_answer` and `run_item`.

Do not commit dataset files unless the license permits redistribution. Document provenance, license, and any transformation you applied in [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), following the pattern used for GPQA-Diamond.

## Licensing

Contributions are accepted under the [Apache License 2.0](LICENSE). Code adapted from another project must retain its attribution header and be recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

For questions about the method or the paper, contact Yanwen Peng at `ypeng86@sheffield.ac.uk`.
