# Contributing

StateBridge is a research release. The most useful contributions are reproduction reports, bug fixes, and extensions to new models or benchmarks.

Please open an issue before starting substantial work, so we can confirm the change fits the scope in [RELEASE_NOTES.md](RELEASE_NOTES.md). Paper baselines and one-off analysis scripts are outside this repository.

## Reporting a bug

Include the exact command you ran, the model, GPU type and count, the output of `pip list | grep -E "torch|transformers"`, and either the full traceback or the scores you obtained alongside the scores you expected.

Reproduction gaps count as bugs. If a documented command does not produce the documented number, we want to know.

## Pull requests

Keep each change focused. Match the existing style: type hints on public functions, docstrings in the format used in `methods/state_bridge.py`.

If your change could affect method behavior, run at least one benchmark before and after with a fixed `--seed` and report both numbers. Update `README.md`, `RELEASE_NOTES.md`, or `data/README.md` if you change the documented interface, defaults, or data handling.

## Adding a model or benchmark

A new model works without code changes if it exposes a chat template, standard input embeddings, and accepts `inputs_embeds`. Isolate any special handling in `models.py` rather than in the alignment code. Cross-family reproduction reports are directly useful evidence for the portability claim.

For a new benchmark, add a loader to `data.py` following the existing adapters and register the task in `ALL_TASKS`. Do not commit dataset files unless the license permits redistribution; document provenance and license in [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Licensing

Contributions are accepted under the [Apache License 2.0](LICENSE). Code adapted from another project must retain its attribution header and be recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contact: `ypeng86@sheffield.ac.uk`
