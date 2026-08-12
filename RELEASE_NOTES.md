# Release Notes

## v0.1.0 — Initial core research release

This is the first public StateBridge release. It provides the core training-free hidden-state alignment method and its evaluation entry point without claiming a complete reproduction package or a stable public API.

### Included

- Closed-form hidden-state alignment with centering, whitening, orthogonal Procrustes alignment, norm calibration, and vocabulary anchoring.
- Continuous-prefix transfer through `inputs_embeds` in a homogeneous four-agent pipeline.
- Single- and multi-GPU evaluation entry points for the supported reasoning, question-answering, and code-generation tasks.
- Dataset provenance, third-party notices, and GitHub-ready method and result figures.
- Default prefix scale set to `1.0`.

### Scope

- All agents in a run use the same pretrained model weights.
- Model weights are not distributed with this repository.
- Paper baselines, one-off ablations, and analysis scripts are outside this core release.
- CLI and internal interfaces may change during the `0.x` series.

### Versioning

StateBridge follows [Semantic Versioning](https://semver.org/) while its public interface stabilizes:

- `0.1.x` — fixes and documentation updates that preserve the current interface and method behavior.
- `0.2.0` — new core capabilities, supported tasks or models, or changes to public defaults and interfaces.
- `1.0.0` — a stable public CLI/API and release scope.
