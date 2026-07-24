# cjm-capability-demucs

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

A Demucs v4 source-separation capability for the cjm-substrate runtime that extracts vocals to remove background noise and music from speech audio.

## Modules

- **`cjm_capability_demucs.capability`** — Demucs v4 audio source separation capability — provides vocals extraction for removing background noise and music from speech audio.

## API

### `cjm_capability_demucs.capability`

- `DemucsCapabilityConfig` _class_ — Configuration for the Demucs processing capability.
- `DemucsProcessingCapability` _class_ — Demucs v4 source-separation tool capability for vocals extraction (stage 8: pure compute).

## Dependencies

**Depends on:** `cjm-capability-primitives`, `cjm-demucs-v4`, `cjm-substrate`, `cjm-substrate-torch-utils`
