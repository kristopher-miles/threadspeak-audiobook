# Project Core Overview

`app/project_core` is the internal decomposition of `ProjectManager` from `app/project.py`.
It keeps existing behavior but separates responsibilities so each area is easier to reason about, test, and maintain.

## How It Is Composed

`ProjectManager` (in `app/project.py`) composes mixins in this order:

1. `ProjectIOStateMixin`
2. `ProjectRuntimeStateMixin`
3. `ProjectChunkStoreMixin`
4. `ProjectChunkEditingMixin`
5. `ProjectVoiceMixin`
6. `ProjectProofreadASRMixin`
7. `ProjectAudioRepairMixin`
8. `ProjectAudioExportMixin`
9. `ProjectGenerationMixin`

This order matters when one mixin depends on methods defined by an earlier one.

## Module Responsibilities

- `constants.py`
  - Shared thresholds and defaults for chunking, proofreading, repair batching, and export trimming.

- `chunking.py`
  - Script-entry normalization and chunk construction helpers.
  - Handles compatibility between legacy merged chunking and paragraph-id 1:1 chunk mapping.

- `mixins/io_state.py`
  - JSON load/save for config, state, paragraphs, source document, and transcription cache.

- `mixins/runtime_state.py`
  - In-memory runtime chunk overlays, generation token safety, and background flush/postprocess workers.

- `mixins/chunk_store.py`
  - Durable `chunks.json` lifecycle, chunk reference resolution, script-to-chunk sync, and integrity helpers.

- `mixins/chunk_editing.py`
  - Editor mutations: insert/delete/restore, chapter deletion, stale audio invalidation, split/merge structural repairs.

- `mixins/voice.py`
  - Speaker/narrator voice resolution, design voice support, and voice-config invalidation planning.

- `mixins/proofread_asr.py`
  - ASR transcription access, similarity scoring, and commit/reset proofreading states.

- `mixins/audio_repair.py`
  - Lost-audio relinking workflows (filename + ASR assisted matching).

- `mixins/audio_export.py`
  - Merge timeline generation and final export to MP3/M4B/Audacity artifacts, including optional trim/normalize.

- `mixins/generation.py`
  - TTS chunk generation orchestration (single, parallel, and batch pathways).

## Practical Boundary

`project_core` is intended as an internal implementation layer for `ProjectManager`.
The stable external API remains the methods exposed by `ProjectManager` and the API/router entrypoints that call it.
