# Project Structure

This page reflects the current Threadspeak layout and where data actually lives at runtime.

## Path Model

Threadspeak is launched from `app/` (`start.js` runs `python app.py` in that directory), but runtime state is no longer stored in repo root.

- `BASE_DIR` = `repo/app`
- `REPO_ROOT` = `repo/`
- `config/prompts/` = editable prompt files
- `runtime/current/project/` = active project state, user assets, and final exports
- `runtime/runs/<run_id>/` = per-run temp files, scratch work, and transient logs

The path contract is centralized in [`app/runtime_layout.py`](../app/runtime_layout.py).

## Source Code Layout

```text
threadspeak-audiobook.git/
├── app/
│   ├── app.py                         # Runtime entrypoint + compatibility export surface
│   ├── runtime_layout.py              # Centralized config/runtime path definitions
│   ├── api/
│   │   ├── main.py                    # Router composition
│   │   ├── shared.py                  # Global paths/state/models/schemas
│   │   └── routers/
│   │       ├── config_router.py       # Config/prompts/upload/setup endpoints
│   │       ├── workflow_router.py     # Ingestion and workflow orchestration
│   │       ├── voices_router.py       # Voice config/suggestion/dictionary endpoints
│   │       ├── editor_audio_router.py # Chunk/audio/proofread/export endpoints
│   │       ├── scripts_router.py      # Save/load/archive endpoints
│   │       ├── voice_designer_router.py
│   │       ├── clone_voices_router.py
│   │       ├── lora_router.py
│   │       └── dataset_builder_router.py
│   ├── project.py                     # Core project/chunk/audio management
│   ├── scripts/                       # Operational CLI/workflow entrypoints
│   │   ├── process_paragraphs.py
│   │   ├── assign_dialogue.py
│   │   ├── extract_temperament.py
│   │   ├── create_script.py
│   │   ├── generate_script.py
│   │   ├── review_script.py
│   │   ├── proofread_runner.py
│   │   ├── lost_audio_repair_runner.py
│   │   ├── train_lora.py
│   │   └── migrate_runtime_layout.py
│   ├── resources/
│   │   └── builtin_lora/              # Bundled adapter presets + manifest
│   ├── tts.py                         # TTS backends and batching
│   ├── static/                        # Frontend shell/fragments/scripts/styles
│   ├── prompt_defaults/               # Factory prompt presets used by reset/default APIs
│   └── requirements.txt
├── config/
│   └── prompts/                       # Editable prompt files used by the app
├── runtime/
│   ├── current/
│   │   └── project/                   # Active project state and assets
│   └── runs/                          # Run-scoped temp/log/export scratch space
├── wiki/                              # Project docs (this page included)
├── .github/ISSUE_TEMPLATE/            # Issue forms/config
├── README.md
├── pinokio.js
├── install.js / start.js / reset.js / update.js
└── icon.png / favicon.ico / manifests
```

## Active Project Data

Persistent project files live under `runtime/current/project/`:

- `annotated_script.json` - canonical working script document
- `paragraphs.json` - ingestion output with paragraph/chapter structure
- `chunks.json` - line/chunk timeline used by editor/audio
- `voice_config.json` - speaker voice assignments and options
- `voices.json` - detected voice/speaker list
- `state.json` - selected source path + project-level flags

## Project Subdirectories

- `workflow/`
  - `processing_workflow_state.json`, `new_mode_workflow_state.json`
  - `audio_queue_state.json`, `audio_cancel_tombstone.json`
  - `script_generation_checkpoint.json`, `script_review_checkpoint.json`
- `db/`
  - `chunks.sqlite3`
  - `chunks.queue.log`
  - `transcription_cache.json`
  - `voice_state.audit.jsonl`
- `repair/`
  - `script_sanity_check.json`
  - `script_repair_trace.jsonl`
- `exports/`
  - `cloned_audiobook.mp3`
  - `optimized_audiobook.zip`
  - `audacity_export.zip`
  - `audiobook.m4b`
  - `m4b_cover.jpg`
  - sanity preview/audio assembly outputs
- `uploads/` - imported source documents
- `voicelines/` - generated line audio, including `discarded/`
- `clone_voices/` - uploaded clone reference audio + `manifest.json`
- `designed_voices/` - generated designed voices + `manifest.json` + `previews/`
- `archives/script_snapshots/` - saved script snapshots and companions
- `archives/project_archives/` - saved project archive ZIPs
- `archives/backups/` - chunk/script backups
- `workspace/dataset_builder/` - dataset builder working projects
- `workspace/lora_datasets/` - LoRA training datasets
- `workspace/lora_models/` - trained LoRA adapters

## Run-Scoped Scratch Space

Transient files created while a job is running live under `runtime/runs/<run_id>/`:

- `tmp/` - merge/export scratch directories, temp WAVs, temp metadata files, import extracts
- `logs/` - run-specific LLM/review logs and transient diagnostics
- `exports/` - optional run-specific staging outputs before promotion into the active project

## Editable Prompts

Prompt values are synchronized between API config and `config/prompts/`:

- `config/prompts/default_prompts.txt`
- `config/prompts/review_prompts.txt`
- `config/prompts/attribution_prompts.txt`
- `config/prompts/voice_prompt.txt`
- `config/prompts/dialogue_identification_system_prompt.txt`
- `config/prompts/temperament_extraction_system_prompt.txt`

See [Prompt Customization](Prompt-Customization) for behavior and precedence.

## Practical Notes

- If you need to inspect or back up a project, focus on `runtime/current/project/`.
- If you are debugging transient artifacts, inspect `runtime/runs/` rather than repo root.
- Repo root should now contain launcher files, docs, manifests, and source code, not active runtime state.
