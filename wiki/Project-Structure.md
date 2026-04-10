# Project Structure

This page reflects the current Threadspeak layout and where data actually lives at runtime.

## Path Model

Threadspeak is launched from `app/` (`start.js` runs `python app.py` in that directory), but the API uses two path roots:

- `BASE_DIR` = `repo/app`
- `ROOT_DIR` = `repo/` (parent of `app`)

Most active project artifacts (scripts, chunks, voices, exports, workflow state) are written under `ROOT_DIR` (repo root), not under `app/`.

## Source Code Layout

```text
threadspeak-audiobook.git/
├── app/
│   ├── app.py                         # Runtime entrypoint + compatibility export surface
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
│   ├── process_paragraphs.py          # Source -> paragraphs/chapters stage
│   ├── assign_dialogue.py             # Dialogue speaker identification stage
│   ├── extract_temperament.py         # Temperament extraction stage
│   ├── create_script.py               # Paragraphs -> script/chunks stage
│   ├── project.py                     # Core project/chunk/audio management
│   ├── proofread_runner.py            # Whisper-based proofread flow
│   ├── tts.py                         # TTS backends and batching
│   ├── static/                        # Frontend shell/fragments/scripts/styles
│   ├── prompt_defaults/               # Factory prompt presets used by reset/default APIs
│   └── requirements.txt
├── wiki/                              # Project docs (this page included)
├── .github/ISSUE_TEMPLATE/            # Issue forms/config
├── README.md
├── pinokio.js
├── install.js / start.js / reset.js / update.js
└── (runtime artifacts at repo root; see sections below)
```

## Runtime Data (Repo Root)

These paths are actively read/written during normal operation:

- `annotated_script.json` - canonical working script document
- `paragraphs.json` - ingestion output with paragraph/chapter structure
- `chunks.json` - line/chunk timeline used by editor/audio
- `voice_config.json` - speaker voice assignments and options
- `voices.json` - detected voice/speaker list
- `state.json` - selected source path + project-level flags
- `processing_workflow_state.json`, `new_mode_workflow_state.json` - workflow progress snapshots
- `audio_queue_state.json` - audio queue recovery state
- `script_sanity_check.json`, `script_repair_trace.jsonl` - sanity/repair outputs
- `transcription_cache.json` - ASR/proofread cache

## Runtime Directories (Repo Root)

- `uploads/` - imported source documents
- `scripts/` - saved script snapshots (`*.json`, companions)
- `saved_projects/` - project archive ZIPs
- `voicelines/` - generated line audio (includes `discarded/` pool)
- `clone_voices/` - uploaded clone reference audio + `manifest.json`
- `designed_voices/` - generated designed voices + `manifest.json` + `previews/`
- `dataset_builder/` - dataset builder working projects
- `lora_datasets/` - LoRA training datasets
- `lora_models/` - trained LoRA adapters
- `builtin_lora/` - bundled adapter presets + manifest
- `logs/` - service/debug logs

## Prompt Files (Repo Root)

Prompt values are synchronized between API config and these files:

- `default_prompts.txt`
- `review_prompts.txt`
- `attribution_prompts.txt`
- `voice_prompt.txt`
- `dialogue_identification_system_prompt.txt`
- `temperament_extraction_system_prompt.txt`

See [Prompt Customization](Prompt-Customization) for behavior and precedence.

## Practical Notes

- If you need to inspect or back up a project, focus on repo-root runtime files/directories, not only `app/`.
- Some similarly named paths may exist under `app/` from historical compatibility; the active API path constants in `app/api/shared.py` point core project state to repo root.
