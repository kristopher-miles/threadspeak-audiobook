# Prompt Customization

This page documents the current prompt system used by Threadspeak.

## Prompt keys currently supported

Prompt values are stored under `config.json -> prompts` and synchronized to prompt files.

Pair prompts (must contain `---SEPARATOR---` in file form):
- `system_prompt` / `user_prompt` -> `default_prompts.txt`
- `review_system_prompt` / `review_user_prompt` -> `review_prompts.txt`
- `attribution_system_prompt` / `attribution_user_prompt` -> `attribution_prompts.txt`

Single prompts:
- `voice_prompt` -> `voice_prompt.txt`
- `dialogue_identification_system_prompt` -> `dialogue_identification_system_prompt.txt`
- `temperament_extraction_system_prompt` -> `temperament_extraction_system_prompt.txt`

## Which prompts matter for the current Threadspeak pipeline

Primary (current non-legacy flow):
- `dialogue_identification_system_prompt`
- `temperament_extraction_system_prompt`
- `voice_prompt`

Also used in some maintenance/repair workflows:
- `attribution_system_prompt`
- `attribution_user_prompt`

Legacy-oriented pairs still exist in config/UI for compatibility:
- `system_prompt` / `user_prompt`
- `review_system_prompt` / `review_user_prompt`

## API behavior

- `GET /api/config`
  - Returns current config.
  - Backfills missing prompt keys from prompt files/fallbacks and persists those additions.

- `GET /api/default_prompts`
  - Returns defaults from current root prompt files (not factory bundle).

- `GET /api/factory_default_prompts`
  - Returns factory defaults from `app/prompt_defaults/`.

- `POST /api/config` and `POST /api/config/setup`
  - Merge incoming prompt fields with existing prompt config.
  - Synchronize prompt files on disk via server-side prompt sync.

## UI behavior (Setup -> Prompt Settings)

- Editing prompt fields in Setup and saving writes both `config.json` and corresponding prompt files.
- "Reset to Defaults" loads factory defaults (`/api/factory_default_prompts`) into fields.

## File format requirements

Pair files must contain exactly one separator token:

```text
...system prompt...
---SEPARATOR---
...user prompt...
```

Single-prompt files should contain plain text only.

## Practical recommendations

- Keep tool-calling/system prompts concise and deterministic.
- Test prompt changes on a short chapter before full runs.
- If behavior regresses, use factory reset and reapply minimal edits.
