# Voice Persistence Investigation

## Key Findings

- Voice state is already durably modeled in `chunks.sqlite3`:
  - `voice_profiles`
  - `voice_settings`
  - `chapter_narrator_overrides`
  - `voice_auto_aliases`
- The system is not DB-first in practice because JSON compatibility paths still exist:
  - `voice_config.json`
  - voice-related fields inside `state.json`
- Several read paths repair or infer voice state, which hides the real source of data loss:
  - `GET /api/voices` can auto-populate and save voice config during page load.
  - `load_chunks_view()` currently triggers narrator repair via `ensure_chapter_narrator_voice_can_narrate()`.
- Archive/save/load flows still export/import JSON compatibility data and can overwrite DB truth.
- Logging is currently insufficient to diagnose voice-state corruption or unexpected rewrites.

## Replacement Direction

- Make SQLite the only source of truth for all voice state.
- Remove JSON fallback/import/export paths for voice state.
- Keep the chapter narrator safety repair, but split it into:
  - a pure read decision,
  - one explicit durable write when repair is needed.
- Add structured project-local voice audit logging behind a single default-on flag.

## Implemented State

- Voice persistence is now DB-first through `chunks.sqlite3`.
- `voice_config.json` is no longer used as a persistence source during:
  - create-script
  - runtime voice save/load
  - script snapshot save/load
  - project archive save/load
- Voice-related fields are pruned from `state.json` on read/write:
  - `narrator_threshold`
  - `narrator_overrides`
  - `auto_narrator_aliases`
- `GET /api/voices` is now read-only and no longer persists inferred reusable voices during page load.
- Chapter narrator safety repair remains in place, but is now split into:
  - `get_chapter_narrator_voice_repair()` for read-time detection
  - `ensure_chapter_narrator_voice_can_narrate(..., repair=...)` for one explicit logged durable repair write
- Voice writes now increment a durable `voice_state_revision` and emit audit entries to `voice_state.audit.jsonl`.
- Voice audit logging is behind one default-on constant:
  - `VOICE_AUDIT_LOG_ENABLED_DEFAULT = True`
  - disabling it later should be a one-line default change
- Legacy compat method names still exist as inert no-ops so callers do not explode during transition:
  - `export_voice_config_compat()`
  - `export_voice_state_compat()`
  - `import_voice_compat()`

## Remaining Intentional Legacy References

- `VOICE_CONFIG_PATH` still exists only so stale legacy files can be deleted during reset/save/delete cleanup.
- `.voice_config.json` companion cleanup remains in script snapshot deletion paths to remove old artifacts.
- `voice_config_path` is still carried as an unused constructor attribute on the SQLite store and project manager for now; it is no longer used as an active persistence path.

## Verification

- Targeted suites passing after migration:
  - `app/test_workflow_entrypoints.py`
  - `app/test_project_archive.py`
  - `app/test_voice_reuse.py`
  - `app/test_project.py`
  - `app/test_script_provider.py`
  - `app/test_project_contract.py`
- Combined verification run:
  - `172 passed, 2 skipped`
