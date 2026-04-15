# Editor Audio Fixture Worklog

## Goal

Capture real editor-phase line generation outputs for `test_book.epub` so the local Qwen simulator can replay full per-line audio generation deterministically in e2e tests.

This fixture set is for reproducible end-to-end flow validation, not model-quality scoring.

## Locked Inputs

- source book: `app/test_fixtures/files/test_book.epub`
- script seed fixture: `app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json`
- voice profile manifest: `app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json`
- editor generation workers: `1`

## Deliverables

- local Qwen editor fixture:
  - `app/test_fixtures/e2e_sim/qwen_local_editor_audio_test_book.json`
- captured line WAV assets:
  - `app/test_fixtures/e2e_sim/audio/editor_audio_test_book/*.wav`
- capture metadata manifest:
  - `app/test_fixtures/e2e_sim/editor_audio_test_book_manifest.json`
- replay test:
  - `app/test_e2e_editor_audio_fixture_replay.py`

## Runbook

```bash
rtk app/env/bin/python app/scripts/capture_editor_audio_fixtures.py \
  --source-book app/test_fixtures/files/test_book.epub \
  --script-seed-fixture app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json \
  --voice-profile-manifest app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json \
  --tts-local-backend auto
```

## Capture Records

- Status: partial (2 failed lines)
- Last run timestamp: 2026-04-15T07:47:12.929887+00:00
- Backend used: mlx
- Captured lines: 70
- Captured speakers: 6
- Qwen fixture path: `app/test_fixtures/e2e_sim/qwen_local_editor_audio_test_book.json`
- Manifest path: `app/test_fixtures/e2e_sim/editor_audio_test_book_manifest.json`
- Asset directory: `app/test_fixtures/e2e_sim/audio/editor_audio_test_book`
