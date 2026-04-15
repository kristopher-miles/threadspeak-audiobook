# Voice Profile Fixture Worklog

## Goal

Capture reusable-voice generation fixtures for `test_book.epub` that cover:
- LM Studio suggestion prompts per speaker (`/api/voices/suggest_descriptions_bulk`)
- local Qwen voice-design audio outputs per speaker (`/api/voices/design_generate`)

These fixtures are for deterministic e2e replay of voice-profile flows, not model-quality scoring.

## Locked Inputs

- source book: `app/test_fixtures/files/test_book.epub`
- script seed fixture: `app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json`
- LM Studio base URL: `http://127.0.0.1:1234`
- LM Studio suggestion model: `qwen/qwen3.5-9b`
- local TTS mode: `local`
- local backend: `mlx` (auto-selected because `qwen_tts` is unavailable in this environment)
- llm workers: `1`

## Deliverables

- LM Studio fixture:
  - `app/test_fixtures/e2e_sim/lmstudio_voice_profiles_test_book.json`
- local Qwen fixture:
  - `app/test_fixtures/e2e_sim/qwen_local_voice_profiles_test_book.json`
- captured WAV assets:
  - `app/test_fixtures/e2e_sim/audio/voice_profiles_test_book/*.wav`
- capture metadata:
  - `app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json`
- replay test:
  - `app/test_e2e_voice_profile_fixture_replay.py`

## Runbook

```bash
rtk app/env/bin/python app/scripts/capture_voice_profile_fixtures.py \
  --lmstudio-base-url http://127.0.0.1:1234 \
  --llm-model qwen/qwen3.5-9b \
  --tts-local-backend mlx
```

## Capture Records

- Status: complete
- Last run timestamp: 2026-04-15T07:19:17.821747+00:00
- Speakers: Bitera, Father, Kisten, Maddie, Mother, NARRATOR
- LM fixture path: `app/test_fixtures/e2e_sim/lmstudio_voice_profiles_test_book.json`
- Qwen fixture path: `app/test_fixtures/e2e_sim/qwen_local_voice_profiles_test_book.json`
- Asset directory: `app/test_fixtures/e2e_sim/audio/voice_profiles_test_book`
- Notes: captured from isolated API workflow seeded from Generate Script fixture, then live LM suggestions + local Qwen design generation
