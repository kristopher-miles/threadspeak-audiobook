# E2E Simulators (LM Studio + Local Qwen)

This repository now includes test-only simulation resources for true end-to-end tests without real model latency.

The simulators keep runtime call paths intact:
- LM Studio simulation is HTTP-based and is called through the existing OpenAI-compatible client path.
- Qwen simulation stays on `tts.mode=local` and is injected only into local model initialization.

## Safety

Simulation is disabled unless explicitly enabled.

Default runtime behavior is unchanged when these env vars are not set.

## Environment Variables

- `THREADSPEAK_E2E_SIM_ENABLED=1`
  - Enables simulator-aware behavior in local Qwen model initialization.
- `THREADSPEAK_E2E_QWEN_FIXTURE=/abs/path/to/qwen_fixture.json`
  - Fixture path for local Qwen simulation.
- `THREADSPEAK_E2E_QWEN_REPORT_PATH=/tmp/qwen-sim-report.json`
  - Optional report file written by the local Qwen simulator (pending/consumed state).
- `THREADSPEAK_E2E_SIM_STRICT=1`
  - Strict fixture matching for both simulators. Defaults to strict when fixture omits `strict`.
- `THREADSPEAK_E2E_PROOFREAD_FIXTURE=/abs/path/to/proofread_text_fixture.json`
  - Enables transcript replay for proofread/ASR flows.
- `THREADSPEAK_E2E_PROOFREAD_FALLBACK=chunk_text`
  - Missing-entry policy for proofread transcript fixture (`chunk_text` or `fail`).
- `THREADSPEAK_E2E_PROOFREAD_REPORT_PATH=/tmp/proofread-report.json`
  - Optional report output for proofread transcript replay.
- `THREADSPEAK_E2E_PROOFREAD_TRACE_PATH=/tmp/proofread-trace.jsonl`
  - Optional trace output for proofread transcript replay.

## LM Studio Fixture Format

Example fixture paths:
- `app/test_fixtures/e2e_sim/lmstudio_voice_description.json`
- `app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json`
- `app/test_fixtures/e2e_sim/lmstudio_voice_profiles_test_book.json`
- `app/test_fixtures/e2e_sim/qwen_local_editor_audio_test_book.json`

Top-level shape:

```json
{
  "strict": true,
  "routes": {
    "GET /api/v1/models": [
      { "response": { "models": [] } }
    ],
    "POST /v1/chat/completions": [
      {
        "expect": { "model": "sim-tool-model" },
        "response": { "id": "...", "choices": [...] }
      }
    ]
  }
}
```

Rules:
- Keys are `"<METHOD> <PATH>"`.
- Each route has a FIFO list of scripted interactions.
- `expect` is a partial JSON subset matcher against request payload.
- `response` is returned as JSON.
- Optional `stream_events` emits SSE chunks for streaming endpoints.
- In strict mode, unexpected or mismatched requests fail the test.

## Local Qwen Fixture Format

Path: `app/test_fixtures/e2e_sim/qwen_local_voice_design.json`

Top-level shape:

```json
{
  "strict": true,
  "sample_rate": 24000,
  "default_duration_ms": 280,
  "methods": {
    "generate_voice_design": [
      {
        "expect": { "text": "...", "instruct": "..." },
        "audio": { "kind": "tone", "duration_ms": 420 }
      }
    ]
  }
}
```

Supported method keys:
- `generate_custom_voice`
- `create_voice_clone_prompt`
- `generate_voice_clone`
- `generate_voice_design`

## Proofread Transcript Fixture Format

Path:
- `app/test_fixtures/e2e_sim/proofread_text_test_book.json`

Top-level shape:

```json
{
  "strict": true,
  "fallback_mode": "chunk_text",
  "entries": [
    {
      "audio_path": "voicelines/voiceline_<uid>_narrator.mp3",
      "transcript_text": "Transcript text returned to proofread.",
      "uid": "<optional>",
      "speaker": "NARRATOR",
      "line_id": 0
    }
  ]
}
```

Rules:
- `audio_path` must match the clip path loaded in chunk data.
- `transcript_text` is returned through the normal `transcribe_audio_path` path.
- If an entry is missing and `fallback_mode=chunk_text`, test mode derives transcript from current chunk text.
- If strict is enabled and no fixture/fallback transcript is available, proofread hard-fails.

Audio shape options per interaction:
- `audio` object for one output spec (replicated as needed).
- `audios` list for explicit per-item outputs in batch-like calls.
- `audio_wav_base64` for direct WAV payload fixtures.
- `audio_wav_path` for WAV files stored on disk (absolute path or path relative to the fixture file).
- `audios[].wav_path` for per-item WAV file paths.

## Running

Run the simulator resource tests:

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_sim_resources.py
```

Run the proofread transcript simulator tests:

```bash
rtk app/env/bin/python -m pytest -q app/test_proofread_text_sim.py
```

Run the captured Generate Script replay test:

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_generate_script_fixture_replay.py
```

Run the captured Voice Profile replay test:

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_voice_profile_fixture_replay.py
```

Run the captured Editor Audio replay test:

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_editor_audio_fixture_replay.py
```

Run the full harness replay test (script + voices + editor audio):

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_full_harness_fixture_replay.py
```

Run the stage-1 UI E2E test (non-legacy Process Script flow):

```bash
rtk app/env/bin/python -m pytest -q app/test_e2e_stage1_script_ui.py
```

One-time browser setup (required on new environments):

```bash
rtk app/env/bin/python -m playwright install chromium
```

Stage-1 UI rule:
- Once browser flow begins, do not bypass UI actions.
- All runtime operations must come from clicks/form inputs or direct selector-targeted UI interactions.
- No direct API task triggers during the browser phase.

Top-level E2E continuity rule:
- Never take shortcuts to make later phases pass.
- Phase 2 must start from Phase 1's real UI run output; Phase 3 must start from Phase 2's real UI run output.
- No direct task/data seeding between phases once the browser flow begins.
- Any attempted shortcut (manual API trigger, injected phase data, pre-seeded phase state) is invalid and must be reported immediately.

Regenerate the Generate Script fixture from live LM Studio responses:

```bash
rtk app/env/bin/python app/scripts/capture_generate_script_fixture.py \
  --model-name qwen/qwen3.5-9b
```

Regenerate the Voice Profile fixtures (LM suggestions + local Qwen voice-design assets):

```bash
rtk app/env/bin/python app/scripts/capture_voice_profile_fixtures.py \
  --lmstudio-base-url http://127.0.0.1:1234 \
  --llm-model gemma-4-e2b-it \
  --tts-local-backend auto
```

Regenerate the Editor Audio fixtures (real local Qwen line generation, workers=1):

```bash
rtk app/env/bin/python app/scripts/capture_editor_audio_fixtures.py \
  --source-book app/test_fixtures/files/test_book.epub \
  --script-seed-fixture app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json \
  --voice-profile-manifest app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json \
  --tts-local-backend auto
```

Build the full harness profile (phase fixtures + combined Qwen fixture):

```bash
rtk app/env/bin/python app/scripts/build_full_e2e_harness_profile.py
```

Build the proofread transcript fixture from captured non-legacy editor artifacts:

```bash
rtk app/env/bin/python app/scripts/build_proofread_text_fixture.py
```

The test starts:
- an isolated app server clone,
- a local LM Studio simulator server,
- local Qwen simulation through env-gated provider injection.

Then it validates:
- `/api/voices/suggest_description` via LM Studio simulation,
- `/api/voices/design_generate` via local Qwen simulation,
- `/api/generate_batch` line-by-line editor generation via local Qwen simulation,
- `/api/proofread` transcript reads via proofread text fixture simulation,
- fixture consumption (no unconsumed scripted interactions).
