# Generate Script Fixture Worklog

## Goal

Produce a committed LM Studio simulator fixture for the **Generate Script** stage using real responses captured from an isolated app run with:
- input book: `app/test_fixtures/files/test_book.epub`
- model target: `qwen/qwen3.5-9b`
- LM Studio base URL: `http://127.0.0.1:1234`
- deterministic capture setting: `llm_workers=1`

This fixture is used to replay realistic, valid responses for e2e testing (not to validate model quality).

## Locked Decisions

- Capture path: API workflow (`/api/upload` + `/api/new_mode_workflow/start`) in isolated app clone.
- Prompt/config policy: pinned capture config in isolated clone.
- Model fallback policy: fail hard if target model is unavailable/unloadable.
- Fixture lifecycle: committed static fixture; refresh only on explicit request.

## Non-Legacy Enforcement (Mandatory)

All fixture capture and e2e flows for this test must use the modern non-legacy workflow only.

Forbidden endpoints for this test (must never be called):
- `POST /api/generate_script`
- `POST /api/review_script`
- `POST /api/script_sanity_check`
- `POST /api/replace_missing_chunks`
- `POST /api/processing/start`
- `POST /api/processing/pause`
- `POST /api/chunks/repair_legacy`
- `GET /api/annotated_script`

Required workflow endpoint for script stage:
- `POST /api/new_mode_workflow/start` (with `process_voices=false`, `generate_audio=false` for stage-1 capture)

No-shortcut policy (mandatory):
- Never seed or inject script/voice/editor state to bypass earlier phases when validating this workflow.
- Script capture data must come from a real non-legacy ingestion run against the uploaded test EPUB.

## Deliverables

- Capture utility script that:
  - runs isolated app clone,
  - validates/loads target LM Studio model,
  - runs non-legacy script workflow (`/api/new_mode_workflow/start`) with `test_book.epub`,
  - records live LM Studio `/v1/chat/completions` traffic (including streaming tool calls) via capture proxy,
  - emits simulator fixture JSON for `/v1/chat/completions` replay.
- Committed fixture file for Generate Script.
- Replay e2e test using the new fixture.

## Runbook

```bash
rtk app/env/bin/python app/scripts/capture_generate_script_fixture.py \
  --model-name qwen/qwen3.5-9b
```

Optional output path override:

```bash
rtk app/env/bin/python app/scripts/capture_generate_script_fixture.py \
  --output app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json
```

## Capture Records

- Status: complete
- Last run timestamp: 2026-04-15T07:10:01.971952+00:00
- Captured run_id: new-mode-01c41d67-fb15-4a15-8c98-737554dcff0c
- Captured model: `qwen/qwen3.5-9b`
- Captured chat calls: 78
- Fixture path: `app/test_fixtures/e2e_sim/lmstudio_generate_script_test_book.json`
- Notes: captured from isolated non-legacy workflow (`/api/new_mode_workflow/start`) with llm_workers=1
