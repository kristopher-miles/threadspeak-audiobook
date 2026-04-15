# Proofread Fixture Worklog

## Goal

Create a reusable text fixture for proofread runs in test mode so `/api/proofread` can execute without real ASR latency while still using normal internal scoring/parsing paths.

This fixture targets the non-legacy workflow artifacts captured from `test_book.epub`.

## Locked Inputs

- full Qwen harness fixture: `app/test_fixtures/e2e_sim/qwen_local_full_e2e_test_book.json`
- editor audio manifest: `app/test_fixtures/e2e_sim/editor_audio_test_book_manifest.json`
- output fixture: `app/test_fixtures/e2e_sim/proofread_text_test_book.json`
- fallback mode: `chunk_text`

## Deliverables

- proofread transcript fixture:
  - `app/test_fixtures/e2e_sim/proofread_text_test_book.json`
- fixture build script:
  - `app/scripts/build_proofread_text_fixture.py`
- simulator provider:
  - `app/e2e_sim/proofread_text_sim.py`
- integration tests:
  - `app/test_proofread_text_sim.py`

## Runbook

```bash
rtk app/env/bin/python app/scripts/build_proofread_text_fixture.py
```

## Capture Records

- Status: complete
- Last run source: non-legacy full harness capture artifacts
- Entries generated: 72
- Fallback policy: `chunk_text`
