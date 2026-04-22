# Test Layout

This suite uses a domain-first structure.

- `project/`: ProjectManager and chunk/audio/proofread behavior
- `api/`: API endpoint and router behavior (legacy flat files still being migrated)
- `editor_ui/`: legacy JS harness tests for editor/proofread UI logic
- `e2e/`: browser and fixture replay end-to-end flows

## Naming

Use `test_<domain>_<behavior>.py` for new files and keep each file focused on one behavior area.

## Running

- Full: `rtk app/env/bin/python -m pytest -q`
- Project domain: `rtk app/env/bin/python -m pytest -q app/tests/project`
- Editor UI domain: `rtk app/env/bin/python -m pytest -q app/tests/editor_ui`
- E2E domain: `rtk app/env/bin/python -m pytest -q app/tests/e2e`
- LM Studio live E2E lane: `rtk app/env/bin/python -m pytest -q app/tests/e2e --run-lmstudio-live-e2e -k lmstudio_live`
- Fresh-clone E2E lane: `rtk app/env/bin/python -m pytest -q app/tests/e2e --run-fresh-clone-e2e -k fresh_clone`
- Fresh-clone live real-backend E2E lane: `rtk app/env/bin/python -m pytest -q app/tests/e2e --run-fresh-clone-live-e2e -k fresh_clone_live`
- Fresh-clone live resumable partial mode: `rtk app/env/bin/python -m pytest -q app/tests/e2e --run-fresh-clone-live-e2e --fresh-clone-live-partial -k fresh_clone_live`
- Cross-platform sanity pass (warn-only): `rtk app/env/bin/python app/scripts/cross_platform_sanity_check.py`
- Cross-platform sanity pass (strict/non-zero on warnings): `rtk app/env/bin/python app/scripts/cross_platform_sanity_check.py --strict`

## Guidance

- Prefer extracting shared harness/setup into `_helpers.py` modules.
- Keep files scoped; avoid re-growing monoliths.
- The fresh-clone E2E lane is intentionally off by default because it clones `origin/main`
  and bootstraps a fresh `app/env` before driving the real UI.
- The LM Studio live E2E lane is intentionally off by default because it hits a real reachable
  local LM Studio backend and will run whenever that backend is available unless explicitly gated.
- The fresh-clone live lane is intentionally off by default because it also requires a reachable
  local LM Studio backend with at least one tool-capable model.
- The `--fresh-clone-live-partial` flag reuses the prior clone root and stage checkpoint after failures
  so reruns resume from the first incomplete stage.
- On successful completion of the live lane (partial or non-partial), test artifacts are cleaned up:
  clone root, checkpoint, and pointer state are removed, and only the finished MP3 is retained.
- The retained MP3 is copied to `${TMPDIR:-/tmp}/threadspeak_fresh_clone_live_outputs/` and that path
  is reported in the benchmark block (`Final MP3: path=...`) and cleanup log line.
- Non-partial runs also perform a preflight cleanup of any previously detected partial clone residue
  before starting.
- Local commits run the warn-only sanity checker via `.githooks/pre-commit` when
  `core.hooksPath` is set by `python app/scripts/install_git_hooks.py`.
