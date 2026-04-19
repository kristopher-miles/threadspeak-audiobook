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
- Fresh-clone E2E lane: `rtk app/env/bin/python -m pytest -q app/tests/e2e --run-fresh-clone-e2e -k fresh_clone`
- Cross-platform sanity pass (warn-only): `rtk app/env/bin/python app/scripts/cross_platform_sanity_check.py`
- Cross-platform sanity pass (strict/non-zero on warnings): `rtk app/env/bin/python app/scripts/cross_platform_sanity_check.py --strict`

## Guidance

- Prefer extracting shared harness/setup into `_helpers.py` modules.
- Keep files scoped; avoid re-growing monoliths.
- The fresh-clone E2E lane is intentionally off by default because it clones `origin/main`
  and bootstraps a fresh `app/env` before driving the real UI.
- Local commits run the warn-only sanity checker via `.githooks/pre-commit` when
  `core.hooksPath` is set by `python app/scripts/install_git_hooks.py`.
