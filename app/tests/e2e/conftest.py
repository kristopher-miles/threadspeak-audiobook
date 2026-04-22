import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-fresh-clone-e2e",
        action="store_true",
        default=False,
        help="run the opt-in fresh-clone end-to-end flow that boots a git clone of the configured ref (default: HEAD) and overlays local worktree changes by default",
    )
    parser.addoption(
        "--run-fresh-clone-live-e2e",
        action="store_true",
        default=False,
        help="run the opt-in heavy fresh-clone live end-to-end flow that uses real local LM Studio and local voice generation",
    )
    parser.addoption(
        "--run-lmstudio-live-e2e",
        action="store_true",
        default=False,
        help="run the opt-in LM Studio live E2E tests that hit a real local LM Studio backend",
    )
    parser.addoption(
        "--run-fresh-clone-live-narrated-e2e",
        action="store_true",
        default=False,
        help="run the opt-in heavy fresh-clone live narrated end-to-end flow (full-cast toggle off) against real local backends",
    )
    parser.addoption(
        "--fresh-clone-live-partial",
        action="store_true",
        default=False,
        help="for the live fresh-clone E2E lane, keep and reuse the last clone root and resume from checkpointed stages",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "fresh_clone_e2e: requires --run-fresh-clone-e2e because it clones a git ref into a fresh app env (set THREADSPEAK_E2E_FRESH_CLONE_REF / THREADSPEAK_E2E_FRESH_CLONE_INCLUDE_WORKTREE to override)",
    )
    config.addinivalue_line(
        "markers",
        "fresh_clone_live_e2e: requires --run-fresh-clone-live-e2e because it runs a heavy fresh-clone full-project flow against live local LLM/TTS backends",
    )
    config.addinivalue_line(
        "markers",
        "lmstudio_live_e2e: requires --run-lmstudio-live-e2e because it hits a real local LM Studio backend",
    )
    config.addinivalue_line(
        "markers",
        "fresh_clone_live_narrated_e2e: requires --run-fresh-clone-live-narrated-e2e because it runs a heavy fresh-clone narrated full-project flow against live local LLM/TTS backends",
    )


def pytest_collection_modifyitems(config, items):
    run_fresh_clone = bool(config.getoption("--run-fresh-clone-e2e"))
    run_fresh_clone_live = bool(config.getoption("--run-fresh-clone-live-e2e"))
    run_lmstudio_live = bool(config.getoption("--run-lmstudio-live-e2e"))
    run_fresh_clone_live_narrated = bool(config.getoption("--run-fresh-clone-live-narrated-e2e"))
    fresh_clone_skip = pytest.mark.skip(reason="requires --run-fresh-clone-e2e")
    fresh_clone_live_skip = pytest.mark.skip(reason="requires --run-fresh-clone-live-e2e")
    lmstudio_live_skip = pytest.mark.skip(reason="requires --run-lmstudio-live-e2e")
    fresh_clone_live_narrated_skip = pytest.mark.skip(reason="requires --run-fresh-clone-live-narrated-e2e")
    for item in items:
        if "fresh_clone_e2e" in item.keywords and not run_fresh_clone:
            item.add_marker(fresh_clone_skip)
        if "fresh_clone_live_e2e" in item.keywords and not run_fresh_clone_live:
            item.add_marker(fresh_clone_live_skip)
        if "lmstudio_live_e2e" in item.keywords and not run_lmstudio_live:
            item.add_marker(lmstudio_live_skip)
        if "fresh_clone_live_narrated_e2e" in item.keywords and not run_fresh_clone_live_narrated:
            item.add_marker(fresh_clone_live_narrated_skip)
