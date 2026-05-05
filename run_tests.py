#!/usr/bin/env python3
"""Canonical Threadspeak test harness.

Run this script exactly as the standard entrypoint for tests:
  python run_tests.py

Do not invent ad-hoc pytest command variants in automation. This harness is the
single source of truth for deterministic test execution.

Default behavior (no args):
1) Run full suite: pytest -q
2) Run cross-platform sanity check (warn-only)

Optional flags are documented in --help and are the only supported variants.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "app"
TEST_SUPPORT_DIR = APP_DIR / "test_support"
WIN_ENV_PY = APP_DIR / "env" / "Scripts" / "python.exe"
POSIX_ENV_PY = APP_DIR / "env" / "bin" / "python"
EXTRA_ARGS_ENV = "THREADSPEAK_TEST_PYTEST_ARGS"
SPLIT_STAGE_ENV = "THREADSPEAK_E2E_RUN_SPLIT_STAGE_UI"


def _resolve_env_python() -> Path:
    if WIN_ENV_PY.exists():
        return WIN_ENV_PY
    if POSIX_ENV_PY.exists():
        return POSIX_ENV_PY
    raise FileNotFoundError(
        "Missing app test environment python.\n"
        f"Expected one of:\n  - {WIN_ENV_PY}\n  - {POSIX_ENV_PY}\n"
        "Run install first, then retry."
    )


def _split_args(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    return shlex.split(raw, posix=(os.name != "nt"))


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    rendered = " ".join(cmd)
    print(f"[test-harness] Running: {rendered}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _temp_root_is_writable(path: str) -> bool:
    candidate = Path(path).expanduser()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(candidate), delete=True):
            pass
    except Exception:
        return False
    return True


def _ensure_writable_temp_env(env: dict[str, str]) -> None:
    preferred_roots = [
        env.get("TMPDIR"),
        env.get("TMP"),
        env.get("TEMP"),
        tempfile.gettempdir(),
    ]
    selected_root = None
    for root in preferred_roots:
        if root and _temp_root_is_writable(root):
            selected_root = Path(root).expanduser()
            break

    if selected_root is None:
        selected_root = REPO_ROOT / ".tmp_test_runtime"
        selected_root.mkdir(parents=True, exist_ok=True)

    selected_text = str(selected_root)
    env["TMPDIR"] = selected_text
    env["TEMP"] = selected_text
    env["TMP"] = selected_text


def _ensure_test_support_path(env: dict[str, str]) -> None:
    current = env.get("PYTHONPATH", "")
    entries = [str(TEST_SUPPORT_DIR)]
    if current:
        entries.append(current)
    env["PYTHONPATH"] = os.pathsep.join(entries)


def _has_explicit_pytest_selection(pytest_args: str) -> bool:
    extra_args = _split_args(pytest_args)
    if not extra_args:
        return False
    return any(not str(arg).startswith("-") for arg in extra_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical Threadspeak test runner. Use this instead of custom pytest commands."
    )
    parser.add_argument(
        "--pytest-args",
        default=os.environ.get(EXTRA_ARGS_ENV, ""),
        help=f"Additional args appended to primary 'pytest -q' run. Default from ${EXTRA_ARGS_ENV}.",
    )
    parser.add_argument(
        "--fresh-clone-e2e",
        action="store_true",
        help="Also run dedicated fresh-clone E2E lane after the primary run (passes --critical-path-e2e).",
    )
    parser.add_argument(
        "--split-stage-ui",
        action="store_true",
        help="Enable split Stage-UI tests (Stage 1-6) that are skipped by default.",
    )
    parser.add_argument(
        "--sanity-strict",
        action="store_true",
        help="Run cross-platform sanity check in strict mode (non-zero on findings).",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip cross-platform sanity check step.",
    )
    args = parser.parse_args()

    try:
        python_bin = _resolve_env_python()
    except FileNotFoundError as exc:
        print(f"[test-harness] ERROR: {exc}", flush=True)
        return 2

    env = os.environ.copy()
    _ensure_writable_temp_env(env)
    _ensure_test_support_path(env)
    if args.split_stage_ui:
        env[SPLIT_STAGE_ENV] = "1"
    else:
        env.pop(SPLIT_STAGE_ENV, None)

    primary_cmd = [str(python_bin), "-m", "pytest", "-q"]
    if args.fresh_clone_e2e:
        primary_cmd.append("--critical-path-e2e")
    primary_cmd.extend(_split_args(args.pytest_args))
    rc = _run(primary_cmd, cwd=APP_DIR, env=env)
    if rc != 0:
        return rc

    if args.fresh_clone_e2e and not _has_explicit_pytest_selection(args.pytest_args):
        fresh_cmd = [
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            "tests/e2e",
            "--critical-path-e2e",
            "-k",
            "fresh_clone",
        ]
        rc = _run(fresh_cmd, cwd=APP_DIR, env=env)
        if rc != 0:
            return rc

    if not args.skip_sanity:
        sanity_cmd = [str(python_bin), "scripts/cross_platform_sanity_check.py"]
        if args.sanity_strict:
            sanity_cmd.append("--strict")
        rc = _run(sanity_cmd, cwd=APP_DIR, env=env)
        if rc != 0:
            return rc

    print("[test-harness] Completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
