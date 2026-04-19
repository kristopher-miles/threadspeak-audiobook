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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "app"
WIN_ENV_PY = APP_DIR / "env" / "Scripts" / "python.exe"
POSIX_ENV_PY = APP_DIR / "env" / "bin" / "python"
EXTRA_ARGS_ENV = "THREADSPEAK_TEST_PYTEST_ARGS"
SPLIT_STAGE_ENV = "THREADSPEAK_E2E_RUN_SPLIT_STAGE_UI"
TMP_ROOT_ENV = "THREADSPEAK_TEST_TMPDIR"


def _runtime_tmp_root() -> Path:
    raw = str(os.environ.get(TMP_ROOT_ENV, "")).strip()
    root = Path(raw) if raw else (REPO_ROOT / "cache" / "t")
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        help="Also run dedicated fresh-clone E2E lane after the primary run.",
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
    runtime_tmp = _runtime_tmp_root()
    env[TMP_ROOT_ENV] = str(runtime_tmp)
    env["TMPDIR"] = str(runtime_tmp)
    env["TEMP"] = str(runtime_tmp)
    env["TMP"] = str(runtime_tmp)
    cache_dir = (runtime_tmp / "pytest_cache").as_posix()
    env["PYTEST_ADDOPTS"] = (
        f"{env.get('PYTEST_ADDOPTS', '').strip()} -o cache_dir={cache_dir}"
    ).strip()
    if args.split_stage_ui:
        env[SPLIT_STAGE_ENV] = "1"
    else:
        env.pop(SPLIT_STAGE_ENV, None)

    primary_cmd = [str(python_bin), "-m", "pytest", "-q"]
    if args.fresh_clone_e2e:
        primary_cmd.append("--run-fresh-clone-e2e")
    primary_cmd.extend(_split_args(args.pytest_args))
    rc = _run(primary_cmd, cwd=APP_DIR, env=env)
    if rc != 0:
        return rc

    if args.fresh_clone_e2e:
        fresh_cmd = [
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            "tests/e2e",
            "--run-fresh-clone-e2e",
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
