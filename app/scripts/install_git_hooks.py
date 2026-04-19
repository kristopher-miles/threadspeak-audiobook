"""Install local git hooks for this repository.

This script is intentionally non-fatal: it warns on failure and exits 0 so
install flows keep working in environments where git hooks are unavailable.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".githooks"
PRE_COMMIT_HOOK = HOOKS_DIR / "pre-commit"


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    if not PRE_COMMIT_HOOK.exists():
        print(f"[hooks] Skip: missing hook script at {PRE_COMMIT_HOOK}")
        return 0

    proc = _run_git("rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0:
        print("[hooks] Skip: repository is not a git work tree.")
        return 0

    set_proc = _run_git("config", "core.hooksPath", ".githooks")
    if set_proc.returncode != 0:
        stderr = (set_proc.stderr or "").strip()
        print(f"[hooks] Warning: failed to set core.hooksPath: {stderr}")
        return 0

    if os.name != "nt":
        mode = PRE_COMMIT_HOOK.stat().st_mode
        PRE_COMMIT_HOOK.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print("[hooks] Installed: git core.hooksPath -> .githooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
