"""Warn-only heuristic checker for likely cross-platform execution hazards.

This is intentionally broad and lightweight. It flags suspicious text patterns
that often break either Windows or macOS/Linux when added without guards.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "cross_platform_sanity.json"
SCANNABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".sh",
    ".ps1",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    regex: re.Pattern[str]
    risk: str
    message: str
    guard_regexes: tuple[re.Pattern[str], ...]


WINDOWS_GUARD_REGEXES = (
    re.compile(r"platform\s*===?\s*['\"]win32['\"]"),
    re.compile(r"platform\.system\(\)\s*==\s*['\"]Windows['\"]"),
    re.compile(r"os\.name\s*==\s*['\"]nt['\"]"),
    re.compile(r"if\s+\[\s*\"?\$OSTYPE\"?\s*=\s*\"?msys"),
)

POSIX_GUARD_REGEXES = (
    re.compile(r"platform\s*===?\s*['\"](linux|darwin)['\"]"),
    re.compile(r"platform\.system\(\)\s*==\s*['\"](Linux|Darwin)['\"]"),
    re.compile(r"os\.name\s*!=\s*['\"]nt['\"]"),
)

RULES = (
    Rule(
        rule_id="windows_path_literal",
        regex=re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\[^\s\"']*"),
        risk="likely_breaks_posix",
        message="Windows absolute path literal may break macOS/Linux.",
        guard_regexes=WINDOWS_GUARD_REGEXES,
    ),
    Rule(
        rule_id="windows_shell_token",
        regex=re.compile(r"\b(cmd\.exe|powershell|pwsh|where\.exe)\b", re.IGNORECASE),
        risk="likely_breaks_posix",
        message="Windows shell token found without nearby Windows guard.",
        guard_regexes=WINDOWS_GUARD_REGEXES,
    ),
    Rule(
        rule_id="posix_shell_token",
        regex=re.compile(r"(/bin/(bash|sh|zsh)\b|\bchmod\s+\+x\b|\brm\s+-rf\b)", re.IGNORECASE),
        risk="likely_breaks_windows",
        message="POSIX-only shell pattern found without nearby Linux/macOS guard.",
        guard_regexes=POSIX_GUARD_REGEXES,
    ),
    Rule(
        rule_id="unix_home_shortcut",
        regex=re.compile(r"(?<![A-Za-z0-9_])~/(?!/)", re.IGNORECASE),
        risk="likely_breaks_windows",
        message="POSIX home shortcut path may break Windows shells.",
        guard_regexes=POSIX_GUARD_REGEXES,
    ),
)


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {"exclude_globs": [], "suppressions": []}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[xplat-sanity] Warning: failed to read config {config_path}: {exc}")
        return {"exclude_globs": [], "suppressions": []}
    if not isinstance(data, dict):
        return {"exclude_globs": [], "suppressions": []}
    data.setdefault("exclude_globs", [])
    data.setdefault("suppressions", [])
    return data


def _is_guarded(lines: list[str], idx: int, guard_regexes: tuple[re.Pattern[str], ...]) -> bool:
    start = max(0, idx - 3)
    end = min(len(lines), idx + 4)
    window = "\n".join(lines[start:end])
    return any(regex.search(window) for regex in guard_regexes)


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
    except OSError:
        return True
    return b"\x00" in chunk


def _git_file_list(args: list[str]) -> list[Path]:
    try:
        proc = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"[xplat-sanity] Warning: could not run git: {exc}")
        return []
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if stderr:
            print(f"[xplat-sanity] Warning: git command failed: {stderr}")
        return []
    files = []
    for raw in proc.stdout.splitlines():
        rel = raw.strip()
        if not rel:
            continue
        files.append((REPO_ROOT / rel).resolve())
    return files


def _list_candidate_files(staged: bool) -> list[Path]:
    if staged:
        files = _git_file_list(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])
        if files:
            return files
    files = _git_file_list(["git", "ls-files"])
    if files:
        return files
    return [p for p in REPO_ROOT.rglob("*") if p.is_file()]


def _should_skip(path: Path, config: dict) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if path.suffix.lower() not in SCANNABLE_EXTENSIONS:
        return True
    for pattern in config.get("exclude_globs", []):
        if fnmatch.fnmatch(rel, pattern):
            return True
    return _is_binary(path)


def _is_suppressed(path: Path, rule_id: str, config: dict) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    for entry in config.get("suppressions", []):
        if not isinstance(entry, dict):
            continue
        entry_rule = entry.get("rule", "*")
        entry_path = entry.get("path", "*")
        if entry_rule not in ("*", rule_id):
            continue
        if fnmatch.fnmatch(rel, entry_path):
            return True
    return False


def run_check(staged: bool, config_path: Path, strict: bool) -> int:
    config = _load_config(config_path)
    files = [p for p in _list_candidate_files(staged) if p.exists() and p.is_file()]
    warnings: list[tuple[str, int, str, str, str]] = []

    for path in files:
        if _should_skip(path, config):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            for rule in RULES:
                if not rule.regex.search(line):
                    continue
                if _is_guarded(lines, idx, rule.guard_regexes):
                    continue
                if _is_suppressed(path, rule.rule_id, config):
                    continue
                rel = path.relative_to(REPO_ROOT).as_posix()
                warnings.append((rel, idx + 1, rule.rule_id, rule.risk, rule.message))

    if not warnings:
        print("[xplat-sanity] OK: no obvious cross-platform hazards found.")
        return 0

    print("[xplat-sanity] WARN: possible cross-platform hazards detected:")
    for rel, line_no, rule_id, risk, message in warnings:
        print(f"  - {rel}:{line_no} [{risk}] {rule_id}: {message}")

    print(
        "[xplat-sanity] This checker is warn-only by default. "
        "Use --strict to return non-zero when warnings exist."
    )
    if strict:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Warn-only cross-platform sanity checker.")
    parser.add_argument("--staged", action="store_true", help="Scan only staged files.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Config path for excludes/suppressions.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when warnings are present.",
    )
    args = parser.parse_args()
    return run_check(
        staged=args.staged,
        config_path=Path(args.config).resolve(),
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
