"""Shared helpers for stage UI E2E tests."""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict

import pytest
import requests

from e2e_sim import LMStudioSimServer
from runtime_layout import RuntimeLayout
from pathlib import Path


playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright
PlaywrightTimeoutError = playwright_sync.TimeoutError


APP_DIR = Path(__file__).resolve().parents[2]
SOURCE_LAYOUT = RuntimeLayout.from_app_dir(str(APP_DIR))
SOURCE_REPO_DIR = SOURCE_LAYOUT.repo_root
SOURCE_APP_DIR = SOURCE_LAYOUT.app_dir
WATCHDOG_IDLE_SECONDS = 10.0
SCRIPT_LEAK_CHECK_SECONDS = 5.0
WATCHDOG_POLL_SECONDS = 0.35
WATCHDOG_MAX_SECONDS = 120.0
FRESH_CLONE_BOOTSTRAP_TIMEOUT_SECONDS = 1800.0
MODEL_DOWNLOAD_DISABLE_ENV = "THREADSPEAK_DISABLE_MODEL_DOWNLOADS"
MODEL_DOWNLOAD_FORBIDDEN_PATTERNS = (
    "model not cached locally, downloading",
    "attempting auto-download",
    "built-in adapter downloaded:",
    "downloaded builtin_",
    "downloaded qwen/",
    "downloaded qwen3",
)


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _report_directory(prefix: str):
    if _env_true("THREADSPEAK_E2E_KEEP_REPORTS"):
        report_root = tempfile.mkdtemp(prefix=prefix)
        print(f"[e2e-debug] keeping report directory: {report_root}", flush=True)
        yield report_root
        return
    with tempfile.TemporaryDirectory(prefix=prefix) as report_root:
        yield report_root


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_command(command: list[str], *, cwd: str, env: dict | None = None, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise AssertionError(
            f"Command failed with exit code {completed.returncode}: {rendered}\n"
            f"{(completed.stdout or '')[-4000:]}"
        )
    return completed


def _assert_no_model_download_attempts(text: str, *, context: str) -> None:
    lowered = str(text or "").lower()
    matched = [pattern for pattern in MODEL_DOWNLOAD_FORBIDDEN_PATTERNS if pattern in lowered]
    if matched:
        raise AssertionError(
            f"Detected forbidden model download attempt during {context}.\n"
            f"Matched patterns: {matched}\n"
            f"Recent output:\n{str(text or '')[-4000:]}"
        )


def _resolve_clone_source_commit(source_repo_dir: str, source_ref: str = "HEAD") -> str:
    completed = _run_command(
        ["git", "-C", source_repo_dir, "rev-parse", source_ref],
        cwd=source_repo_dir,
    )
    return str(completed.stdout or "").strip()


def _clone_repo_git_ref(source_repo_dir: str, clone_root: str, *, source_ref: str = "HEAD") -> str:
    source_repo_dir = os.path.abspath(source_repo_dir)
    clone_root = os.path.abspath(clone_root)
    os.makedirs(clone_root, exist_ok=True)
    expected_commit = _resolve_clone_source_commit(source_repo_dir, source_ref)
    _run_command(
        ["git", "clone", "--local", "--shared", "--no-checkout", source_repo_dir, clone_root],
        cwd=os.path.dirname(clone_root),
    )
    _run_command(
        ["git", "-C", clone_root, "checkout", "--detach", expected_commit],
        cwd=clone_root,
    )
    actual_commit = str(
        _run_command(["git", "-C", clone_root, "rev-parse", "HEAD"], cwd=clone_root).stdout or ""
    ).strip()
    if actual_commit != expected_commit:
        raise AssertionError(
            f"Fresh clone did not resolve to git ref {source_ref}.\n"
            f"Expected: {expected_commit}\n"
            f"Actual:   {actual_commit}"
        )
    return actual_commit


def _fresh_clone_install_commands(python_executable: str, *, host_platform: str | None = None, host_arch: str | None = None) -> list[list[str]]:
    current_platform = str(host_platform or sys.platform).lower()
    current_arch = str(host_arch or platform.machine()).lower()
    commands: list[list[str]] = [
        [python_executable, "-m", "pip", "uninstall", "-y", "google-genai"],
        [python_executable, "-m", "pip", "install", "--upgrade", "pip"],
        [python_executable, "-m", "pip", "install", "-r", "requirements.txt"],
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "fastapi",
            "uvicorn",
            "pydantic",
            "openai",
            "python-docx",
            "pytest",
            "numpy",
            "pydub",
            "soundfile",
            "librosa",
            "requests",
            "aiofiles",
            "python-multipart",
        ],
    ]
    if current_platform == "darwin" and current_arch == "arm64":
        commands.extend([
            [python_executable, "-m", "pip", "uninstall", "-y", "qwen-tts"],
            [python_executable, "-m", "pip", "install", "mlx-audio==0.4.2", "sentencepiece", "tiktoken"],
        ])
    else:
        commands.append([python_executable, "-m", "pip", "install", "qwen-tts==0.1.1"])
    commands.append([
        python_executable,
        "-c",
        "import fastapi, openai, pytest, uvicorn, pydantic, docx, numpy, pydub, soundfile, librosa; print('Dependency check OK')",
    ])
    return commands


def _bootstrap_clone_app_env(
    clone_root: str,
    *,
    timeout_seconds: float = FRESH_CLONE_BOOTSTRAP_TIMEOUT_SECONDS,
) -> str:
    app_dir = os.path.join(clone_root, "app")
    python_bin = os.path.join(app_dir, "env", "bin", "python")
    if os.path.isdir(os.path.join(app_dir, "env")):
        shutil.rmtree(os.path.join(app_dir, "env"), ignore_errors=True)

    source_env_python = os.path.join(SOURCE_APP_DIR, "env", "bin", "python")
    if os.path.exists(source_env_python):
        base_python = str(Path(source_env_python).resolve())
    else:
        base_python = shutil.which("python3") or sys.executable
    _run_command([base_python, "-m", "venv", "env"], cwd=app_dir, timeout=300.0)
    for command in _fresh_clone_install_commands(python_bin):
        completed = _run_command(command, cwd=app_dir, timeout=timeout_seconds)
        _assert_no_model_download_attempts(completed.stdout or "", context="fresh clone bootstrap")
    return python_bin


def _deep_update(base: dict, patch: dict) -> dict:
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object at {path}")
    return data


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _seed_clone_config_values(app_dir: str, values_patch: dict | None) -> None:
    if not values_patch:
        return

    config_path = os.path.join(app_dir, "config.json")
    config_payload = _read_json(config_path) if os.path.exists(config_path) else {}
    _deep_update(config_payload, values_patch)
    _write_json(config_path, config_payload)

    scripts_config_path = os.path.join(app_dir, "scripts", "config.json")
    scripts_payload = _read_json(scripts_config_path) if os.path.exists(scripts_config_path) else {}
    for section, value in (values_patch or {}).items():
        if isinstance(value, dict):
            target = scripts_payload.get(section)
            if not isinstance(target, dict):
                target = {}
                scripts_payload[section] = target
            _deep_update(target, value)
        else:
            scripts_payload[section] = value
    _write_json(scripts_config_path, scripts_payload)


class _IsolatedServer:
    def __init__(self, *, config_patch: dict, env_overrides: dict | None = None):
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.app_dir = ""
        self.layout: RuntimeLayout | None = None
        self.log_path = ""
        self._config_patch = dict(config_patch or {})
        self._env_overrides = dict(env_overrides or {})

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_e2e_stage1_ui_")
        self.app_dir = os.path.join(self._temp_root, "app")
        shutil.copytree(
            SOURCE_APP_DIR,
            self.app_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "env"),
        )

        for filename in (
            "default_prompts.txt",
            "review_prompts.txt",
            "attribution_prompts.txt",
            "voice_prompt.txt",
            "dialogue_identification_system_prompt.txt",
            "temperament_extraction_system_prompt.txt",
        ):
            source = os.path.join(SOURCE_REPO_DIR, "config", "prompts", filename)
            if not os.path.exists(source):
                continue
            prompt_dir = os.path.join(self._temp_root, "config", "prompts")
            os.makedirs(prompt_dir, exist_ok=True)
            shutil.copy2(source, os.path.join(prompt_dir, filename))

        self._seed_config()
        self.log_path = os.path.join(self._temp_root, "isolated-server.log")

        port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{port}"

        env = os.environ.copy()
        env["PINOKIO_SHARE_LOCAL"] = "false"
        env["PINOKIO_SHARE_LOCAL_PORT"] = str(port)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self._env_overrides)

        log_handle = open(self.log_path, "w", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=self.app_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            log_handle.close()

        deadline = time.time() + 60
        while time.time() < deadline:
            if self._proc.poll() is not None:
                output = _tail_file(self.log_path)
                raise AssertionError(
                    f"Isolated server exited early with code {self._proc.returncode}.\n{output[-3000:]}"
                )
            try:
                response = requests.get(f"{self.base_url}/", timeout=1.5)
                if response.status_code < 500:
                    self.layout = RuntimeLayout.from_app_dir(self.app_dir)
                    return self
            except Exception:
                pass
            time.sleep(0.3)

        raise AssertionError(f"Timed out waiting for isolated server at {self.base_url}")

    def _seed_config(self) -> None:
        config_path = os.path.join(self.app_dir, "config.json")
        config_payload = _read_json(config_path) if os.path.exists(config_path) else {}
        # Always start isolated runs from an unlocked/no-file baseline.
        config_payload["current_file"] = None
        config_payload["generation_mode_locked"] = False
        _deep_update(config_payload, self._config_patch)
        _write_json(config_path, config_payload)

        scripts_config_path = os.path.join(self.app_dir, "scripts", "config.json")
        scripts_payload = _read_json(scripts_config_path) if os.path.exists(scripts_config_path) else {}
        for section in ("llm", "generation", "tts"):
            value = self._config_patch.get(section)
            if isinstance(value, dict):
                target = scripts_payload.get(section)
                if not isinstance(target, dict):
                    target = {}
                    scripts_payload[section] = target
                _deep_update(target, value)
        if isinstance(config_payload.get("prompts"), dict):
            scripts_payload["prompts"] = config_payload["prompts"]
        _write_json(scripts_config_path, scripts_payload)

    def __exit__(self, exc_type, exc, tb):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._temp_root and os.path.isdir(self._temp_root) and not _env_true("THREADSPEAK_E2E_KEEP_ISOLATED_ROOT"):
            shutil.rmtree(self._temp_root, ignore_errors=True)


class _FreshCloneServer:
    def __init__(
        self,
        *,
        env_overrides: dict | None = None,
        bootstrap_config_values: dict | None = None,
        bootstrap_timeout_seconds: float = FRESH_CLONE_BOOTSTRAP_TIMEOUT_SECONDS,
    ):
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.repo_root = ""
        self.app_dir = ""
        self.layout: RuntimeLayout | None = None
        self.python_path = ""
        self.checked_out_commit = ""
        self.log_path = ""
        self._env_overrides = dict(env_overrides or {})
        self._bootstrap_config_values = dict(bootstrap_config_values or {})
        self._bootstrap_timeout_seconds = float(bootstrap_timeout_seconds)

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_e2e_fresh_clone_")
        self.repo_root = os.path.join(self._temp_root, "repo")
        self.checked_out_commit = _clone_repo_git_ref(
            SOURCE_REPO_DIR,
            self.repo_root,
            source_ref="refs/remotes/origin/main",
        )
        self.app_dir = os.path.join(self.repo_root, "app")
        _seed_clone_config_values(self.app_dir, self._bootstrap_config_values)
        self.python_path = _bootstrap_clone_app_env(
            self.repo_root,
            timeout_seconds=self._bootstrap_timeout_seconds,
        )
        self.log_path = os.path.join(self.repo_root, "fresh-clone-server.log")

        port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{port}"

        env = os.environ.copy()
        env["PINOKIO_SHARE_LOCAL"] = "false"
        env["PINOKIO_SHARE_LOCAL_PORT"] = str(port)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault(MODEL_DOWNLOAD_DISABLE_ENV, "1")
        env.update(self._env_overrides)

        log_handle = open(self.log_path, "w", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                [self.python_path, "app.py"],
                cwd=self.app_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            log_handle.close()

        deadline = time.time() + 120
        while time.time() < deadline:
            if self._proc.poll() is not None:
                output = _tail_file(self.log_path)
                _assert_no_model_download_attempts(output, context="fresh clone server startup")
                raise AssertionError(
                    f"Fresh clone server exited early with code {self._proc.returncode}.\n{output[-4000:]}"
                )
            try:
                response = requests.get(f"{self.base_url}/", timeout=1.5)
                if response.status_code < 500:
                    self.layout = RuntimeLayout.from_app_dir(self.app_dir)
                    return self
            except Exception:
                pass
            time.sleep(0.3)

        raise AssertionError(f"Timed out waiting for fresh clone server at {self.base_url}")

    def __exit__(self, exc_type, exc, tb):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self.log_path and os.path.exists(self.log_path):
            _assert_no_model_download_attempts(_tail_file(self.log_path, max_chars=12000), context="fresh clone server runtime")
        if self._temp_root and os.path.isdir(self._temp_root) and not _env_true("THREADSPEAK_E2E_KEEP_ISOLATED_ROOT"):
            shutil.rmtree(self._temp_root, ignore_errors=True)


def _report_console(console_errors: list[str], page_errors: list[str], warnings: list[str]) -> str:
    lines = ["Browser diagnostics:"]
    if console_errors:
        lines.append("console.error:")
        lines.extend(f"  - {line}" for line in console_errors)
    if page_errors:
        lines.append("pageerror:")
        lines.extend(f"  - {line}" for line in page_errors)
    if warnings:
        lines.append("console.warn:")
        lines.extend(f"  - {line}" for line in warnings)
    if len(lines) == 1:
        lines.append("  - none")
    return "\n".join(lines)


def _assert_status(response: requests.Response, expected: int, context: str) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={response.text[:1200]}"
        )


def _tail_file(path: str, max_chars: int = 4000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = handle.read()
    except FileNotFoundError:
        return "(missing)"
    except Exception as exc:
        return f"(unreadable: {exc})"
    if not data:
        return "(empty)"
    return data[-max_chars:]


def _extract_logs(payload: dict) -> list[str]:
    return [str(item) for item in (payload.get("logs") or [])]


def _collect_fatal_log_lines(logs: list[str]) -> list[str]:
    fatal_patterns = (
        "failed with return code",
        "failed to call llm provider",
        "api call failed",
        "unexpected interaction",
        "no scripted entries remain",
        "traceback (most recent call last)",
    )
    fatal = []
    for line in logs:
        text = str(line or "").strip()
        lower = text.lower()
        if text.startswith("ERROR:") or text.startswith("Error:"):
            fatal.append(text)
            continue
        if any(pattern in lower for pattern in fatal_patterns):
            fatal.append(text)
    return fatal


def _wait_for_activity(
    description: str,
    probe_fn,
    done_fn,
    *,
    inactivity_seconds: float = WATCHDOG_IDLE_SECONDS,
    poll_seconds: float = WATCHDOG_POLL_SECONDS,
    max_total_seconds: float = WATCHDOG_MAX_SECONDS,
):
    last_snapshot = None
    start_at = time.time()
    last_progress_at = time.time()
    while True:
        snapshot = probe_fn()
        snapshot_key = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, default=str)
        if snapshot_key != last_snapshot:
            last_snapshot = snapshot_key
            last_progress_at = time.time()
            if os.environ.get("THREADSPEAK_E2E_DEBUG_WATCHDOG", "").strip() == "1":
                compact = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
                print(f"[watchdog] {description}: {compact[:800]}", flush=True)
        if done_fn(snapshot):
            return snapshot
        elapsed = time.time() - start_at
        if elapsed > max_total_seconds:
            raise AssertionError(
                f"{description} exceeded max runtime ({int(max_total_seconds)}s).\n"
                f"Last observed state:\n{json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)}"
            )
        if time.time() - last_progress_at > inactivity_seconds:
            raise AssertionError(
                f"{description} stalled with no new output for >{int(inactivity_seconds)}s.\n"
                f"Last observed state:\n{json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)}"
            )
        time.sleep(poll_seconds)


def _fetch_task_status(base_url: str, task_name: str) -> dict:
    response = requests.get(f"{base_url}/api/status/{task_name}", timeout=10)
    _assert_status(response, 200, f"poll {task_name}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"/api/status/{task_name} returned non-object payload: {type(payload).__name__}")
    return payload


def _wait_for_bootstrap_ready(page) -> None:
    def probe():
        return page.evaluate(
            """() => {
                const hasScriptNav = !!document.querySelector('.nav-link[data-tab="script"]');
                const hasSetupNav = !!document.querySelector('.nav-link[data-tab="setup"]');
                const hasFileUpload = !!document.querySelector('#file-upload');
                const hasProcessBtn = !!document.querySelector('#btn-process-script-v2');
                return {
                    done: !!window.__THREADSPEAK_BOOTSTRAP_DONE,
                    error: window.__THREADSPEAK_BOOTSTRAP_ERROR ? String(window.__THREADSPEAK_BOOTSTRAP_ERROR) : '',
                    step: window.__THREADSPEAK_BOOTSTRAP_STEP ? String(window.__THREADSPEAK_BOOTSTRAP_STEP) : '',
                    last_activity: Number(window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY || 0),
                    has_script_nav: hasScriptNav,
                    has_setup_nav: hasSetupNav,
                    has_file_upload: hasFileUpload,
                    has_process_btn: hasProcessBtn,
                };
            }"""
        )

    def done(snapshot):
        if snapshot.get("error"):
            raise AssertionError(f"UI bootstrap failed: {snapshot['error']}")
        return bool(
            snapshot.get("done")
            and snapshot.get("has_script_nav")
            and snapshot.get("has_setup_nav")
            and snapshot.get("has_file_upload")
            and snapshot.get("has_process_btn")
        )

    _wait_for_activity("Waiting for app bootstrap", probe, done)


def _wait_for_script_tab_ready(page) -> None:
    page.locator('.nav-link[data-tab="script"]').click()

    def probe():
        return page.evaluate(
            """() => {
                const tab = document.querySelector('#script-tab');
                const controls = document.querySelector('#new-mode-controls');
                const processBtn = document.querySelector('#btn-process-script-v2');
                const resetBtn = document.querySelector('#btn-reset-project-script-v2');
                const processVoices = document.querySelector('#process-voices-toggle-v2');
                const fileUpload = document.querySelector('#file-upload');
                const visible = (el) => {
                    if (!el) return false;
                    const style = getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                };
                return {
                    tab_visible: visible(tab),
                    controls_visible: visible(controls),
                    has_process_btn: !!processBtn,
                    has_reset_btn: !!resetBtn,
                    has_process_voices_toggle: !!processVoices,
                    has_file_upload: !!fileUpload,
                    script_logs_len: String(document.querySelector('#script-logs')?.innerText || '').trim().length,
                };
            }"""
        )

    def done(snapshot):
        return bool(
            snapshot.get("tab_visible")
            and snapshot.get("controls_visible")
            and snapshot.get("has_process_btn")
            and snapshot.get("has_reset_btn")
            and snapshot.get("has_process_voices_toggle")
            and snapshot.get("has_file_upload")
        )

    _wait_for_activity("Waiting for Script tab UI to become ready", probe, done)


def _wait_for_upload_loaded(page) -> None:
    def probe():
        return page.evaluate(
            """() => {
                const statusEl = document.querySelector('#upload-status');
                const text = String(statusEl?.innerText || '').trim();
                const fileUploadSection = document.querySelector('#file-upload-section');
                const uploadHidden = !!fileUploadSection && getComputedStyle(fileUploadSection).display === 'none';
                return { status_text: text, upload_hidden: uploadHidden };
            }"""
        )

    def done(snapshot):
        text = snapshot.get("status_text") or ""
        if "Failed to load file" in text:
            raise AssertionError(f"Upload failed: {text}")
        return "Loaded:" in text and bool(snapshot.get("upload_hidden"))

    _wait_for_activity("Waiting for upload status output", probe, done)


def _maybe_reset_project_from_script_tab(page) -> bool:
    deadline = time.time() + SCRIPT_LEAK_CHECK_SECONDS
    leak_snapshot = None
    while time.time() < deadline:
        leak_snapshot = page.evaluate(
            """() => {
                const uploadText = String(document.querySelector('#upload-status')?.innerText || '').trim();
                const scriptLogs = String(document.querySelector('#script-logs')?.innerText || '').trim();
                const uploadSection = document.querySelector('#file-upload-section');
                const uploadHidden = !!uploadSection && getComputedStyle(uploadSection).display === 'none';
                const voicesNav = document.querySelector('.nav-link[data-tab="voices"]');
                const voicesUnlocked = !!voicesNav && !voicesNav.classList.contains('nav-locked');
                const editorNav = document.querySelector('.nav-link[data-tab="editor"]');
                const editorUnlocked = !!editorNav && !editorNav.classList.contains('nav-locked');
                const resetBtn = document.querySelector('#btn-reset-project-script-v2');
                const resetVisible = !!resetBtn && getComputedStyle(resetBtn).display !== 'none';
                const leaked = uploadText.includes('Loaded:') || uploadHidden || scriptLogs.length > 0 || voicesUnlocked || editorUnlocked;
                return {
                    leaked,
                    upload_text: uploadText,
                    logs_len: scriptLogs.length,
                    voices_unlocked: voicesUnlocked,
                    editor_unlocked: editorUnlocked,
                    reset_visible: resetVisible,
                };
            }"""
        )
        if leak_snapshot.get("leaked"):
            break
        time.sleep(0.25)

    if not leak_snapshot or not leak_snapshot.get("leaked"):
        return False

    reset_btn = page.locator("#btn-reset-project-script-v2")
    if not reset_btn.is_visible():
        raise AssertionError(
            "Detected leaked project state on Script tab, but Reset Project button is not available.\n"
            f"Leak snapshot: {json.dumps(leak_snapshot, ensure_ascii=False)}"
        )

    reset_btn.click()
    confirm_ok = page.locator("#confirmModalOk")
    confirm_ok.wait_for(state="visible", timeout=5000)
    confirm_ok.click()

    page.wait_for_load_state("domcontentloaded", timeout=10000)
    _wait_for_bootstrap_ready(page)
    _wait_for_script_tab_ready(page)
    return True


def _wait_for_new_mode_script_completion(base_url: str, expected_log: str) -> dict:
    def probe():
        payload = _fetch_task_status(base_url, "new_mode_workflow")
        logs = _extract_logs(payload)
        errors = _collect_fatal_log_lines(logs)
        return {
            "running": bool(payload.get("running")),
            "paused": bool(payload.get("paused")),
            "current_stage": payload.get("current_stage"),
            "last_error": payload.get("last_error"),
            "completed_stages": list(payload.get("completed_stages") or []),
            "logs_count": len(logs),
            "last_log": logs[-1] if logs else "",
            "has_expected_log": any(expected_log in line for line in logs),
            "errors": errors,
            "logs_tail": logs[-15:],
        }

    def done(snapshot):
        if snapshot.get("errors"):
            raise AssertionError(f"Script workflow emitted errors: {snapshot['errors']}")
        if snapshot.get("last_error"):
            raise AssertionError(f"Script workflow failed: {snapshot['last_error']}")
        completed = set(str(item) for item in (snapshot.get("completed_stages") or []))
        return (
            not snapshot.get("running")
            and not snapshot.get("paused")
            and "create_script" in completed
            and bool(snapshot.get("has_expected_log"))
        )

    return _wait_for_activity("Waiting for script workflow output", probe, done, poll_seconds=0.8)


def _wait_for_nav_unlocked(page, nav_selector: str, label: str) -> None:
    def probe():
        return page.evaluate(
            """(selector) => {
                const nav = document.querySelector(selector);
                const logs = String(document.querySelector('#script-logs')?.innerText || '');
                return {
                    exists: !!nav,
                    locked: !!nav && nav.classList.contains('nav-locked'),
                    class_name: nav ? String(nav.className || '') : '',
                    logs_len: logs.length,
                    logs_tail: logs.slice(Math.max(0, logs.length - 500)),
                };
            }""",
            nav_selector,
        )

    def done(snapshot):
        return bool(snapshot.get("exists")) and not bool(snapshot.get("locked"))

    _wait_for_activity(f"Waiting for {label} navigation unlock", probe, done)


def _wait_for_nav_locked(page, nav_selector: str, label: str) -> None:
    def probe():
        return page.evaluate(
            """(selector) => {
                const nav = document.querySelector(selector);
                return {
                    exists: !!nav,
                    locked: !!nav && nav.classList.contains('nav-locked'),
                    class_name: nav ? String(nav.className || '') : '',
                };
            }""",
            nav_selector,
        )

    def done(snapshot):
        return bool(snapshot.get("exists")) and bool(snapshot.get("locked"))

    _wait_for_activity(f"Waiting for {label} navigation lock", probe, done)


def _read_script_step_states(page) -> Dict[str, str]:
    payload = page.evaluate(
        """() => {
            const mapping = {
                process_paragraphs: 'icon-process-paragraphs',
                assign_dialogue: 'icon-assign-dialogue',
                extract_temperament: 'icon-extract-temperament',
                create_script: 'icon-create-script',
            };
            const states = {};
            for (const [key, id] of Object.entries(mapping)) {
                const el = document.getElementById(id);
                states[key] = el ? String(el.dataset.stepState || '').trim() : '';
            }
            return states;
        }"""
    )
    return dict(payload or {})


def _wait_for_script_step_states(page, expected_states: Dict[str, str]) -> Dict[str, str]:
    def probe():
        return _read_script_step_states(page)

    def done(snapshot):
        return all(str(snapshot.get(key) or "") == str(value) for key, value in expected_states.items())

    return _wait_for_activity("Waiting for Script step icon states", probe, done)


def _reset_project_from_script_tab(page) -> None:
    page.locator('.nav-link[data-tab="script"]').click()
    _wait_for_script_tab_ready(page)

    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/reset_project")
            and response.request.method == "POST"
        ),
        timeout=10000,
    ) as response_info:
        page.locator("#btn-reset-project-script-v2").click()
        confirmed = _confirm_modal_if_present(page, timeout_ms=5000)
        assert confirmed, "Reset Project confirmation modal did not appear."

    response = response_info.value
    if int(response.status) != 200:
        response_text = ""
        try:
            response_text = response.text()
        except Exception:
            response_text = ""
        raise AssertionError(
            f"Reset Project request failed with status {response.status}.\n"
            f"Response body:\n{response_text[:2000]}"
        )

    page.wait_for_load_state("domcontentloaded", timeout=10000)
    _wait_for_bootstrap_ready(page)
    _wait_for_script_tab_ready(page)


def _iter_runtime_files(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    rel_paths: list[str] = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(current_root, filename)
            rel_paths.append(os.path.relpath(full_path, root_dir))
    return sorted(rel_paths)


def _assert_runtime_audio_artifacts_removed(layout: RuntimeLayout) -> None:
    missing_targets = [
        layout.audiobook_path,
        layout.optimized_export_path,
        layout.audacity_export_path,
        layout.m4b_path,
        layout.m4b_cover_path,
    ]
    lingering_targets = [path for path in missing_targets if os.path.exists(path)]
    if lingering_targets:
        raise AssertionError(
            "Reset Project left generated export artifacts behind.\n"
            f"Lingering targets: {json.dumps(lingering_targets, ensure_ascii=False, indent=2)}"
        )

    exports_files = _iter_runtime_files(layout.exports_dir)
    voicelines_files = _iter_runtime_files(layout.voicelines_dir)
    uploads_files = _iter_runtime_files(layout.uploads_dir)
    if exports_files or voicelines_files or uploads_files:
        raise AssertionError(
            "Reset Project left generated runtime files behind.\n"
            f"exports/: {json.dumps(exports_files, ensure_ascii=False, indent=2)}\n"
            f"voicelines/: {json.dumps(voicelines_files, ensure_ascii=False, indent=2)}\n"
            f"uploads/: {json.dumps(uploads_files, ensure_ascii=False, indent=2)}"
        )


def _switch_llm_via_setup_ui(
    page,
    *,
    llm_base_url: str,
    llm_model_name: str,
    tts_mode: str | None = None,
    tts_parallel_workers: int | None = None,
) -> None:
    page.locator('.nav-link[data-tab="setup"]').click()

    def probe_setup():
        return page.evaluate(
            """() => {
                const toggle = document.querySelector('#legacy-mode-toggle');
                const llmUrl = document.querySelector('#llm-url');
                const llmModel = document.querySelector('#llm-model');
                const llmWorkers = document.querySelector('#llm-workers');
                const ttsMode = document.querySelector('#tts-mode');
                const parallelWorkers = document.querySelector('#parallel-workers');
                return {
                    has_toggle: !!toggle,
                    legacy_checked: !!toggle && !!toggle.checked,
                    has_llm_url: !!llmUrl,
                    has_llm_model: !!llmModel,
                    has_llm_workers: !!llmWorkers,
                    has_tts_mode: !!ttsMode,
                    has_parallel_workers: !!parallelWorkers,
                };
            }"""
        )

    def setup_ready(snapshot):
        return bool(
            snapshot.get("has_toggle")
            and snapshot.get("has_llm_url")
            and snapshot.get("has_llm_model")
            and snapshot.get("has_tts_mode")
            and snapshot.get("has_parallel_workers")
        )

    setup_snapshot = _wait_for_activity("Waiting for Setup tab UI", probe_setup, setup_ready)
    assert not setup_snapshot.get("legacy_checked"), "Expected non-legacy mode to be enabled."

    llm_url = page.locator("#llm-url")
    llm_model = page.locator("#llm-model")
    llm_workers = page.locator("#llm-workers")
    tts_mode_locator = page.locator("#tts-mode")
    parallel_workers_locator = page.locator("#parallel-workers")

    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/config/setup")
            and response.request.method == "POST"
            and response.status == 200
        ),
        timeout=10000,
    ):
        llm_url.fill(llm_base_url)
        llm_url.blur()
        llm_model.fill(llm_model_name)
        llm_model.blur()
        llm_workers.fill("1")
        llm_workers.blur()
        if tts_mode is not None:
            tts_mode_locator.select_option(tts_mode)
        if tts_parallel_workers is not None:
            parallel_workers_locator.fill(str(int(tts_parallel_workers)))
            parallel_workers_locator.blur()


def _wait_for_voice_generation_completion(page, expected_speakers: set[str]) -> dict:
    sorted_speakers = sorted(expected_speakers)

    def probe():
        return page.evaluate(
            """(speakers) => {
                const bulkBtn = document.querySelector('#generate-outstanding-voices-btn');
                const ready = [];
                const missing = [];
                const present = [];
                const states = {};
                for (const speaker of speakers) {
                    const card = document.querySelector(`.voice-card[data-voice="${speaker}"]`);
                    if (!card) {
                        missing.push(speaker);
                        continue;
                    }
                    present.push(speaker);
                    const btn = card.querySelector('.design-generate-btn');
                    const ref = String(card.querySelector('.design-ref-audio')?.value || '').trim();
                    const text = String(btn?.textContent || '').trim();
                    const retry = !!btn && text === 'Retry' && btn.classList.contains('btn-warning');
                    states[speaker] = {
                        has_card: true,
                        card_class: String(card.className || ''),
                        ref_audio: ref,
                        btn_text: text,
                        btn_class: String(btn?.className || ''),
                        btn_disabled: !!btn?.disabled,
                        retry,
                    };
                    if (retry && ref) {
                        ready.push(speaker);
                    }
                }
                return {
                    bulk_disabled: !!bulkBtn && !!bulkBtn.disabled,
                    bulk_text: String(bulkBtn?.textContent || '').trim(),
                    ready_speakers: ready,
                    present_speakers: present,
                    missing_speakers: missing,
                    speaker_states: states,
                };
            }""",
            sorted_speakers,
        )

    def done(snapshot):
        ready = set(str(item) for item in (snapshot.get("ready_speakers") or []))
        return ready == expected_speakers and not bool(snapshot.get("bulk_disabled"))

    return _wait_for_activity("Waiting for voice generation output", probe, done)


def _set_narrator_threshold_and_wait(page, *, value: int, expected_speakers: set[str]) -> None:
    threshold_input = page.locator("#narrator-threshold-input")
    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/voices/settings")
            and response.request.method == "POST"
            and response.status == 200
        ),
        timeout=10000,
    ):
        threshold_input.fill(str(int(value)))
        threshold_input.blur()

    sorted_speakers = sorted(expected_speakers)
    _wait_for_activity(
        "Waiting for narrator-threshold refresh",
        lambda: page.evaluate(
            """([speakers, expectedValue]) => {
                const states = {};
                for (const speaker of speakers) {
                    const card = document.querySelector(`.voice-card[data-voice="${speaker}"]`);
                    if (!card) {
                        states[speaker] = { has_card: false, narrator_threshold_active: false, alias_active: false };
                        continue;
                    }
                    states[speaker] = {
                        has_card: true,
                        narrator_threshold_active: card.classList.contains('narrator-threshold-active'),
                        alias_active: card.classList.contains('alias-active'),
                        card_class: String(card.className || ''),
                    };
                }
                const input = document.querySelector('#narrator-threshold-input');
                const raw = String(input?.value || '').trim();
                const parsed = Number.parseInt(raw, 10);
                return {
                    states,
                    narrator_threshold_value: Number.isFinite(parsed) ? parsed : null,
                    narrator_threshold_expected: Number(expectedValue),
                };
            }""",
            [sorted_speakers, int(value)],
        ),
        lambda snapshot: all(
            bool((snapshot.get("states") or {}).get(speaker, {}).get("has_card"))
            for speaker in sorted_speakers
        )
        and snapshot.get("narrator_threshold_value") is not None
        and int(snapshot.get("narrator_threshold_value")) == int(value),
    )


def _read_voice_card_states(page) -> Dict[str, Dict[str, Any]]:
    payload = page.evaluate(
        """() => {
            const states = {};
            const cards = Array.from(document.querySelectorAll('.voice-card'));
            for (const card of cards) {
                const speaker = String(card.getAttribute('data-voice') || '').trim();
                if (!speaker) continue;
                const btn = card.querySelector('.design-generate-btn');
                const refAudio = String(card.querySelector('.design-ref-audio')?.value || '').trim();
                const btnText = String(btn?.textContent || '').trim();
                states[speaker] = {
                    narrator_threshold_active: card.classList.contains('narrator-threshold-active'),
                    alias_active: card.classList.contains('alias-active'),
                    card_class: String(card.className || ''),
                    btn_text: btnText,
                    btn_class: String(btn?.className || ''),
                    btn_disabled: !!btn?.disabled,
                    retry: !!btn && btnText === 'Retry' && btn.classList.contains('btn-warning'),
                    ref_audio: refAudio,
                };
            }
            return states;
        }"""
    )
    return dict(payload or {})


def _wait_for_preview_playback(page, *, speaker: str, ref_audio: str) -> None:
    def probe():
        return page.evaluate(
            """([speakerName, refAudio]) => {
                const card = document.querySelector(`.voice-card[data-voice="${speakerName}"]`);
                const audio = window._sharedPreviewAudio;
                const src = String(audio?.currentSrc || audio?.src || '');
                return {
                    card_found: !!card,
                    has_shared_audio: !!audio,
                    src,
                    src_matches: src.includes(refAudio),
                    current_time: Number(audio?.currentTime || 0),
                    paused: !!audio?.paused,
                    ended: !!audio?.ended,
                    ready_state: Number(audio?.readyState || 0),
                };
            }""",
            [speaker, ref_audio],
        )

    def done(snapshot):
        return bool(
            snapshot.get("card_found")
            and snapshot.get("has_shared_audio")
            and snapshot.get("src_matches")
            and (
                float(snapshot.get("current_time") or 0.0) > 0.0
                or not snapshot.get("paused")
                or snapshot.get("ended")
                or int(snapshot.get("ready_state") or 0) >= 2
            )
        )

    _wait_for_activity(f"Waiting for preview playback for {speaker}", probe, done)


def _wait_for_editor_audio_completion(base_url: str, *, baseline_job_id: int = 0) -> dict:
    def probe():
        payload = _fetch_task_status(base_url, "audio")
        metrics = payload.get("metrics") or {}
        logs = _extract_logs(payload)
        errors = _collect_fatal_log_lines(logs)
        recent_jobs = list(payload.get("recent_jobs") or [])
        latest_job = dict(recent_jobs[0] or {}) if recent_jobs else {}
        return {
            "running": bool(payload.get("running")),
            "queue_count": len(payload.get("queue") or []),
            "has_current_job": bool(payload.get("current_job")),
            "remaining_words": int(metrics.get("remaining_words") or 0),
            "error_clips": int(metrics.get("error_clips") or 0),
            "latest_job_id": int(latest_job.get("id") or 0),
            "latest_job_status": str(latest_job.get("status") or "").strip().lower(),
            "latest_processed_clips": int(latest_job.get("processed_clips") or 0),
            "latest_generation_finished": bool(latest_job.get("generation_finished")),
            "logs_count": len(logs),
            "last_log": logs[-1] if logs else "",
            "errors": errors,
            "logs_tail": logs[-15:],
        }

    def done(snapshot):
        if snapshot.get("errors"):
            raise AssertionError(f"Audio render emitted fatal errors: {snapshot['errors']}")
        return (
            not snapshot.get("running")
            and int(snapshot.get("queue_count") or 0) == 0
            and not snapshot.get("has_current_job")
            and int(snapshot.get("remaining_words") or 0) == 0
            and int(snapshot.get("error_clips") or 0) == 0
            and int(snapshot.get("latest_job_id") or 0) > int(baseline_job_id or 0)
            and str(snapshot.get("latest_job_status") or "") == "completed"
            and int(snapshot.get("latest_processed_clips") or 0) > 0
            and bool(snapshot.get("latest_generation_finished"))
        )

    return _wait_for_activity("Waiting for editor audio render output", probe, done, poll_seconds=0.8)


def _wait_for_proofread_completion(base_url: str) -> dict:
    def probe():
        payload = _fetch_task_status(base_url, "proofread")
        logs = _extract_logs(payload)
        errors = _collect_fatal_log_lines(logs)
        progress = dict(payload.get("progress") or {})
        return {
            "running": bool(payload.get("running")),
            "phase": str(progress.get("phase") or "").strip().lower(),
            "processed": int(progress.get("processed") or 0),
            "pending_total": int(progress.get("pending_total") or 0),
            "passed": int(progress.get("passed") or 0),
            "failed": int(progress.get("failed") or 0),
            "auto_failed": int(progress.get("auto_failed") or 0),
            "skipped": int(progress.get("skipped") or 0),
            "logs_count": len(logs),
            "last_log": logs[-1] if logs else "",
            "has_success_log": any("Task proofread completed successfully." in line for line in logs),
            "errors": errors,
            "logs_tail": logs[-20:],
        }

    def done(snapshot):
        if snapshot.get("errors"):
            raise AssertionError(f"Proofread emitted fatal errors: {snapshot['errors']}")
        return (
            not snapshot.get("running")
            and bool(snapshot.get("has_success_log"))
            and str(snapshot.get("phase") or "") == "complete"
        )

    return _wait_for_activity("Waiting for proofread output", probe, done, poll_seconds=0.8)


def _wait_for_audio_merge_completion(base_url: str) -> dict:
    def probe():
        payload = _fetch_task_status(base_url, "audio")
        logs = _extract_logs(payload)
        errors = _collect_fatal_log_lines(logs)
        merge_progress = dict(payload.get("merge_progress") or {})
        return {
            "running": bool(payload.get("running")),
            "merge_running": bool(payload.get("merge_running")),
            "merge_stage": str(merge_progress.get("stage") or "").strip().lower(),
            "merge_progress_running": bool(merge_progress.get("running")),
            "logs_count": len(logs),
            "last_log": logs[-1] if logs else "",
            "has_success_log": any("Merge complete:" in line for line in logs),
            "has_failure_log": any(
                ("Merge failed:" in line) or ("Merge error:" in line)
                for line in logs
            ),
            "errors": errors,
            "logs_tail": logs[-20:],
        }

    def done(snapshot):
        if snapshot.get("errors"):
            raise AssertionError(f"Export merge emitted fatal errors: {snapshot['errors']}")
        if snapshot.get("has_failure_log"):
            raise AssertionError(
                "Export merge failed.\n"
                f"Last observed state:\n{json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)}"
            )
        return (
            not snapshot.get("running")
            and not snapshot.get("merge_running")
            and not snapshot.get("merge_progress_running")
            and bool(snapshot.get("has_success_log"))
        )

    return _wait_for_activity("Waiting for merged export output", probe, done, poll_seconds=0.8)


def _confirm_modal_if_present(page, *, timeout_ms: int = 3000) -> bool:
    confirm_ok = page.locator("#confirmModalOk")
    try:
        confirm_ok.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        return False
    confirm_ok.click()
    return True


def _looks_like_mp3(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            data = handle.read(8192)
    except Exception:
        return False
    if len(data) < 4:
        return False
    if data.startswith(b"ID3"):
        return True
    # Fallback: detect MPEG audio frame sync in the header bytes.
    for i in range(0, len(data) - 1):
        b1 = data[i]
        b2 = data[i + 1]
        if b1 == 0xFF and (b2 & 0xE0) == 0xE0:
            return True
    return False


def _audio_duration_seconds(path: str) -> float:
    from pydub import AudioSegment

    segment = AudioSegment.from_file(path)
    return float(len(segment)) / 1000.0


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_lock_owner_pid(lock_path: str) -> int | None:
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            first_line = (handle.readline() or "").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not first_line.isdigit():
        return None
    return int(first_line)


@contextmanager
def _exclusive_run_lock(lock_name: str):
    lock_dir = os.path.join(tempfile.gettempdir(), "threadspeak-e2e-locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{lock_name}.lock")
    pid = os.getpid()
    created = False

    for _ in range(3):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{pid}\n")
                handle.write(f"{time.time():.6f}\n")
            created = True
            break
        except FileExistsError:
            owner_pid = _read_lock_owner_pid(lock_path)
            if owner_pid and _pid_is_alive(owner_pid):
                raise AssertionError(
                    "Stage-3 editor e2e is already running.\n"
                    f"Existing PID: {owner_pid}\n"
                    f"Lock file: {lock_path}\n"
                    f"Cancel existing run first (example: `kill {owner_pid}`), then retry."
                )
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                continue

    if not created:
        raise AssertionError(f"Could not acquire stage-3 run lock at {lock_path}")

    try:
        yield
    finally:
        try:
            owner_pid = _read_lock_owner_pid(lock_path)
            if owner_pid == pid:
                os.remove(lock_path)
        except FileNotFoundError:
            pass


def _run_stage1_to_voices_tab(
    *,
    page,
    app_base_url: str,
    book_path: str,
    script_llm_base_url: str | None = None,
    script_llm_model_name: str | None = None,
    tts_mode: str | None = None,
    tts_parallel_workers: int | None = None,
) -> None:
    page.goto(app_base_url, wait_until="domcontentloaded", timeout=10000)
    _wait_for_bootstrap_ready(page)
    _wait_for_script_tab_ready(page)
    _maybe_reset_project_from_script_tab(page)

    if script_llm_base_url or script_llm_model_name:
        assert script_llm_base_url, "script_llm_base_url is required when overriding the script-stage LLM."
        assert script_llm_model_name, "script_llm_model_name is required when overriding the script-stage LLM."
        _switch_llm_via_setup_ui(
            page,
            llm_base_url=script_llm_base_url,
            llm_model_name=script_llm_model_name,
            tts_mode=tts_mode,
            tts_parallel_workers=tts_parallel_workers,
        )
        _wait_for_script_tab_ready(page)

    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/upload")
            and response.request.method == "POST"
            and response.status == 200
        ),
        timeout=10000,
    ):
        page.locator("#file-upload").set_input_files(book_path)
    _wait_for_upload_loaded(page)

    page.locator('.nav-link[data-tab="setup"]').click()
    setup_snapshot = _wait_for_activity(
        "Waiting for Setup tab to validate mode",
        lambda: page.evaluate(
            """() => {
                const toggle = document.querySelector('#legacy-mode-toggle');
                const llmUrl = document.querySelector('#llm-url');
                return {
                    has_toggle: !!toggle,
                    legacy_checked: !!toggle && !!toggle.checked,
                    has_llm_url: !!llmUrl,
                };
            }"""
        ),
        lambda snapshot: bool(snapshot.get("has_toggle") and snapshot.get("has_llm_url")),
    )
    assert not setup_snapshot.get("legacy_checked"), "Expected non-legacy mode to be enabled."

    _wait_for_script_tab_ready(page)
    process_voices_toggle = page.locator("#process-voices-toggle-v2")
    if process_voices_toggle.is_checked():
        process_voices_toggle.click()
    assert not process_voices_toggle.is_checked(), "Process Voices must be unchecked for stage 1."

    page.locator("#btn-process-script-v2").click()
    expected_log = "All steps complete. Script is ready in the Editor tab."
    _wait_for_new_mode_script_completion(app_base_url, expected_log)

    _wait_for_nav_unlocked(page, '.nav-link[data-tab="voices"]', "Voices tab")
    page.locator('.nav-link[data-tab="voices"]').click()
    _wait_for_activity(
        "Waiting for Voices tab content",
        lambda: {
            "voices_visible": bool(page.locator("#voices-tab").is_visible()),
            "header": page.locator("#voices-tab .card-header").inner_text().strip()
            if page.locator("#voices-tab .card-header").count()
            else "",
        },
        lambda snapshot: bool(snapshot.get("voices_visible") and snapshot.get("header", "").startswith("Voice Configuration")),
    )


def _run_stage2_to_stage4_proofread_flow(
    *,
    page,
    app_base_url: str,
    voice_server_base_url: str,
    voice_model_name: str,
    expected_speakers: set[str],
) -> dict:
    _switch_llm_via_setup_ui(
        page,
        llm_base_url=voice_server_base_url,
        llm_model_name=voice_model_name,
    )

    page.locator('.nav-link[data-tab="voices"]').click()
    _wait_for_activity(
        "Waiting for Voices tab",
        lambda: {"visible": bool(page.locator("#voices-tab").is_visible())},
        lambda snapshot: bool(snapshot.get("visible")),
    )

    pre_generation_states = _read_voice_card_states(page)
    speaker_list = set(pre_generation_states.keys())
    assert speaker_list == expected_speakers, f"Unexpected speaker rows before generation: {speaker_list}"
    eligible_speakers = {
        speaker
        for speaker, state in pre_generation_states.items()
        if not bool(state.get("alias_active")) and not bool(state.get("narrator_threshold_active"))
    }
    assert eligible_speakers, "No eligible voice cards available for outstanding generation."

    page.locator("#generate-outstanding-voices-btn").click()
    _wait_for_voice_generation_completion(page, eligible_speakers)

    post_generation_states = _read_voice_card_states(page)
    speaker_list = set(post_generation_states.keys())
    assert speaker_list == expected_speakers, f"Unexpected speaker rows after generation: {speaker_list}"

    for speaker in sorted(eligible_speakers):
        state = post_generation_states.get(speaker) or {}
        ref_audio = str(state.get("ref_audio") or "").strip()
        assert ref_audio, f"Missing design ref audio for eligible speaker {speaker}"
        assert bool(state.get("retry")), f"Expected Retry state for eligible speaker {speaker}: {state}"

    playable_speakers = [
        speaker
        for speaker, state in sorted(post_generation_states.items())
        if str(state.get("ref_audio") or "").strip()
    ]
    assert playable_speakers, "No playable voice previews available after outstanding generation."
    for speaker in playable_speakers:
        card_selector = f'.voice-card[data-voice="{speaker}"]'
        ref_audio = page.locator(f"{card_selector} .design-ref-audio").input_value().strip()
        page.locator(f"{card_selector} .design-play-btn").click()
        _wait_for_preview_playback(page, speaker=speaker, ref_audio=ref_audio)

    _wait_for_nav_unlocked(page, '.nav-link[data-tab="editor"]', "Editor tab")
    page.locator('.nav-link[data-tab="editor"]').click()
    _wait_for_activity(
        "Waiting for Editor tab",
        lambda: {"visible": bool(page.locator("#editor-tab").is_visible())},
        lambda snapshot: bool(snapshot.get("visible")),
    )

    page.locator("#editor-chapter-select").select_option("__whole_project__")
    _wait_for_activity(
        "Waiting for Whole Project chunks before render",
        lambda: page.evaluate(
            """() => {
                const rows = Array.from(document.querySelectorAll('#chunks-table-body tr'));
                let textRows = 0;
                for (const row of rows) {
                    const textVal = String(row.querySelector('textarea.chunk-text')?.value || '').trim();
                    if (textVal) textRows += 1;
                }
                return { rows: rows.length, text_rows: textRows };
            }"""
        ),
        lambda snapshot: int(snapshot.get("text_rows") or 0) > 0,
    )

    pre_render_audio = _fetch_task_status(app_base_url, "audio")
    pre_recent_jobs = list((pre_render_audio or {}).get("recent_jobs") or [])
    baseline_job_id = int(((pre_recent_jobs[0] or {}).get("id") or 0)) if pre_recent_jobs else 0

    render_pending_button = page.locator("#btn-batch-fast")
    assert render_pending_button.is_enabled(), "Render Pending button should be enabled before stage-3 run."
    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/generate_batch_fast")
            and response.request.method == "POST"
            and response.status == 200
        ),
        timeout=10000,
    ):
        render_pending_button.click()

    _wait_for_editor_audio_completion(app_base_url, baseline_job_id=baseline_job_id)

    _wait_for_nav_unlocked(page, '.nav-link[data-tab="voices"]', "Voices tab")
    page.locator('.nav-link[data-tab="voices"]').click()
    _wait_for_activity(
        "Waiting for Voices tab reload",
        lambda: {"visible": bool(page.locator("#voices-tab").is_visible())},
        lambda snapshot: bool(snapshot.get("visible")),
    )

    _wait_for_nav_unlocked(page, '.nav-link[data-tab="editor"]', "Editor tab")
    page.locator('.nav-link[data-tab="editor"]').click()
    _wait_for_activity(
        "Waiting for Editor tab reload",
        lambda: {"visible": bool(page.locator("#editor-tab").is_visible())},
        lambda snapshot: bool(snapshot.get("visible")),
    )

    page.locator("#editor-chapter-select").select_option("__whole_project__")
    _wait_for_activity(
        "Waiting for Whole Project rows after reload",
        lambda: {
            "rows": int(
                page.evaluate(
                    "() => document.querySelectorAll('#chunks-table-body tr').length"
                )
            ),
            "text_rows": int(
                page.evaluate(
                    """() => {
                        let n = 0;
                        for (const row of Array.from(document.querySelectorAll('#chunks-table-body tr'))) {
                            const text = String(row.querySelector('textarea.chunk-text')?.value || '').trim();
                            if (text) n += 1;
                        }
                        return n;
                    }"""
                )
            ),
        },
        lambda snapshot: int(snapshot.get("text_rows") or 0) > 0,
    )

    words_text = page.locator("#editor-estimate-words").inner_text().strip()
    words_value = int("".join(ch for ch in words_text if ch.isdigit()) or "0")
    assert words_value == 0, f"Expected remaining words to be 0, got: {words_text}"
    errors_text = page.locator("#editor-estimate-errors").inner_text().strip().lower()
    assert errors_text.startswith("0 clip"), f"Expected 0 errors, got: {errors_text}"

    page.locator("#editor-chapter-select").select_option("__whole_project__")
    _wait_for_activity(
        "Waiting for Whole Project rows",
        lambda: {
            "rows": int(
                page.evaluate(
                    "() => document.querySelectorAll('#chunks-table-body tr').length"
                )
            )
        },
        lambda snapshot: int(snapshot.get("rows") or 0) > 0,
    )

    audio_check = page.evaluate(
        """() => {
            const rows = Array.from(document.querySelectorAll('#chunks-table-body tr'));
            const details = [];
            let textRows = 0;
            for (const row of rows) {
                const text = String(row.querySelector('textarea.chunk-text')?.value || '').trim();
                if (!text) continue;
                textRows += 1;
                const audio = row.querySelector('audio.chunk-audio');
                const audioPath = String(audio?.getAttribute('data-audio-path') || '').trim();
                const done = row.classList.contains('status-done');
                if (!audioPath || !done) {
                    details.push({
                        id: String(row.getAttribute('data-id') || ''),
                        has_audio: Boolean(audioPath),
                        status_done: done,
                    });
                }
            }
            return { text_rows: textRows, missing: details };
        }"""
    )
    assert int(audio_check.get("text_rows") or 0) > 0, "Expected at least one text clip row in Whole Project view."
    assert not audio_check.get("missing"), f"Rows missing audio or done status: {audio_check.get('missing')}"

    _wait_for_nav_unlocked(page, '.nav-link[data-tab="proofread"]', "Proofread tab")
    page.locator('.nav-link[data-tab="proofread"]').click()
    _wait_for_activity(
        "Waiting for Proofread tab",
        lambda: page.evaluate(
            """() => ({
                visible: !!document.querySelector('#proofread-tab') && getComputedStyle(document.querySelector('#proofread-tab')).display !== 'none',
                has_book_btn: !!document.querySelector('#btn-proofread-book'),
                has_logs: !!document.querySelector('#proofread-logs'),
                has_table: !!document.querySelector('#proofread-table-body')
            })"""
        ),
        lambda snapshot: bool(
            snapshot.get("visible")
            and snapshot.get("has_book_btn")
            and snapshot.get("has_logs")
            and snapshot.get("has_table")
        ),
    )

    page.locator("#proofread-chapter-select").select_option("__whole_project__")
    _wait_for_activity(
        "Waiting for Proofread rows before run",
        lambda: page.evaluate(
            """() => ({
                row_count: document.querySelectorAll('#proofread-table-body tr[data-proofread-id]').length,
                chapter_value: String(document.querySelector('#proofread-chapter-select')?.value || '')
            })"""
        ),
        lambda snapshot: (
            int(snapshot.get("row_count") or 0) > 0
            and str(snapshot.get("chapter_value") or "") == "__whole_project__"
        ),
    )

    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/proofread")
            and response.request.method == "POST"
            and response.status == 200
        ),
        timeout=10000,
    ):
        page.locator("#btn-proofread-book").click()

    _wait_for_proofread_completion(app_base_url)

    page.locator("#proofread-chapter-select").select_option("__whole_project__")
    final_snapshot = _wait_for_activity(
        "Waiting for proofread strict pass summary",
        lambda: {
            "status": _fetch_task_status(app_base_url, "proofread"),
            "ui": page.evaluate(
                """() => {
                    const parseIntFrom = (id) => {
                        const raw = String(document.querySelector(id)?.innerText || '').trim();
                        const digits = raw.replace(/[^0-9]/g, '');
                        return digits ? Number.parseInt(digits, 10) : 0;
                    };
                    return {
                        row_count: document.querySelectorAll('#proofread-table-body tr[data-proofread-id]').length,
                        chapter_value: String(document.querySelector('#proofread-chapter-select')?.value || ''),
                        passed: parseIntFrom('#proofread-passed'),
                        failed: parseIntFrom('#proofread-failed'),
                        auto_failed: parseIntFrom('#proofread-auto-failed'),
                        phase_text: String(document.querySelector('#proofread-phase')?.innerText || '').trim().toLowerCase(),
                    };
                }"""
            ),
        },
        lambda snapshot: (
            not bool((snapshot.get("status") or {}).get("running"))
            and str(((snapshot.get("ui") or {}).get("chapter_value") or "")) == "__whole_project__"
            and int(((snapshot.get("ui") or {}).get("row_count") or 0)) > 0
            and int(((snapshot.get("ui") or {}).get("failed") or 0)) == 0
            and int(((snapshot.get("ui") or {}).get("auto_failed") or 0)) == 0
            and int(((snapshot.get("ui") or {}).get("passed") or 0))
            == int(((snapshot.get("ui") or {}).get("row_count") or 0))
        ),
    )
    ui_summary = dict(final_snapshot.get("ui") or {})
    assert int(ui_summary.get("row_count") or 0) > 0
    assert int(ui_summary.get("failed") or 0) == 0
    assert int(ui_summary.get("auto_failed") or 0) == 0
    assert int(ui_summary.get("passed") or 0) == int(ui_summary.get("row_count") or 0)
    return {"proofread_ui_summary": ui_summary}


__all__ = [name for name in globals() if not name.startswith("__")]
