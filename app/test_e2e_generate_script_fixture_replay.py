import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import requests

from e2e_sim import LMStudioSimServer
from runtime_layout import RuntimeLayout


SOURCE_LAYOUT = RuntimeLayout.from_app_dir(os.path.dirname(os.path.abspath(__file__)))
SOURCE_REPO_DIR = SOURCE_LAYOUT.repo_root
SOURCE_APP_DIR = SOURCE_LAYOUT.app_dir


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_status(response: requests.Response, expected: int, context: str) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={response.text[:1200]}"
        )


class _IsolatedServer:
    def __init__(self):
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.app_dir = ""

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_generate_script_fixture_")
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

        port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{port}"

        env = os.environ.copy()
        env["PINOKIO_SHARE_LOCAL"] = "false"
        env["PINOKIO_SHARE_LOCAL_PORT"] = str(port)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=self.app_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        deadline = time.time() + 60
        while time.time() < deadline:
            if self._proc.poll() is not None:
                output = ""
                if self._proc.stdout:
                    try:
                        output = self._proc.stdout.read() or ""
                    except Exception:
                        output = ""
                raise AssertionError(
                    f"Isolated server exited early with code {self._proc.returncode}.\n{output[-3000:]}"
                )
            try:
                response = requests.get(f"{self.base_url}/", timeout=1.5)
                if response.status_code < 500:
                    return self
            except Exception:
                pass
            time.sleep(0.3)

        raise AssertionError(f"Timed out waiting for isolated server at {self.base_url}")

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
        if self._temp_root and os.path.isdir(self._temp_root):
            shutil.rmtree(self._temp_root, ignore_errors=True)


def _poll_task(base_url: str, task_name: str, timeout_seconds: int = 600) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/{task_name}", timeout=30)
        _assert_status(response, 200, f"poll {task_name}")
        payload = response.json()
        if not payload.get("running"):
            return payload
        time.sleep(2)
    raise AssertionError(f"Timed out waiting for '{task_name}'")


def test_generate_script_replays_from_captured_fixture():
    fixture_path = os.path.join(
        SOURCE_APP_DIR,
        "test_fixtures",
        "e2e_sim",
        "lmstudio_generate_script_test_book.json",
    )
    assert os.path.exists(fixture_path), f"Missing fixture: {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    assert isinstance(fixture, dict)

    model_name = str(((fixture.get("metadata") or {}).get("model_name") or "").strip() or "google/gemma-4-26b-a4b")

    with LMStudioSimServer(fixture_path) as lm_server:
        with _IsolatedServer() as app_server:
            setup_payload = {
                "llm": {
                    "base_url": f"{lm_server.base_url}/v1",
                    "api_key": "local",
                    "model_name": model_name,
                    "llm_workers": 1,
                },
                "generation": {
                    "chunk_size": 600,
                    "max_tokens": 1024,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "banned_tokens": [],
                    "merge_narrators": False,
                    "orphaned_text_to_narrator_on_repair": True,
                    "legacy_mode": False,
                    "temperament_words": 150,
                },
            }
            response = requests.post(f"{app_server.base_url}/api/config/setup", json=setup_payload, timeout=60)
            _assert_status(response, 200, "setup config")

            config_response = requests.get(f"{app_server.base_url}/api/config", timeout=30)
            _assert_status(config_response, 200, "get config")
            config_payload = config_response.json()
            scripts_config_path = os.path.join(app_server.app_dir, "scripts", "config.json")
            with open(scripts_config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "llm": setup_payload["llm"],
                        "generation": setup_payload["generation"],
                        "prompts": config_payload.get("prompts") or {},
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")
            with open(book_path, "rb") as handle:
                files = {"file": ("test_book.epub", io.BytesIO(handle.read()), "application/epub+zip")}
                upload_response = requests.post(f"{app_server.base_url}/api/upload", files=files, timeout=120)
            _assert_status(upload_response, 200, "upload book")

            start_response = requests.post(f"{app_server.base_url}/api/generate_script", json={}, timeout=60)
            _assert_status(start_response, 200, "start generate script")

            status_payload = _poll_task(app_server.base_url, "script")
            logs = [str(item) for item in (status_payload or {}).get("logs") or []]
            for line in logs:
                lower = line.lower()
                assert not line.startswith("Error:"), f"Generate Script error log: {line}"
                assert "failed with return code" not in lower, f"Generate Script failed: {line}"

            script_response = requests.get(f"{app_server.base_url}/api/annotated_script", timeout=30)
            _assert_status(script_response, 200, "fetch annotated script")
            entries = script_response.json()
            assert isinstance(entries, list)
            assert len(entries) > 0

        lm_server.assert_all_consumed()
