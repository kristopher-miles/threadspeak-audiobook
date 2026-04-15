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
    def __init__(self, env_overrides=None):
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.app_dir = ""
        self.env_overrides = dict(env_overrides or {})

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_voice_profile_fixture_")
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
        env.update(self.env_overrides)

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


def _sync_scripts_config(app_base_url: str, app_dir: str, setup_payload: dict) -> None:
    config_response = requests.get(f"{app_base_url}/api/config", timeout=30)
    _assert_status(config_response, 200, "get config")
    config_payload = config_response.json()
    scripts_config_path = os.path.join(app_dir, "scripts", "config.json")
    with open(scripts_config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "llm": setup_payload.get("llm") or {},
                "generation": setup_payload.get("generation") or {},
                "prompts": config_payload.get("prompts") or {},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _poll_task(base_url: str, task_name: str, timeout_seconds: int = 900) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/{task_name}", timeout=30)
        _assert_status(response, 200, f"poll {task_name}")
        payload = response.json()
        if not payload.get("running"):
            return payload
        time.sleep(2)
    raise AssertionError(f"Timed out waiting for '{task_name}'")


def test_voice_profile_fixtures_replay():
    fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
    script_fixture = os.path.join(fixtures_dir, "lmstudio_generate_script_test_book.json")
    voice_llm_fixture = os.path.join(fixtures_dir, "lmstudio_voice_profiles_test_book.json")
    qwen_fixture = os.path.join(fixtures_dir, "qwen_local_voice_profiles_test_book.json")
    manifest_path = os.path.join(fixtures_dir, "voice_profiles_test_book_manifest.json")
    book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")

    assert os.path.exists(script_fixture), f"Missing fixture: {script_fixture}"
    assert os.path.exists(voice_llm_fixture), f"Missing fixture: {voice_llm_fixture}"
    assert os.path.exists(qwen_fixture), f"Missing fixture: {qwen_fixture}"
    assert os.path.exists(manifest_path), f"Missing fixture: {manifest_path}"
    assert os.path.exists(book_path), f"Missing source: {book_path}"

    with open(script_fixture, "r", encoding="utf-8") as handle:
        script_payload = json.load(handle)
    with open(voice_llm_fixture, "r", encoding="utf-8") as handle:
        voice_payload = json.load(handle)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    script_model = str(((script_payload.get("metadata") or {}).get("model_name") or "").strip())
    voice_model = str(((voice_payload.get("metadata") or {}).get("model_name") or "").strip())
    speakers = [str(value).strip() for value in (manifest.get("speakers") or []) if str(value).strip()]
    assert script_model
    assert voice_model
    assert speakers

    with tempfile.TemporaryDirectory(prefix="threadspeak_voice_profile_report_") as report_root:
        qwen_report = os.path.join(report_root, "qwen-report.json")

        with LMStudioSimServer(script_fixture) as script_server:
            with LMStudioSimServer(voice_llm_fixture) as voice_server:
                with _IsolatedServer(
                    env_overrides={
                        "THREADSPEAK_E2E_SIM_ENABLED": "1",
                        "THREADSPEAK_E2E_QWEN_FIXTURE": qwen_fixture,
                        "THREADSPEAK_E2E_QWEN_REPORT_PATH": qwen_report,
                        "THREADSPEAK_E2E_SIM_STRICT": "1",
                    }
                ) as app_server:
                    setup_script = {
                        "llm": {
                            "base_url": f"{script_server.base_url}/v1",
                            "api_key": "local",
                            "model_name": script_model,
                            "llm_workers": 1,
                        },
                        "tts": {
                            "mode": "local",
                            "local_backend": "qwen",
                            "device": "cpu",
                            "language": "English",
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
                        },
                    }
                    response = requests.post(f"{app_server.base_url}/api/config/setup", json=setup_script, timeout=60)
                    _assert_status(response, 200, "setup script phase")
                    _sync_scripts_config(app_server.base_url, app_server.app_dir, setup_script)

                    with open(book_path, "rb") as handle:
                        files = {"file": ("test_book.epub", io.BytesIO(handle.read()), "application/epub+zip")}
                        upload_response = requests.post(f"{app_server.base_url}/api/upload", files=files, timeout=120)
                    _assert_status(upload_response, 200, "upload")

                    start_response = requests.post(f"{app_server.base_url}/api/generate_script", json={}, timeout=60)
                    _assert_status(start_response, 200, "start generate script")
                    script_status = _poll_task(app_server.base_url, "script")
                    logs = [str(item) for item in (script_status or {}).get("logs") or []]
                    for line in logs:
                        lower = line.lower()
                        assert not line.startswith("Error:"), f"script error: {line}"
                        assert "failed with return code" not in lower, f"script failed: {line}"

                    script_server.assert_all_consumed()

                    setup_voice = {
                        "llm": {
                            "base_url": f"{voice_server.base_url}/v1",
                            "api_key": "local",
                            "model_name": voice_model,
                            "llm_workers": 1,
                        },
                        "tts": {
                            "mode": "local",
                            "local_backend": "qwen",
                            "device": "cpu",
                            "language": "English",
                        },
                        "generation": {
                            "max_tokens": 1024,
                        },
                    }
                    response = requests.post(f"{app_server.base_url}/api/config/setup", json=setup_voice, timeout=60)
                    _assert_status(response, 200, "setup voice phase")

                    suggestion_response = requests.post(
                        f"{app_server.base_url}/api/voices/suggest_descriptions_bulk",
                        json={"speakers": speakers},
                        timeout=300,
                    )
                    _assert_status(suggestion_response, 200, "bulk suggest")
                    suggestion_payload = suggestion_response.json()
                    results = list(suggestion_payload.get("results") or [])
                    failures = list(suggestion_payload.get("failures") or [])
                    assert not failures, f"suggestion failures: {failures}"
                    assert len(results) == len(speakers)

                    voices_response = requests.get(f"{app_server.base_url}/api/voices", timeout=30)
                    _assert_status(voices_response, 200, "get voices")
                    voices_payload = voices_response.json()
                    sample_by_speaker = {
                        str((item or {}).get("name") or "").strip(): str(
                            (item or {}).get("suggested_sample_text") or ""
                        ).strip()
                        for item in voices_payload
                        if str((item or {}).get("name") or "").strip()
                    }

                    isolated_layout = RuntimeLayout.from_app_dir(app_server.app_dir)
                    for item in results:
                        speaker = str(item.get("speaker") or "").strip()
                        description = str(item.get("voice") or "").strip()
                        sample_text = sample_by_speaker.get(speaker) or ""
                        assert speaker and description and sample_text

                        generate_response = requests.post(
                            f"{app_server.base_url}/api/voices/design_generate",
                            json={
                                "speaker": speaker,
                                "description": description,
                                "sample_text": sample_text,
                                "force": False,
                            },
                            timeout=900,
                        )
                        _assert_status(generate_response, 200, f"design generate {speaker}")
                        generated = generate_response.json()
                        ref_audio = str(generated.get("ref_audio") or "").strip()
                        assert ref_audio.startswith("clone_voices/")
                        out_path = os.path.join(isolated_layout.project_dir, ref_audio)
                        assert os.path.exists(out_path)
                        assert os.path.getsize(out_path) > 0

                voice_server.assert_all_consumed()

        with open(qwen_report, "r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
        assert report_payload.get("pending") == {}
