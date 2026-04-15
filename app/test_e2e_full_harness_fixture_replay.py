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
        self.layout: RuntimeLayout | None = None
        self.env_overrides = dict(env_overrides or {})

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_full_harness_fixture_")
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
                    self.layout = RuntimeLayout.from_app_dir(self.app_dir)
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


def _poll_task(base_url: str, task_name: str, timeout_seconds: int = 1200) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/{task_name}", timeout=30)
        _assert_status(response, 200, f"poll {task_name}")
        payload = response.json()
        if not payload.get("running"):
            return payload
        time.sleep(2)
    raise AssertionError(f"Timed out waiting for '{task_name}'")


def _wait_for_audio_idle(base_url: str, timeout_seconds: int = 1800) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/audio", timeout=30)
        _assert_status(response, 200, "poll audio")
        payload = response.json()
        if (
            not bool(payload.get("running"))
            and not list(payload.get("queue") or [])
            and not payload.get("current_job")
        ):
            return payload
        time.sleep(2)
    raise AssertionError("Timed out waiting for audio queue to become idle")


def test_full_harness_fixtures_replay():
    fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
    harness_path = os.path.join(fixtures_dir, "full_e2e_test_book_harness.json")
    voice_manifest_path = os.path.join(fixtures_dir, "voice_profiles_test_book_manifest.json")
    book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")

    assert os.path.exists(harness_path), f"Missing harness: {harness_path}"
    assert os.path.exists(voice_manifest_path), f"Missing voice manifest: {voice_manifest_path}"
    assert os.path.exists(book_path), f"Missing source book: {book_path}"

    with open(harness_path, "r", encoding="utf-8") as handle:
        harness = json.load(handle)

    script_lm_fixture = os.path.join(SOURCE_REPO_DIR, harness["phases"]["generate_script"]["lm_fixture"])
    voice_lm_fixture = os.path.join(SOURCE_REPO_DIR, harness["phases"]["voice_profiles"]["lm_fixture"])
    qwen_fixture = os.path.join(SOURCE_REPO_DIR, harness["phases"]["editor_audio"]["qwen_fixture"])
    proofread_text_fixture = os.path.join(SOURCE_REPO_DIR, harness["phases"]["proofread"]["text_fixture"])

    assert os.path.exists(script_lm_fixture)
    assert os.path.exists(voice_lm_fixture)
    assert os.path.exists(qwen_fixture)
    assert os.path.exists(proofread_text_fixture)

    script_model = str(harness["phases"]["generate_script"]["model_name"])
    voice_model = str(harness["phases"]["voice_profiles"]["model_name"])
    assert script_model
    assert voice_model

    with open(voice_manifest_path, "r", encoding="utf-8") as handle:
        voice_manifest = json.load(handle)
    speakers = [str(value).strip() for value in (voice_manifest.get("speakers") or []) if str(value).strip()]
    assert speakers

    with tempfile.TemporaryDirectory(prefix="threadspeak_full_harness_report_") as report_root:
        qwen_report = os.path.join(report_root, "qwen-report.json")

        with LMStudioSimServer(script_lm_fixture) as script_server:
            with LMStudioSimServer(voice_lm_fixture) as voice_server:
                with _IsolatedServer(
                    env_overrides={
                        "THREADSPEAK_E2E_SIM_ENABLED": "1",
                        "THREADSPEAK_E2E_QWEN_FIXTURE": qwen_fixture,
                        "THREADSPEAK_E2E_QWEN_REPORT_PATH": qwen_report,
                        "THREADSPEAK_E2E_PROOFREAD_FIXTURE": proofread_text_fixture,
                        "THREADSPEAK_E2E_PROOFREAD_FALLBACK": "chunk_text",
                        "THREADSPEAK_E2E_SIM_STRICT": "1",
                    }
                ) as app_server:
                    # Phase 1: Generate Script
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
                            "parallel_workers": 1,
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
                    _poll_task(app_server.base_url, "script")

                    # Phase 2: Character generation
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
                            "parallel_workers": 1,
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
                    assert not failures
                    assert len(results) == len(speakers)

                    by_speaker = {
                        str(item.get("speaker") or "").strip(): str(item.get("voice") or "").strip()
                        for item in results
                        if str(item.get("speaker") or "").strip()
                    }

                    with open(voice_manifest_path, "r", encoding="utf-8") as handle:
                        voice_manifest_data = json.load(handle)
                    sample_by_speaker = {
                        str(item.get("speaker") or "").strip(): str(item.get("sample_text") or "").strip()
                        for item in (voice_manifest_data.get("voice_profiles") or [])
                        if str(item.get("speaker") or "").strip()
                    }

                    for speaker in speakers:
                        description = by_speaker.get(speaker)
                        sample_text = sample_by_speaker.get(speaker)
                        assert description and sample_text
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

                    # Phase 3: Editor audio generation
                    run_response = requests.post(
                        f"{app_server.base_url}/api/generate_batch",
                        json={"scope": "project", "scope_mode": "project", "regenerate_all": False},
                        timeout=60,
                    )
                    _assert_status(run_response, 200, "start editor generate_batch")
                    audio_status = _wait_for_audio_idle(app_server.base_url)
                    recent_jobs = list(audio_status.get("recent_jobs") or [])
                    assert recent_jobs, "No recent audio jobs found"
                    latest = dict(recent_jobs[0] or {})
                    assert str(latest.get("status") or "").strip().lower() == "completed"

                    chunks_response = requests.get(f"{app_server.base_url}/api/chunks", timeout=120)
                    _assert_status(chunks_response, 200, "get chunks")
                    chunks = list(chunks_response.json() or [])
                    non_empty = [chunk for chunk in chunks if str((chunk or {}).get("text") or "").strip()]
                    assert non_empty
                    for chunk in non_empty:
                        status = str((chunk or {}).get("status") or "").strip().lower()
                        assert status == "done"
                        audio_ref = str((chunk or {}).get("audio_path") or "").strip()
                        assert audio_ref
                        audio_abs = audio_ref if os.path.isabs(audio_ref) else os.path.join(app_server.layout.project_dir, audio_ref)
                        assert os.path.exists(audio_abs)

                script_server.assert_all_consumed()
                voice_server.assert_all_consumed()

        with open(qwen_report, "r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
        pending = dict(report_payload.get("pending") or {})
        allowed_pending = {"generate_voice_clone", "generate_voice_design"}
        disallowed_pending = {
            key: value
            for key, value in pending.items()
            if str(key) not in allowed_pending
        }
        assert not disallowed_pending, f"Unexpected pending Qwen interactions: {disallowed_pending}"
