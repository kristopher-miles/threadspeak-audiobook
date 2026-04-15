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


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


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
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_editor_audio_fixture_")
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


def _seed_clone_voices(base_url: str, project_dir: str, voice_manifest_path: str) -> None:
    with open(voice_manifest_path, "r", encoding="utf-8") as handle:
        voice_manifest = json.load(handle)

    profiles = [dict(item or {}) for item in (voice_manifest.get("voice_profiles") or [])]
    assert profiles

    voices_response = requests.get(f"{base_url}/api/voices", timeout=60)
    _assert_status(voices_response, 200, "get voices")
    voices = list(voices_response.json() or [])
    by_norm = {
        _normalize_name(str((row or {}).get("name") or "")): dict(row or {})
        for row in voices
        if _normalize_name(str((row or {}).get("name") or ""))
    }
    profile_by_norm = {
        _normalize_name(str(item.get("speaker") or "")): item
        for item in profiles
        if _normalize_name(str(item.get("speaker") or ""))
    }

    config_update = {}
    for norm_name, row in by_norm.items():
        profile = profile_by_norm.get(norm_name)
        assert profile is not None, f"Missing voice profile for {(row or {}).get('name')}"

        ref_audio = str(profile.get("ref_audio") or "").strip()
        assert ref_audio
        source_asset_rel = str(profile.get("fixture_audio_path") or "").strip()
        source_asset_abs = source_asset_rel if os.path.isabs(source_asset_rel) else os.path.join(SOURCE_REPO_DIR, source_asset_rel)
        assert os.path.exists(source_asset_abs)

        target_audio_abs = ref_audio if os.path.isabs(ref_audio) else os.path.join(project_dir, ref_audio)
        os.makedirs(os.path.dirname(target_audio_abs), exist_ok=True)
        shutil.copy2(source_asset_abs, target_audio_abs)

        speaker = str((row or {}).get("name") or "").strip()
        cfg = dict((row or {}).get("config") or {})
        cfg.update(
            {
                "type": "clone",
                "ref_audio": ref_audio,
                "ref_text": str(profile.get("sample_text") or "").strip(),
                "generated_ref_text": str(profile.get("generated_ref_text") or "").strip(),
                "description": str(profile.get("description") or "").strip(),
            }
        )
        config_update[speaker] = cfg

    save_response = requests.post(
        f"{base_url}/api/voices/batch",
        json={"config": config_update, "confirm_invalidation": False},
        timeout=60,
    )
    _assert_status(save_response, 200, "seed clone voices")


def test_editor_audio_fixture_replay():
    fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
    script_fixture = os.path.join(fixtures_dir, "lmstudio_generate_script_test_book.json")
    voice_manifest = os.path.join(fixtures_dir, "voice_profiles_test_book_manifest.json")
    qwen_fixture = os.path.join(fixtures_dir, "qwen_local_editor_audio_test_book.json")
    editor_manifest = os.path.join(fixtures_dir, "editor_audio_test_book_manifest.json")
    book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")

    assert os.path.exists(script_fixture), f"Missing fixture: {script_fixture}"
    assert os.path.exists(voice_manifest), f"Missing fixture: {voice_manifest}"
    assert os.path.exists(qwen_fixture), f"Missing fixture: {qwen_fixture}"
    assert os.path.exists(editor_manifest), f"Missing fixture: {editor_manifest}"
    assert os.path.exists(book_path), f"Missing source: {book_path}"

    with open(script_fixture, "r", encoding="utf-8") as handle:
        script_payload = json.load(handle)
    with open(editor_manifest, "r", encoding="utf-8") as handle:
        editor_payload = json.load(handle)

    script_model = str(((script_payload.get("metadata") or {}).get("model_name") or "").strip())
    expected_line_count = int(editor_payload.get("line_count") or 0)
    assert script_model
    assert expected_line_count > 0

    with tempfile.TemporaryDirectory(prefix="threadspeak_editor_audio_report_") as report_root:
        qwen_report = os.path.join(report_root, "qwen-report.json")

        with LMStudioSimServer(script_fixture) as script_server:
            with _IsolatedServer(
                env_overrides={
                    "THREADSPEAK_E2E_SIM_ENABLED": "1",
                    "THREADSPEAK_E2E_QWEN_FIXTURE": qwen_fixture,
                    "THREADSPEAK_E2E_QWEN_REPORT_PATH": qwen_report,
                    "THREADSPEAK_E2E_SIM_STRICT": "1",
                }
            ) as app_server:
                assert app_server.layout is not None

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
                script_status = _poll_task(app_server.base_url, "script")
                logs = [str(item) for item in (script_status or {}).get("logs") or []]
                for line in logs:
                    lower = line.lower()
                    assert not line.startswith("Error:"), f"script error: {line}"
                    assert "failed with return code" not in lower, f"script failed: {line}"

                script_server.assert_all_consumed()

                _seed_clone_voices(
                    base_url=app_server.base_url,
                    project_dir=app_server.layout.project_dir,
                    voice_manifest_path=voice_manifest,
                )

                setup_generation = {
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
                        "max_tokens": 1024,
                    },
                }
                response = requests.post(f"{app_server.base_url}/api/config/setup", json=setup_generation, timeout=60)
                _assert_status(response, 200, "setup generation phase")

                run_response = requests.post(
                    f"{app_server.base_url}/api/generate_batch",
                    json={"scope": "project", "scope_mode": "project", "regenerate_all": False},
                    timeout=60,
                )
                _assert_status(run_response, 200, "start editor generate_batch")

                audio_status = _wait_for_audio_idle(app_server.base_url)
                recent_jobs = list(audio_status.get("recent_jobs") or [])
                assert recent_jobs, "No recent_jobs found after audio run"
                latest = dict(recent_jobs[0] or {})
                assert str(latest.get("status") or "").strip().lower() == "completed", f"unexpected audio job status: {latest}"

                chunks_response = requests.get(f"{app_server.base_url}/api/chunks", timeout=120)
                _assert_status(chunks_response, 200, "get chunks after generation")
                chunks = list(chunks_response.json() or [])
                non_empty = [chunk for chunk in chunks if str((chunk or {}).get("text") or "").strip()]
                assert len(non_empty) == expected_line_count, (
                    f"Expected {expected_line_count} non-empty chunks, got {len(non_empty)}"
                )
                for chunk in non_empty:
                    status = str((chunk or {}).get("status") or "").strip().lower()
                    assert status == "done", f"chunk not done: {chunk.get('uid')} status={status}"
                    audio_ref = str((chunk or {}).get("audio_path") or "").strip()
                    assert audio_ref, f"chunk missing audio_path: {chunk.get('uid')}"
                    audio_abs = audio_ref if os.path.isabs(audio_ref) else os.path.join(app_server.layout.project_dir, audio_ref)
                    assert os.path.exists(audio_abs), f"audio file missing: {audio_abs}"

        with open(qwen_report, "r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
        assert report_payload.get("pending") == {}
