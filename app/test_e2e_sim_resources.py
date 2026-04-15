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
SOURCE_APP_DIR = SOURCE_LAYOUT.app_dir
SOURCE_REPO_DIR = SOURCE_LAYOUT.repo_root


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_status(response, expected=200, context=""):
    if response.status_code != expected:
        raise AssertionError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={response.text[:1000]}"
        )


class _IsolatedServer:
    def __init__(self, env_overrides=None):
        self._temp_root = None
        self._proc = None
        self.base_url = ""
        self.app_dir = ""
        self.env_overrides = dict(env_overrides or {})

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_e2e_sim_")
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

    def __exit__(self, exc_type, exc_val, exc_tb):
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


def _post(base_url: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 120)
    return requests.post(f"{base_url}{path}", timeout=timeout, **kwargs)


def test_simulator_resources_cover_llm_and_local_qwen_paths():
    fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
    lm_fixture = os.path.join(fixtures_dir, "lmstudio_voice_description.json")
    qwen_fixture = os.path.join(fixtures_dir, "qwen_local_voice_design.json")

    with tempfile.TemporaryDirectory(prefix="threadspeak_qwen_sim_report_") as report_root:
        qwen_report = os.path.join(report_root, "qwen-report.json")

        with LMStudioSimServer(lm_fixture) as lm_server:
            with _IsolatedServer(
                env_overrides={
                    "THREADSPEAK_E2E_SIM_ENABLED": "1",
                    "THREADSPEAK_E2E_QWEN_FIXTURE": qwen_fixture,
                    "THREADSPEAK_E2E_QWEN_REPORT_PATH": qwen_report,
                    "THREADSPEAK_E2E_SIM_STRICT": "1",
                }
            ) as app_server:
                setup_payload = {
                    "llm": {
                        "base_url": f"{lm_server.base_url}/v1",
                        "api_key": "local",
                        "model_name": "sim-tool-model",
                        "llm_workers": 1,
                    },
                    "tts": {
                        "mode": "local",
                        "local_backend": "qwen",
                        "url": "http://127.0.0.1:7860",
                        "device": "cpu",
                    },
                }
                response = _post(app_server.base_url, "/api/config/setup", json=setup_payload, timeout=60)
                _assert_status(response, 200, "setup config")

                response = _post(
                    app_server.base_url,
                    "/api/voices/suggest_description",
                    json={"speaker": "NARRATOR"},
                    timeout=120,
                )
                _assert_status(response, 200, "suggest description")
                suggestion = response.json()
                assert suggestion.get("status") == "ok"
                assert suggestion.get("voice") == "Calm, measured narrator with warm tone"
                assert suggestion.get("llm_mode") == "tool"
                assert bool(suggestion.get("llm_tool_call_observed")) is True

                response = _post(
                    app_server.base_url,
                    "/api/voices/design_generate",
                    json={
                        "speaker": "NARRATOR",
                        "description": "Confident and gentle storyteller",
                        "sample_text": "The gate opened slowly as dawn broke.",
                        "force": True,
                    },
                    timeout=120,
                )
                _assert_status(response, 200, "design generate")
                generated = response.json()
                assert generated.get("status") == "ok"
                ref_audio = str(generated.get("ref_audio") or "")
                assert ref_audio.startswith("clone_voices/")

                isolated_layout = RuntimeLayout.from_app_dir(app_server.app_dir)
                output_path = os.path.join(isolated_layout.project_dir, ref_audio)
                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0

            lm_server.assert_all_consumed()

        with open(qwen_report, "r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
        assert report_payload.get("pending") == {}
