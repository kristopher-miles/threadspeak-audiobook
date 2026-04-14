import errno
import fcntl
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import pytest
import requests

from runtime_layout import RuntimeLayout


LMSTUDIO_BASE_URL = "http://127.0.0.1:1234"
TOOL_MODEL_NAME = "qwen/qwen3.5-9b"
NON_TOOL_MODEL_NAME = "bartowski/google_gemma-3-4b-it-GGUF"
TOOL_MODEL_CONTEXT_LENGTH = 16384
NON_TOOL_MODEL_CONTEXT_LENGTH = 16384
MAX_RETRIES_PER_SCENARIO = 1
MODEL_LOAD_POLL_INTERVAL_SECONDS = 5
MODEL_LOAD_WAIT_TIMEOUT_SECONDS = 300
LIVE_TEST_LOCK_PATH = os.path.join(tempfile.gettempdir(), "threadspeak_lmstudio_live_endpoints.lock")

SOURCE_LAYOUT = RuntimeLayout.from_app_dir(os.path.dirname(os.path.abspath(__file__)))
SOURCE_REPO_DIR = SOURCE_LAYOUT.repo_root
SOURCE_APP_DIR = SOURCE_LAYOUT.app_dir

MODE_PATTERN = re.compile(r"llm_mode=([^\s]+)\s+tool_call_observed=(True|False)")

SAMPLE_BOOK_TEXT = """CHAPTER ONE
Elena paused at the threshold and listened to the rain hammering the tin roof.
"We should leave now," she whispered.
Marcus shook his head and kept his eyes on the old ledger.
"Not yet," he said quietly. "If we run now, we lose everything."
Elena gripped the chair until her knuckles turned white.
"Then tell me the truth," Elena said, staring at him.
Marcus swallowed. "I burned the letter," he admitted.

CHAPTER TWO
The hallway was cold and smelled faintly of smoke.
"You lied to me before," Elena said.
"I know," Marcus replied, "and I'm done lying."
Outside, thunder rolled over the empty road.
"""


def _normalize_http_url(raw_url: str) -> str:
    value = str(raw_url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


@contextmanager
def _global_live_test_run_guard():
    handle = open(LIVE_TEST_LOCK_PATH, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            handle.seek(0)
            owner = (handle.read() or "").strip()
            owner_hint = f" lock owner: {owner}" if owner else ""
            raise AssertionError(
                "Refusing to run LM Studio live endpoint test concurrently. "
                "Another test run is already in progress, and parallel runs can crash the host. "
                f"Wait for the current run to finish and retry. Lock file: {LIVE_TEST_LOCK_PATH}.{owner_hint}"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at_epoch": int(time.time()),
                },
                ensure_ascii=False,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _probe_lmstudio_or_skip() -> None:
    try:
        response = requests.get(f"{LMSTUDIO_BASE_URL}/v1/models", timeout=3)
    except Exception as exc:
        pytest.skip(f"LM Studio is not reachable at {LMSTUDIO_BASE_URL}: {exc}")
    if response.status_code >= 500:
        pytest.skip(
            f"LM Studio probe failed at {LMSTUDIO_BASE_URL}/v1/models with status {response.status_code}"
        )


def _lmstudio_model_name_matches(model_payload: dict, wanted_model: str) -> bool:
    wanted = str(wanted_model or "").strip()
    if not wanted:
        return False
    candidates = {
        str(model_payload.get("key") or "").strip(),
        str(model_payload.get("display_name") or "").strip(),
    }
    for instance in model_payload.get("loaded_instances") or []:
        if isinstance(instance, dict):
            candidates.add(str(instance.get("id") or "").strip())
    return wanted in candidates


def _wait_for_lmstudio_model_loaded(model_name: str):
    deadline = time.time() + MODEL_LOAD_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            response = requests.get(f"{LMSTUDIO_BASE_URL}/api/v1/models", timeout=30)
            if response.status_code == 200:
                payload = response.json()
                models = payload.get("models") if isinstance(payload, dict) else None
                if isinstance(models, list):
                    for model in models:
                        if not isinstance(model, dict):
                            continue
                        if not _lmstudio_model_name_matches(model, model_name):
                            continue
                        loaded_instances = model.get("loaded_instances")
                        if loaded_instances is None or bool(loaded_instances):
                            return
        except Exception:
            pass
        time.sleep(MODEL_LOAD_POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"Timed out waiting for LM Studio to load model '{model_name}' "
        f"at {LMSTUDIO_BASE_URL}/api/v1/models"
    )


def _load_lmstudio_model_once(model_name: str, context_length: int):
    response = requests.post(
        f"{LMSTUDIO_BASE_URL}/api/v1/models/load",
        json={
            "model": model_name,
            "context_length": int(context_length),
            "echo_load_config": True,
        },
        timeout=240,
    )
    if response.status_code != 200:
        raise AssertionError(
            f"LM Studio model load failed for '{model_name}' with "
            f"HTTP {response.status_code}: {response.text[:1000]}"
        )
    payload = response.json()
    if str(payload.get("status") or "") != "loaded":
        raise AssertionError(
            f"LM Studio model load returned unexpected status for '{model_name}': {payload}"
        )
    if str(payload.get("type") or "") != "llm":
        raise AssertionError(
            f"LM Studio model load returned unexpected type for '{model_name}': {payload}"
        )
    _wait_for_lmstudio_model_loaded(model_name)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _IsolatedServer:
    def __init__(self):
        self._temp_root = None
        self._proc = None
        self.base_url = ""
        self.app_dir = ""

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_lmstudio_live_")
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
    timeout = kwargs.pop("timeout", 90)
    return requests.post(f"{base_url}{path}", timeout=timeout, **kwargs)


def _get(base_url: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 30)
    return requests.get(f"{base_url}{path}", timeout=timeout, **kwargs)


def _assert_status(response, expected=200, context=""):
    if response.status_code != expected:
        body = response.text[:800]
        raise AssertionError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={body}"
        )


def _configure_llm_for_phase(base_url: str, model_name: str, app_dir: str):
    payload = {
        "llm": {
            "base_url": LMSTUDIO_BASE_URL,
            "api_key": "local",
            "model_name": model_name,
            "llm_workers": 1,
        },
        "generation": {
            "chunk_size": 1200,
            "max_tokens": 512,
            "review_batch_size": 12,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0.0,
        },
    }
    response = _post(base_url, "/api/config/setup", json=payload, timeout=60)
    _assert_status(response, 200, "configure llm")

    # Subprocess scripts still read app/scripts/config.json.
    scripts_config_path = os.path.join(app_dir, "scripts", "config.json")
    with open(scripts_config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "llm": payload["llm"],
                "generation": payload["generation"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _upload_sample_book(base_url: str):
    files = {
        "file": ("lmstudio_live_sample.txt", io.BytesIO(SAMPLE_BOOK_TEXT.encode("utf-8")), "text/plain")
    }
    response = _post(base_url, "/api/upload", files=files, timeout=60)
    _assert_status(response, 200, "upload sample book")


def _wait_for_task(base_url: str, task_name: str, timeout=420, poll_interval=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = _get(base_url, f"/api/status/{task_name}", timeout=30)
        _assert_status(response, 200, f"poll status {task_name}")
        payload = response.json()
        if not payload.get("running"):
            return payload
        time.sleep(poll_interval)
    raise AssertionError(f"Timed out waiting for task '{task_name}'")


def _assert_task_succeeded(task_name: str, status_payload: dict):
    logs = [str(item) for item in (status_payload or {}).get("logs") or []]
    lower_logs = [line.lower() for line in logs]
    for line in lower_logs:
        if "failed with return code" in line:
            raise AssertionError(f"{task_name} failed. logs={logs[-30:]}")
    for line in logs:
        if line.startswith("Error:"):
            raise AssertionError(f"{task_name} error. line={line}")


def _start_and_wait_task(base_url: str, endpoint: str, task_name: str):
    response = _post(base_url, endpoint, timeout=60)
    _assert_status(response, 200, f"start {endpoint}")
    status_payload = _wait_for_task(base_url, task_name)
    _assert_task_succeeded(task_name, status_payload)
    return status_payload


def _prepare_paragraph_pipeline(base_url: str):
    _upload_sample_book(base_url)
    _start_and_wait_task(base_url, "/api/process_paragraphs", "process_paragraphs")


def _prepare_script_pipeline(base_url: str):
    _upload_sample_book(base_url)
    _start_and_wait_task(base_url, "/api/generate_script", "script")


def _assert_mode_telemetry(
    endpoint_name: str,
    telemetry_rows,
    *,
    expected_mode: str,
    expected_tool_call_observed: bool,
):
    if not telemetry_rows:
        raise AssertionError(f"{endpoint_name}: no LLM telemetry rows were captured")
    for mode, tool_call_observed in telemetry_rows:
        if mode != expected_mode:
            raise AssertionError(
                f"{endpoint_name}: expected llm_mode={expected_mode}, got {mode}. rows={telemetry_rows}"
            )
        if bool(tool_call_observed) != bool(expected_tool_call_observed):
            raise AssertionError(
                f"{endpoint_name}: expected tool_call_observed={expected_tool_call_observed}, got {tool_call_observed}. "
                f"rows={telemetry_rows}"
            )


def _extract_generic_telemetry_rows(logs):
    rows = []
    for line in logs or []:
        match = MODE_PATTERN.search(str(line))
        if not match:
            continue
        rows.append((match.group(1), match.group(2) == "True"))
    return rows


def _scenario_generate_script(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_script_pipeline(base_url)
    status = _get(base_url, "/api/status/script", timeout=30).json()
    rows = _extract_generic_telemetry_rows(status.get("logs") or [])
    _assert_mode_telemetry("generate_script", rows, expected_mode=expected_mode, expected_tool_call_observed=expected_tool)


def _scenario_review_script(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_script_pipeline(base_url)
    status = _start_and_wait_task(base_url, "/api/review_script", "review")
    rows = _extract_generic_telemetry_rows(status.get("logs") or [])
    _assert_mode_telemetry("review_script", rows, expected_mode=expected_mode, expected_tool_call_observed=expected_tool)


def _scenario_assign_dialogue(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_paragraph_pipeline(base_url)
    status = _start_and_wait_task(base_url, "/api/assign_dialogue", "assign_dialogue")
    rows = _extract_generic_telemetry_rows(status.get("logs") or [])
    _assert_mode_telemetry("assign_dialogue", rows, expected_mode=expected_mode, expected_tool_call_observed=expected_tool)


def _scenario_extract_temperament(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_paragraph_pipeline(base_url)
    status = _start_and_wait_task(base_url, "/api/extract_temperament", "extract_temperament")
    rows = _extract_generic_telemetry_rows(status.get("logs") or [])
    _assert_mode_telemetry("extract_temperament", rows, expected_mode=expected_mode, expected_tool_call_observed=expected_tool)


def _scenario_script_sanity_check(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_script_pipeline(base_url)
    status = _start_and_wait_task(base_url, "/api/script_sanity_check", "sanity")
    logs = [str(item) for item in (status.get("logs") or [])]

    mode_line = ""
    tool_values_line = ""
    for line in logs:
        if line.startswith("Dialogue-attribution llm modes:"):
            mode_line = line
        if line.startswith("Dialogue-attribution tool_call_observed values:"):
            tool_values_line = line

    if not mode_line:
        raise AssertionError(f"script_sanity_check: missing mode telemetry in logs={logs[-40:]}")
    if not tool_values_line:
        raise AssertionError(f"script_sanity_check: missing tool telemetry in logs={logs[-40:]}")

    mode_values = [part.strip() for part in mode_line.split(":", 1)[1].split(",") if part.strip()]
    tool_values = [part.strip() for part in tool_values_line.split(":", 1)[1].split(",") if part.strip()]
    parsed_rows = [(mode, value.lower() == "true") for mode in mode_values for value in tool_values]
    _assert_mode_telemetry(
        "script_sanity_check",
        parsed_rows,
        expected_mode=expected_mode,
        expected_tool_call_observed=expected_tool,
    )


def _scenario_voice_suggest_description(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_script_pipeline(base_url)
    response = _post(
        base_url,
        "/api/voices/suggest_description",
        json={"speaker": "NARRATOR"},
        timeout=180,
    )
    _assert_status(response, 200, "voice suggest description")
    payload = response.json()
    rows = [(str(payload.get("llm_mode") or ""), bool(payload.get("llm_tool_call_observed")))]
    _assert_mode_telemetry(
        "voices/suggest_description",
        rows,
        expected_mode=expected_mode,
        expected_tool_call_observed=expected_tool,
    )


def _scenario_voice_suggest_descriptions_bulk(base_url: str, expected_mode: str, expected_tool: bool):
    _prepare_script_pipeline(base_url)
    response = _post(
        base_url,
        "/api/voices/suggest_descriptions_bulk",
        json={"speakers": ["NARRATOR", "ELENA", "MARCUS"]},
        timeout=240,
    )
    _assert_status(response, 200, "voice suggest descriptions bulk")
    payload = response.json()
    failures = payload.get("failures") or []
    if failures:
        raise AssertionError(f"voices/suggest_descriptions_bulk returned failures: {failures}")
    results = payload.get("results") or []
    if not results:
        raise AssertionError("voices/suggest_descriptions_bulk returned no results")
    rows = [
        (str(item.get("llm_mode") or ""), bool(item.get("llm_tool_call_observed")))
        for item in results
    ]
    _assert_mode_telemetry(
        "voices/suggest_descriptions_bulk",
        rows,
        expected_mode=expected_mode,
        expected_tool_call_observed=expected_tool,
    )


@dataclass(frozen=True)
class _Scenario:
    name: str
    fn: Callable


SCENARIOS = [
    _Scenario("generate_script", _scenario_generate_script),
    _Scenario("review_script", _scenario_review_script),
    _Scenario("script_sanity_check", _scenario_script_sanity_check),
    _Scenario("assign_dialogue", _scenario_assign_dialogue),
    _Scenario("extract_temperament", _scenario_extract_temperament),
    _Scenario("voices_suggest_description", _scenario_voice_suggest_description),
    _Scenario("voices_suggest_descriptions_bulk", _scenario_voice_suggest_descriptions_bulk),
]


def _execute_scenario_once(
    scenario: _Scenario,
    model_name: str,
    expected_mode: str,
    expected_tool: bool,
):
    with _IsolatedServer() as server:
        _configure_llm_for_phase(server.base_url, model_name, server.app_dir)
        scenario.fn(server.base_url, expected_mode, expected_tool)


def _execute_scenario_with_retry(
    scenario: _Scenario,
    model_name: str,
    expected_mode: str,
    expected_tool: bool,
):
    last_error = None
    for attempt in range(MAX_RETRIES_PER_SCENARIO + 1):
        try:
            _execute_scenario_once(
                scenario,
                model_name,
                expected_mode,
                expected_tool,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES_PER_SCENARIO:
                raise
    raise AssertionError(f"Scenario {scenario.name} failed: {last_error}")


def _run_phase(model_name: str, expected_mode: str, expected_tool: bool, context_length: int):
    _load_lmstudio_model_once(model_name, context_length)
    failures = []
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as executor:
        futures = {
            executor.submit(
                _execute_scenario_with_retry,
                scenario,
                model_name,
                expected_mode,
                expected_tool,
            ): scenario.name
            for scenario in SCENARIOS
        }
        for future in as_completed(futures):
            scenario_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((scenario_name, exc))
    if failures:
        details = "\n".join(f"- {name}: {type(exc).__name__}: {exc}" for name, exc in failures)
        raise AssertionError(
            f"Phase failed for model '{model_name}' (expected_mode={expected_mode}, expected_tool={expected_tool}):\n{details}"
        )


def test_live_lmstudio_llm_endpoints_tool_and_json_fallback():
    with _global_live_test_run_guard():
        _probe_lmstudio_or_skip()

        _run_phase(
            model_name=TOOL_MODEL_NAME,
            expected_mode="tool",
            expected_tool=True,
            context_length=TOOL_MODEL_CONTEXT_LENGTH,
        )
        _run_phase(
            model_name=NON_TOOL_MODEL_NAME,
            expected_mode="json",
            expected_tool=False,
            context_length=NON_TOOL_MODEL_CONTEXT_LENGTH,
        )
