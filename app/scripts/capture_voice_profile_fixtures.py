#!/usr/bin/env python3
"""Capture LM Studio voice suggestions + local Qwen voice-design outputs for E2E fixtures."""

from __future__ import annotations

import argparse
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
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Tuple

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from e2e_sim import LMStudioSimServer  # noqa: E402
from llm import LMStudioModelLoadService  # noqa: E402
from runtime_layout import RuntimeLayout  # noqa: E402


LMSTUDIO_DEFAULT_BASE = "http://127.0.0.1:1234"
LLM_MODEL_DEFAULT = "qwen/qwen3.5-9b"
BOOK_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "files", "test_book.epub")
SCRIPT_SEED_FIXTURE_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_generate_script_test_book.json"
)
LM_FIXTURE_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_voice_profiles_test_book.json"
)
QWEN_FIXTURE_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_voice_profiles_test_book.json"
)
ASSET_DIR_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "audio", "voice_profiles_test_book"
)
MANIFEST_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "voice_profiles_test_book_manifest.json"
)
WORKLOG_PATH = os.path.join(REPO_ROOT, "wiki", "Voice-Profile-Fixture-Worklog.md")


@dataclass
class CapturedVoiceProfile:
    speaker: str
    description: str
    sample_text: str
    generated_ref_text: str
    ref_audio: str
    fixture_audio_path: str
    audio_sha256: str
    audio_size: int
    sample_rate: int


def _assert_status(response: requests.Response, expected: int, context: str) -> None:
    if response.status_code != expected:
        raise RuntimeError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={response.text[:1200]}"
        )


def _normalize_base_url(url: str) -> str:
    raw = str(url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("LM Studio base URL is required")
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _resolve_capture_backend(preference: str) -> str:
    requested = str(preference or "auto").strip().lower()
    if requested not in {"auto", "qwen", "mlx"}:
        requested = "auto"
    if requested == "qwen":
        if not _module_available("qwen_tts"):
            raise RuntimeError("Requested tts local backend 'qwen', but qwen_tts is not installed")
        return "qwen"
    if requested == "mlx":
        if not _module_available("mlx_audio"):
            raise RuntimeError("Requested tts local backend 'mlx', but mlx_audio is not installed")
        return "mlx"
    if _module_available("qwen_tts"):
        return "qwen"
    if _module_available("mlx_audio"):
        return "mlx"
    raise RuntimeError("No supported local Qwen backend is available (missing qwen_tts and mlx_audio)")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("_") or "voice"


def _copy_prompt_files(temp_root: str) -> None:
    for filename in (
        "default_prompts.txt",
        "review_prompts.txt",
        "attribution_prompts.txt",
        "voice_prompt.txt",
        "dialogue_identification_system_prompt.txt",
        "temperament_extraction_system_prompt.txt",
    ):
        source = os.path.join(REPO_ROOT, "config", "prompts", filename)
        if not os.path.exists(source):
            continue
        prompt_dir = os.path.join(temp_root, "config", "prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        shutil.copy2(source, os.path.join(prompt_dir, filename))


class _IsolatedServer:
    def __init__(self, *, keep_temp: bool = False):
        self.keep_temp = bool(keep_temp)
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.app_dir = ""
        self.layout: RuntimeLayout | None = None

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_capture_voice_profiles_")
        self.app_dir = os.path.join(self._temp_root, "app")
        shutil.copytree(
            APP_DIR,
            self.app_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "env"),
        )
        _copy_prompt_files(self._temp_root)

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
                raise RuntimeError(
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

        raise RuntimeError(f"Timed out waiting for isolated server at {self.base_url}")

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
        if self._temp_root and os.path.isdir(self._temp_root) and not self.keep_temp:
            shutil.rmtree(self._temp_root, ignore_errors=True)


def _poll_task(base_url: str, task_name: str, timeout_seconds: int = 1200) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/{task_name}", timeout=30)
        _assert_status(response, 200, f"poll {task_name}")
        payload = response.json()
        if not payload.get("running"):
            return payload
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for '{task_name}'")


def _wait_for_new_mode_script_ready(base_url: str, timeout_seconds: int = 1800) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: Dict[str, Any] = {}
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/new_mode_workflow", timeout=30)
        _assert_status(response, 200, "poll new_mode_workflow")
        payload = dict(response.json() or {})
        last_payload = payload

        logs = [str(item) for item in (payload.get("logs") or [])]
        for line in logs:
            lower = line.lower()
            if line.startswith("ERROR:") or "failed with return code" in lower or "failed to call llm provider" in lower:
                raise RuntimeError(f"Non-legacy workflow failed: {line}")

        completed = {str(item) for item in (payload.get("completed_stages") or [])}
        if (
            not bool(payload.get("running"))
            and not bool(payload.get("paused"))
            and "create_script" in completed
        ):
            return payload
        time.sleep(2)

    raise RuntimeError(
        "Timed out waiting for non-legacy script workflow completion.\n"
        f"Last payload: {json.dumps(last_payload, ensure_ascii=False)}"
    )


def _sync_scripts_config(app_dir: str, setup_payload: Dict[str, Any], config_payload: Dict[str, Any]) -> None:
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


def _upload_book(base_url: str, source_book_abs: str) -> None:
    with open(source_book_abs, "rb") as handle:
        files = {"file": (os.path.basename(source_book_abs), io.BytesIO(handle.read()), "application/epub+zip")}
        response = requests.post(f"{base_url}/api/upload", files=files, timeout=120)
    _assert_status(response, 200, "upload source book")


def _ensure_lmstudio_model_loaded(base_url: str, model_name: str, api_key: str) -> None:
    service = LMStudioModelLoadService(timeout_seconds=240)
    payload = service.load_model(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        model_name=model_name,
        context_length=16384,
        echo_load_config=True,
    )
    status = str(payload.get("status") or "")
    if status != "loaded":
        raise RuntimeError(f"LM Studio model load returned unexpected status: {payload}")


def _build_lm_fixture(*, model_name: str, results: List[Dict[str, Any]], source_book: str) -> Dict[str, Any]:
    post_entries: List[Dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        speaker = str(result.get("speaker") or "").strip()
        voice = str(result.get("voice") or "").strip()
        post_entries.append(
            {
                "expect": {
                    "model": model_name,
                    "tool_choice": "required",
                },
                "response": {
                    "id": f"chatcmpl-voice-profile-{index}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": f"call-voice-profile-{index}",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_voice_description",
                                            "arguments": json.dumps({"voice": voice}, ensure_ascii=False),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 24,
                        "total_tokens": 124,
                    },
                },
                "metadata": {
                    "speaker": speaker,
                },
            }
        )

    return {
        "strict": True,
        "metadata": {
            "purpose": "Captured LM Studio voice suggestions for reusable voice profiles",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_book,
            "model_name": model_name,
            "speaker_count": len(results),
            "speakers": [str(item.get("speaker") or "").strip() for item in results],
        },
        "routes": {
            "GET /api/v1/models": [
                {
                    "response": {
                        "models": [
                            {
                                "key": model_name,
                                "display_name": model_name,
                                "loaded_instances": [{"id": model_name}],
                                "capabilities": {"trained_for_tool_use": True},
                            }
                        ]
                    }
                }
            ],
            "POST /v1/chat/completions": post_entries,
        },
    }


def _build_qwen_fixture(
    *,
    voices: List[CapturedVoiceProfile],
    qwen_fixture_path: str,
    source_book: str,
) -> Dict[str, Any]:
    fixture_dir = os.path.dirname(os.path.abspath(qwen_fixture_path))
    method_entries: List[Dict[str, Any]] = []
    for voice in voices:
        rel_asset = os.path.relpath(voice.fixture_audio_path, fixture_dir)
        method_entries.append(
            {
                # Keep replay deterministic by call order while avoiding brittle strict
                # string matching on long generated sample text fields.
                "audio_wav_path": rel_asset,
                "sample_rate": voice.sample_rate,
                "metadata": {
                    "speaker": voice.speaker,
                    "ref_audio": voice.ref_audio,
                    "sample_text": voice.sample_text,
                    "generated_ref_text": voice.generated_ref_text,
                    "description": voice.description,
                },
            }
        )

    return {
        "strict": True,
        "metadata": {
            "purpose": "Captured local Qwen voice-design outputs for reusable voice profiles",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_book,
            "speaker_count": len(voices),
            "speakers": [voice.speaker for voice in voices],
        },
        "methods": {
            "generate_voice_design": method_entries,
        },
    }


def _wav_info(path: str) -> Tuple[int, int]:
    with wave.open(path, "rb") as handle:
        return int(handle.getframerate()), int(handle.getnframes())


def _audio_sha256(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _update_worklog(
    *,
    speakers: List[str],
    lm_fixture_path: str,
    qwen_fixture_path: str,
    asset_dir: str,
) -> None:
    if not os.path.exists(WORKLOG_PATH):
        return
    with open(WORKLOG_PATH, "r", encoding="utf-8") as handle:
        content = handle.read()

    now = datetime.now(timezone.utc).isoformat()
    content = re.sub(r"- Status: .*", "- Status: complete", content)
    content = re.sub(r"- Last run timestamp: .*", f"- Last run timestamp: {now}", content)
    content = re.sub(r"- Speakers: .*", f"- Speakers: {', '.join(speakers) if speakers else 'none'}", content)
    content = re.sub(
        r"- LM fixture path: .*",
        f"- LM fixture path: `{os.path.relpath(lm_fixture_path, REPO_ROOT)}`",
        content,
    )
    content = re.sub(
        r"- Qwen fixture path: .*",
        f"- Qwen fixture path: `{os.path.relpath(qwen_fixture_path, REPO_ROOT)}`",
        content,
    )
    content = re.sub(
        r"- Asset directory: .*",
        f"- Asset directory: `{os.path.relpath(asset_dir, REPO_ROOT)}`",
        content,
    )
    content = re.sub(
        r"- Notes: .*",
        "- Notes: captured from isolated API workflow seeded from Generate Script fixture, then live LM suggestions + local Qwen design generation",
        content,
    )

    with open(WORKLOG_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def capture_voice_profile_fixtures(
    *,
    lmstudio_base_url: str,
    llm_model: str,
    api_key: str,
    source_book_path: str,
    script_seed_fixture: str,
    lm_fixture_output: str,
    qwen_fixture_output: str,
    asset_output_dir: str,
    manifest_output: str,
    tts_local_backend: str = "auto",
    voice_max_tokens: int = 256,
    keep_temp: bool = False,
) -> Dict[str, Any]:
    base = _normalize_base_url(lmstudio_base_url)
    source_book_abs = os.path.abspath(source_book_path)
    script_seed_abs = os.path.abspath(script_seed_fixture)
    if not os.path.exists(source_book_abs):
        raise FileNotFoundError(f"Source book not found: {source_book_abs}")
    if not os.path.exists(script_seed_abs):
        raise FileNotFoundError(f"Script seed fixture not found: {script_seed_abs}")

    print(f"[capture] Ensuring LM Studio model is loaded: {llm_model}")
    _ensure_lmstudio_model_loaded(base, llm_model, api_key)
    resolved_backend = _resolve_capture_backend(tts_local_backend)
    print(f"[capture] Using local TTS backend: {resolved_backend}")

    with _IsolatedServer(keep_temp=keep_temp) as server:
        if server.layout is None:
            raise RuntimeError("Isolated runtime layout unavailable")
        print(f"[capture] Isolated app started: {server.base_url}")

        _upload_book(server.base_url, source_book_abs)
        print("[capture] Uploaded source EPUB")

        with open(script_seed_abs, "r", encoding="utf-8") as handle:
            script_seed_payload = json.load(handle)
        script_seed_model = str(((script_seed_payload.get("metadata") or {}).get("model_name") or "").strip())
        if not script_seed_model:
            raise RuntimeError("Script seed fixture metadata.model_name is required")

        with LMStudioSimServer(script_seed_abs) as script_sim:
            print("[capture] Seeding script state from Generate Script fixture")
            setup_seed = {
                "llm": {
                    "base_url": f"{script_sim.base_url}/v1",
                    "api_key": "local",
                    "model_name": script_seed_model,
                    "llm_workers": 1,
                },
                "tts": {
                    "mode": "local",
                    "local_backend": resolved_backend,
                    "device": "auto",
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
                    "merge_narrators": False,
                    "orphaned_text_to_narrator_on_repair": True,
                    "legacy_mode": False,
                    "temperament_words": 150,
                },
            }
            response = requests.post(f"{server.base_url}/api/config/setup", json=setup_seed, timeout=60)
            _assert_status(response, 200, "configure script seed setup")
            config_response = requests.get(f"{server.base_url}/api/config", timeout=30)
            _assert_status(config_response, 200, "get config")
            _sync_scripts_config(server.app_dir, setup_seed, config_response.json())

            start_response = requests.post(
                f"{server.base_url}/api/new_mode_workflow/start",
                json={"process_voices": False, "generate_audio": False},
                timeout=60,
            )
            _assert_status(start_response, 200, "start non-legacy script workflow")
            _wait_for_new_mode_script_ready(server.base_url)
            script_sim.assert_all_consumed()
            print("[capture] Seed non-legacy script workflow completed")

        setup_live = {
            "llm": {
                "base_url": f"{base}/v1",
                "api_key": api_key,
                "model_name": llm_model,
                "llm_workers": 1,
            },
            "tts": {
                "mode": "local",
                "local_backend": resolved_backend,
                "device": "auto",
                "language": "English",
            },
            "generation": {
                "max_tokens": max(64, int(voice_max_tokens)),
            },
        }
        response = requests.post(f"{server.base_url}/api/config/setup", json=setup_live, timeout=60)
        _assert_status(response, 200, "configure live LLM + local qwen setup")
        print("[capture] Switched LLM config to live LM Studio for voice suggestions")

        voices_response = requests.get(f"{server.base_url}/api/voices", timeout=30)
        _assert_status(voices_response, 200, "get voices")
        voices_payload = voices_response.json()
        if not isinstance(voices_payload, list) or not voices_payload:
            raise RuntimeError("No voices available after seeded script generation")

        speakers = [str((item or {}).get("name") or "").strip() for item in voices_payload]
        speakers = [speaker for speaker in speakers if speaker]
        if not speakers:
            raise RuntimeError("No speaker names resolved from /api/voices")
        print(f"[capture] Speakers to capture: {', '.join(speakers)}")

        suggestion_response = requests.post(
            f"{server.base_url}/api/voices/suggest_descriptions_bulk",
            json={"speakers": speakers},
            timeout=900,
        )
        _assert_status(suggestion_response, 200, "suggest voice descriptions bulk")
        suggestion_payload = suggestion_response.json()
        suggestion_results = list(suggestion_payload.get("results") or [])
        suggestion_failures = list(suggestion_payload.get("failures") or [])
        if suggestion_failures:
            raise RuntimeError(f"Voice suggestion failures: {suggestion_failures}")
        if len(suggestion_results) != len(speakers):
            raise RuntimeError(
                f"Expected {len(speakers)} suggestion results, got {len(suggestion_results)}"
            )
        print(f"[capture] Captured {len(suggestion_results)} LM voice suggestion responses")

        voices_by_name: Dict[str, Dict[str, Any]] = {
            str((item or {}).get("name") or "").strip(): dict(item or {})
            for item in voices_payload
            if str((item or {}).get("name") or "").strip()
        }

        batch_config: Dict[str, Dict[str, Any]] = {}
        for suggestion in suggestion_results:
            speaker = str(suggestion.get("speaker") or "").strip()
            voice = str(suggestion.get("voice") or "").strip()
            sample_text = str(
                ((voices_by_name.get(speaker) or {}).get("suggested_sample_text") or "")
            ).strip()
            if not sample_text:
                raise RuntimeError(f"No suggested sample text for speaker '{speaker}'")
            batch_config[speaker] = {
                "type": "design",
                "description": voice,
                "ref_text": sample_text,
            }

        batch_response = requests.post(
            f"{server.base_url}/api/voices/batch",
            json={"config": batch_config, "confirm_invalidation": False},
            timeout=60,
        )
        _assert_status(batch_response, 200, "save suggested voice descriptions")

        asset_dir_abs = os.path.abspath(asset_output_dir)
        if os.path.isdir(asset_dir_abs):
            shutil.rmtree(asset_dir_abs, ignore_errors=True)
        os.makedirs(asset_dir_abs, exist_ok=True)

        captured_profiles: List[CapturedVoiceProfile] = []
        for index, suggestion in enumerate(suggestion_results, start=1):
            speaker = str(suggestion.get("speaker") or "").strip()
            description = str(suggestion.get("voice") or "").strip()
            sample_text = str((batch_config.get(speaker) or {}).get("ref_text") or "").strip()
            if not speaker or not description or not sample_text:
                raise RuntimeError(f"Incomplete voice capture input for item {index}: {suggestion}")

            print(f"[capture] [{index}/{len(suggestion_results)}] Generating reusable voice for {speaker}")
            generate_response = requests.post(
                f"{server.base_url}/api/voices/design_generate",
                json={
                    "speaker": speaker,
                    "description": description,
                    "sample_text": sample_text,
                    "force": False,
                },
                timeout=1800,
            )
            _assert_status(generate_response, 200, f"design generate ({speaker})")
            generated = generate_response.json()

            ref_audio = str(generated.get("ref_audio") or "").strip()
            generated_ref_text = str(generated.get("generated_ref_text") or "").strip() or sample_text
            if not ref_audio:
                raise RuntimeError(f"Missing ref_audio for generated voice '{speaker}'")

            source_audio_path = os.path.join(server.layout.project_dir, ref_audio)
            if not os.path.exists(source_audio_path):
                raise RuntimeError(f"Generated audio not found for '{speaker}': {source_audio_path}")

            target_filename = f"{index:02d}_{_sanitize_name(speaker)}.wav"
            copied_audio_path = os.path.join(asset_dir_abs, target_filename)
            shutil.copy2(source_audio_path, copied_audio_path)

            sample_rate, _ = _wav_info(copied_audio_path)
            captured_profiles.append(
                CapturedVoiceProfile(
                    speaker=speaker,
                    description=description,
                    sample_text=sample_text,
                    generated_ref_text=generated_ref_text,
                    ref_audio=ref_audio,
                    fixture_audio_path=copied_audio_path,
                    audio_sha256=_audio_sha256(copied_audio_path),
                    audio_size=os.path.getsize(copied_audio_path),
                    sample_rate=sample_rate,
                )
            )
            print(
                f"[capture] [{index}/{len(suggestion_results)}] Captured audio for {speaker}: "
                f"{os.path.basename(copied_audio_path)} ({os.path.getsize(copied_audio_path)} bytes)"
            )

        lm_fixture = _build_lm_fixture(
            model_name=llm_model,
            results=suggestion_results,
            source_book=os.path.relpath(source_book_abs, REPO_ROOT),
        )
        qwen_fixture = _build_qwen_fixture(
            voices=captured_profiles,
            qwen_fixture_path=qwen_fixture_output,
            source_book=os.path.relpath(source_book_abs, REPO_ROOT),
        )

        lm_fixture_abs = os.path.abspath(lm_fixture_output)
        qwen_fixture_abs = os.path.abspath(qwen_fixture_output)
        manifest_abs = os.path.abspath(manifest_output)
        os.makedirs(os.path.dirname(lm_fixture_abs), exist_ok=True)
        os.makedirs(os.path.dirname(qwen_fixture_abs), exist_ok=True)
        os.makedirs(os.path.dirname(manifest_abs), exist_ok=True)

        with open(lm_fixture_abs, "w", encoding="utf-8") as handle:
            json.dump(lm_fixture, handle, ensure_ascii=False, indent=2)
        with open(qwen_fixture_abs, "w", encoding="utf-8") as handle:
            json.dump(qwen_fixture, handle, ensure_ascii=False, indent=2)

        manifest_payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_book": os.path.relpath(source_book_abs, REPO_ROOT),
            "llm_model": llm_model,
            "tts_local_backend": resolved_backend,
            "speakers": [voice.speaker for voice in captured_profiles],
            "voice_profiles": [
                {
                    "speaker": voice.speaker,
                    "description": voice.description,
                    "sample_text": voice.sample_text,
                    "generated_ref_text": voice.generated_ref_text,
                    "ref_audio": voice.ref_audio,
                    "fixture_audio_path": os.path.relpath(voice.fixture_audio_path, REPO_ROOT),
                    "audio_sha256": voice.audio_sha256,
                    "audio_size": voice.audio_size,
                    "sample_rate": voice.sample_rate,
                }
                for voice in captured_profiles
            ],
            "lm_fixture_path": os.path.relpath(lm_fixture_abs, REPO_ROOT),
            "qwen_fixture_path": os.path.relpath(qwen_fixture_abs, REPO_ROOT),
            "asset_dir": os.path.relpath(asset_dir_abs, REPO_ROOT),
        }
        with open(manifest_abs, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)
        print("[capture] Wrote fixtures + manifest")

        _update_worklog(
            speakers=[voice.speaker for voice in captured_profiles],
            lm_fixture_path=lm_fixture_abs,
            qwen_fixture_path=qwen_fixture_abs,
            asset_dir=asset_dir_abs,
        )

        try:
            requests.post(f"{server.base_url}/api/voices/unload_bulk_generation", json={}, timeout=30)
        except Exception:
            pass

        return {
            "status": "ok",
            "speakers": [voice.speaker for voice in captured_profiles],
            "llm_model": llm_model,
            "tts_local_backend": resolved_backend,
            "lm_fixture_path": lm_fixture_abs,
            "qwen_fixture_path": qwen_fixture_abs,
            "asset_dir": asset_dir_abs,
            "manifest_path": manifest_abs,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture LM voice suggestions + local Qwen voice-profile assets for E2E fixtures"
    )
    parser.add_argument("--lmstudio-base-url", default=LMSTUDIO_DEFAULT_BASE)
    parser.add_argument("--llm-model", default=LLM_MODEL_DEFAULT)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--source-book", default=BOOK_DEFAULT)
    parser.add_argument("--script-seed-fixture", default=SCRIPT_SEED_FIXTURE_DEFAULT)
    parser.add_argument("--lm-output", default=LM_FIXTURE_DEFAULT)
    parser.add_argument("--qwen-output", default=QWEN_FIXTURE_DEFAULT)
    parser.add_argument("--asset-dir", default=ASSET_DIR_DEFAULT)
    parser.add_argument("--manifest-output", default=MANIFEST_DEFAULT)
    parser.add_argument("--tts-local-backend", default="auto", choices=["auto", "qwen", "mlx"])
    parser.add_argument("--voice-max-tokens", type=int, default=256)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = capture_voice_profile_fixtures(
        lmstudio_base_url=args.lmstudio_base_url,
        llm_model=args.llm_model,
        api_key=args.api_key,
        source_book_path=args.source_book,
        script_seed_fixture=args.script_seed_fixture,
        lm_fixture_output=args.lm_output,
        qwen_fixture_output=args.qwen_output,
        asset_output_dir=args.asset_dir,
        manifest_output=args.manifest_output,
        tts_local_backend=args.tts_local_backend,
        voice_max_tokens=int(args.voice_max_tokens),
        keep_temp=bool(args.keep_temp),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
