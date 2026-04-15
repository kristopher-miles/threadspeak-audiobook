#!/usr/bin/env python3
"""Capture real editor-phase per-line audio clips and build Qwen replay fixtures."""

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
from pydub import AudioSegment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from e2e_sim import LMStudioSimServer  # noqa: E402
from runtime_layout import RuntimeLayout  # noqa: E402
from script_store import apply_dictionary_to_text  # noqa: E402


BOOK_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "files", "test_book.epub")
SCRIPT_SEED_FIXTURE_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_generate_script_test_book.json"
)
VOICE_PROFILE_MANIFEST_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "voice_profiles_test_book_manifest.json"
)
QWEN_FIXTURE_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_editor_audio_test_book.json"
)
ASSET_DIR_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "audio", "editor_audio_test_book"
)
MANIFEST_DEFAULT = os.path.join(
    APP_DIR, "test_fixtures", "e2e_sim", "editor_audio_test_book_manifest.json"
)
WORKLOG_PATH = os.path.join(REPO_ROOT, "wiki", "Editor-Audio-Fixture-Worklog.md")


@dataclass
class CapturedLine:
    uid: str
    line_id: int
    speaker: str
    text: str
    instruct: str
    transformed_text: str
    source_audio_path: str
    fixture_audio_path: str
    sample_rate: int
    frames: int
    duration_seconds: float
    audio_size: int
    audio_sha256: str


def _assert_status(response: requests.Response, expected: int, context: str) -> None:
    if response.status_code != expected:
        raise RuntimeError(
            f"{context} expected HTTP {expected}, got {response.status_code}. body={response.text[:1200]}"
        )


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
    return cleaned.strip("_") or "line"


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


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
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_capture_editor_audio_")
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


def _wait_for_audio_idle(base_url: str, timeout_seconds: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/audio", timeout=30)
        _assert_status(response, 200, "poll audio status")
        payload = response.json()
        if (
            not bool(payload.get("running"))
            and not list(payload.get("queue") or [])
            and not payload.get("current_job")
        ):
            return payload
        time.sleep(2)
    raise RuntimeError("Timed out waiting for /api/status/audio to become idle")


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


def _seed_script_from_fixture(
    *,
    base_url: str,
    app_dir: str,
    script_seed_fixture_abs: str,
    tts_local_backend: str,
) -> str:
    with open(script_seed_fixture_abs, "r", encoding="utf-8") as handle:
        script_fixture = json.load(handle)
    model_name = str(((script_fixture.get("metadata") or {}).get("model_name") or "").strip())
    if not model_name:
        raise RuntimeError("Script seed fixture metadata.model_name is required")

    with LMStudioSimServer(script_seed_fixture_abs) as script_sim:
        setup_payload = {
            "llm": {
                "base_url": f"{script_sim.base_url}/v1",
                "api_key": "local",
                "model_name": model_name,
                "llm_workers": 1,
            },
            "tts": {
                "mode": "local",
                "local_backend": tts_local_backend,
                "device": "auto",
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
                "merge_narrators": False,
                "orphaned_text_to_narrator_on_repair": True,
                "legacy_mode": False,
                "temperament_words": 150,
            },
        }
        response = requests.post(f"{base_url}/api/config/setup", json=setup_payload, timeout=60)
        _assert_status(response, 200, "configure script seed setup")
        config_response = requests.get(f"{base_url}/api/config", timeout=30)
        _assert_status(config_response, 200, "get config")
        _sync_scripts_config(app_dir, setup_payload, config_response.json())

        start_response = requests.post(
            f"{base_url}/api/new_mode_workflow/start",
            json={"process_voices": False, "generate_audio": False},
            timeout=60,
        )
        _assert_status(start_response, 200, "start non-legacy script workflow")
        _wait_for_new_mode_script_ready(base_url)
        script_sim.assert_all_consumed()

    return model_name


def _seed_clone_voices_from_manifest(
    *,
    base_url: str,
    project_dir: str,
    voice_manifest_abs: str,
) -> Dict[str, Dict[str, Any]]:
    with open(voice_manifest_abs, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    profiles = [dict(item or {}) for item in (payload.get("voice_profiles") or [])]
    if not profiles:
        raise RuntimeError("Voice profile manifest has no voice_profiles entries")

    voice_rows_response = requests.get(f"{base_url}/api/voices", timeout=60)
    _assert_status(voice_rows_response, 200, "get voices")
    voice_rows = list(voice_rows_response.json() or [])
    existing_by_norm = {
        _normalize_name(str((row or {}).get("name") or "")): dict(row or {})
        for row in voice_rows
        if _normalize_name(str((row or {}).get("name") or ""))
    }

    profile_by_norm = {
        _normalize_name(str(item.get("speaker") or "")): item
        for item in profiles
        if _normalize_name(str(item.get("speaker") or ""))
    }

    missing_profiles = [
        str((row or {}).get("name") or "")
        for row in voice_rows
        if _normalize_name(str((row or {}).get("name") or "")) not in profile_by_norm
    ]
    if missing_profiles:
        raise RuntimeError(f"Voice profile manifest is missing speakers: {missing_profiles}")

    output_config: Dict[str, Dict[str, Any]] = {}
    seeded_profiles: Dict[str, Dict[str, Any]] = {}
    for normalized_name, row in existing_by_norm.items():
        speaker = str((row or {}).get("name") or "").strip()
        profile = profile_by_norm[normalized_name]
        ref_audio = str(profile.get("ref_audio") or "").strip()
        source_asset_rel = str(profile.get("fixture_audio_path") or "").strip()
        source_asset_abs = source_asset_rel if os.path.isabs(source_asset_rel) else os.path.join(REPO_ROOT, source_asset_rel)
        if not os.path.exists(source_asset_abs):
            raise FileNotFoundError(f"Voice fixture audio not found: {source_asset_abs}")
        if not ref_audio:
            raise RuntimeError(f"Voice profile for '{speaker}' is missing ref_audio")

        dest_audio_abs = ref_audio if os.path.isabs(ref_audio) else os.path.join(project_dir, ref_audio)
        os.makedirs(os.path.dirname(dest_audio_abs), exist_ok=True)
        shutil.copy2(source_asset_abs, dest_audio_abs)

        merged_config = dict((row or {}).get("config") or {})
        merged_config.update(
            {
                "type": "clone",
                "ref_audio": ref_audio,
                "ref_text": str(profile.get("sample_text") or "").strip(),
                "generated_ref_text": str(profile.get("generated_ref_text") or "").strip(),
                "description": str(profile.get("description") or "").strip(),
            }
        )
        output_config[speaker] = merged_config
        seeded_profiles[speaker] = profile

    save_response = requests.post(
        f"{base_url}/api/voices/batch",
        json={"config": output_config, "confirm_invalidation": False},
        timeout=60,
    )
    _assert_status(save_response, 200, "seed clone voice config")

    return seeded_profiles


def _update_chunk_fields(base_url: str, uid: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{base_url}/api/chunks/{uid}", json=fields, timeout=60)
    _assert_status(response, 200, f"update chunk {uid}")
    return dict(response.json() or {})


def _cancel_audio(base_url: str) -> None:
    try:
        requests.post(f"{base_url}/api/cancel_audio", json={}, timeout=30)
    except Exception:
        pass


def _unload_tts_engine(base_url: str) -> None:
    try:
        requests.post(f"{base_url}/api/voices/unload_bulk_generation", json={}, timeout=60)
    except Exception:
        pass


def _audio_ref_is_servable(base_url: str, audio_ref: str) -> bool:
    path = str(audio_ref or "").strip().lstrip("/")
    if not path:
        return False
    try:
        response = requests.get(f"{base_url}/{path}", timeout=30)
    except Exception:
        return False
    if response.status_code != 200:
        return False
    return bool(response.content)


def _generate_chunk_once(
    base_url: str,
    uid: str,
    *,
    regenerate: bool,
    timeout_seconds: int,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    max_attempts = max(1, int(max_attempts))
    last_error: RuntimeError | None = None

    for attempt in range(1, max_attempts + 1):
        action = "regenerate" if (regenerate or attempt > 1) else "generate"
        response = requests.post(f"{base_url}/api/chunks/{uid}/{action}", timeout=60)
        _assert_status(response, 200, f"{action} chunk {uid}")

        audio_status = _wait_for_audio_idle(base_url, timeout_seconds=timeout_seconds)
        recent_jobs = list(audio_status.get("recent_jobs") or [])
        if not recent_jobs:
            raise RuntimeError(f"Chunk {uid} generation ended without recent_jobs state")
        latest_job = dict(recent_jobs[0] or {})
        latest_status = str(latest_job.get("status") or "").strip().lower()

        chunk_response = requests.get(f"{base_url}/api/chunks/{uid}", timeout=60)
        _assert_status(chunk_response, 200, f"get chunk {uid} after {action}")
        chunk = dict(chunk_response.json() or {})
        chunk_status = str(chunk.get("status") or "").strip().lower()
        audio_ref = str(chunk.get("audio_path") or "").strip()
        if audio_ref and (chunk_status == "done" or _audio_ref_is_servable(base_url, audio_ref)):
            return chunk

        last_error = RuntimeError(
            f"Chunk {uid} generation attempt {attempt}/{max_attempts} did not produce usable audio. "
            f"job_status={latest_status} chunk_status={chunk_status} audio_path={audio_ref!r} latest_job={latest_job}"
        )
        if attempt >= max_attempts:
            break
        _cancel_audio(base_url)
        time.sleep(0.25)
        print(f"[capture] Retry chunk {uid} after non-clean completion ({latest_status})")

    raise last_error or RuntimeError(f"Chunk {uid} generation failed without detailed error")


def _segment_text_for_capture(text: str, max_words: int = 35) -> List[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return [""]
    words = normalized.split()
    if len(words) <= max_words:
        return [normalized]

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if len(sentence_parts) > 1:
        segmented: List[str] = []
        for part in sentence_parts:
            part_words = part.split()
            if len(part_words) <= max_words:
                segmented.append(part)
                continue
            for index in range(0, len(part_words), max_words):
                segmented.append(" ".join(part_words[index:index + max_words]))
        return segmented

    segmented = []
    for index in range(0, len(words), max_words):
        segmented.append(" ".join(words[index:index + max_words]))
    return segmented


def _combine_audio_segments(segment_paths: List[str], target_wav_path: str) -> None:
    if not segment_paths:
        raise RuntimeError("No segment audio paths were provided for combination")
    if len(segment_paths) == 1:
        _copy_or_convert_to_wav(segment_paths[0], target_wav_path)
        return

    merged = AudioSegment.empty()
    for index, path in enumerate(segment_paths):
        merged += AudioSegment.from_file(path)
        if index < len(segment_paths) - 1:
            merged += AudioSegment.silent(duration=140)
    os.makedirs(os.path.dirname(target_wav_path), exist_ok=True)
    merged.export(target_wav_path, format="wav")


def _copy_or_convert_to_wav(source_abs: str, target_abs: str) -> None:
    os.makedirs(os.path.dirname(target_abs), exist_ok=True)
    if source_abs.lower().endswith(".wav"):
        shutil.copy2(source_abs, target_abs)
        return
    segment = AudioSegment.from_file(source_abs)
    segment.export(target_abs, format="wav")


def _wav_info(path: str) -> Tuple[int, int, float]:
    with wave.open(path, "rb") as handle:
        sample_rate = int(handle.getframerate())
        frames = int(handle.getnframes())
    duration = (frames / sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, frames, duration


def _audio_sha256(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_qwen_fixture(
    *,
    lines: List[CapturedLine],
    voice_profiles: Dict[str, Dict[str, Any]],
    qwen_fixture_path: str,
    source_book: str,
    tts_local_backend: str,
) -> Dict[str, Any]:
    fixture_dir = os.path.dirname(os.path.abspath(qwen_fixture_path))
    if not lines:
        raise RuntimeError("Cannot build Qwen fixture without captured lines")

    first_speaker = lines[0].speaker
    profile = voice_profiles.get(first_speaker)
    if not profile:
        raise RuntimeError(f"Missing seeded voice profile for speaker '{first_speaker}'")
    ref_text = str(profile.get("generated_ref_text") or profile.get("sample_text") or "").strip()
    prompt_entries: List[Dict[str, Any]] = [
        {
            "expect": {
                "has_ref_audio": True,
                "ref_text": ref_text,
            },
            "prompt_ref_text": ref_text,
            "prompt_tokens": max(1, len(ref_text) // 4),
            "metadata": {
                "speaker": first_speaker,
                "ref_audio": str(profile.get("ref_audio") or "").strip(),
            },
        }
    ]

    generation_entries: List[Dict[str, Any]] = []
    for line in lines:
        rel_asset = os.path.relpath(line.fixture_audio_path, fixture_dir)
        generation_entries.append(
            {
                "expect": {
                    "text": line.transformed_text,
                    "has_prompt": True,
                },
                "audio_wav_path": rel_asset,
                "sample_rate": line.sample_rate,
                "metadata": {
                    "uid": line.uid,
                    "line_id": line.line_id,
                    "speaker": line.speaker,
                    "source_audio_path": line.source_audio_path,
                },
            }
        )

    return {
        "strict": True,
        "unordered_methods": ["generate_voice_clone"],
        "metadata": {
            "purpose": "Captured editor-phase local Qwen outputs for full line-by-line generation replay",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_book,
            "tts_local_backend": tts_local_backend,
            "line_count": len(lines),
            "speaker_count": len(prompt_entries),
            "speakers": [entry["metadata"]["speaker"] for entry in prompt_entries],
        },
        "methods": {
            "create_voice_clone_prompt": prompt_entries,
            "generate_voice_clone": generation_entries,
        },
    }


def _update_worklog(
    *,
    line_count: int,
    speaker_count: int,
    backend: str,
    failed_count: int,
    qwen_fixture_path: str,
    manifest_path: str,
    asset_dir: str,
) -> None:
    if not os.path.exists(WORKLOG_PATH):
        return

    with open(WORKLOG_PATH, "r", encoding="utf-8") as handle:
        content = handle.read()

    now = datetime.now(timezone.utc).isoformat()
    status_text = "complete" if int(failed_count or 0) == 0 else f"partial ({int(failed_count)} failed lines)"
    content = re.sub(r"- Status: .*", f"- Status: {status_text}", content)
    content = re.sub(r"- Last run timestamp: .*", f"- Last run timestamp: {now}", content)
    content = re.sub(r"- Backend used: .*", f"- Backend used: {backend}", content)
    content = re.sub(r"- Captured lines: .*", f"- Captured lines: {line_count}", content)
    content = re.sub(r"- Captured speakers: .*", f"- Captured speakers: {speaker_count}", content)
    content = re.sub(
        r"- Qwen fixture path: .*",
        f"- Qwen fixture path: `{os.path.relpath(qwen_fixture_path, REPO_ROOT)}`",
        content,
    )
    content = re.sub(
        r"- Manifest path: .*",
        f"- Manifest path: `{os.path.relpath(manifest_path, REPO_ROOT)}`",
        content,
    )
    content = re.sub(
        r"- Asset directory: .*",
        f"- Asset directory: `{os.path.relpath(asset_dir, REPO_ROOT)}`",
        content,
    )

    with open(WORKLOG_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def capture_editor_audio_fixtures(
    *,
    source_book_path: str,
    script_seed_fixture: str,
    voice_profile_manifest: str,
    qwen_fixture_output: str,
    asset_output_dir: str,
    manifest_output: str,
    tts_local_backend: str = "auto",
    keep_temp: bool = False,
) -> Dict[str, Any]:
    source_book_abs = os.path.abspath(source_book_path)
    script_seed_abs = os.path.abspath(script_seed_fixture)
    voice_manifest_abs = os.path.abspath(voice_profile_manifest)

    if not os.path.exists(source_book_abs):
        raise FileNotFoundError(f"Source book not found: {source_book_abs}")
    if not os.path.exists(script_seed_abs):
        raise FileNotFoundError(f"Script seed fixture not found: {script_seed_abs}")
    if not os.path.exists(voice_manifest_abs):
        raise FileNotFoundError(f"Voice profile manifest not found: {voice_manifest_abs}")

    resolved_backend = _resolve_capture_backend(tts_local_backend)
    print(f"[capture] Using local TTS backend: {resolved_backend}")

    asset_dir_abs = os.path.abspath(asset_output_dir)
    if os.path.isdir(asset_dir_abs):
        shutil.rmtree(asset_dir_abs, ignore_errors=True)
    os.makedirs(asset_dir_abs, exist_ok=True)

    session_size = 12
    session_offset = 0
    script_seed_model = ""
    seeded_profiles: Dict[str, Dict[str, Any]] | None = None
    line_ids: List[int] = []
    canonical_by_line_id: Dict[int, Dict[str, Any]] = {}
    captured_by_line_id: Dict[int, CapturedLine] = {}
    failed_line_errors: Dict[int, Dict[str, Any]] = {}

    while True:
        with _IsolatedServer(keep_temp=keep_temp) as server:
            if server.layout is None:
                raise RuntimeError("Isolated runtime layout unavailable")
            print(f"[capture] Isolated app started: {server.base_url}")

            _upload_book(server.base_url, source_book_abs)
            print("[capture] Uploaded source EPUB")

            script_seed_model = _seed_script_from_fixture(
                base_url=server.base_url,
                app_dir=server.app_dir,
                script_seed_fixture_abs=script_seed_abs,
                tts_local_backend=resolved_backend,
            )
            print("[capture] Seeded script state from fixture")

            seeded_profiles = _seed_clone_voices_from_manifest(
                base_url=server.base_url,
                project_dir=server.layout.project_dir,
                voice_manifest_abs=voice_manifest_abs,
            )
            print(f"[capture] Seeded {len(seeded_profiles)} clone voice profiles from manifest")

            setup_generation = {
                "llm": {
                    "base_url": "http://127.0.0.1:1234/v1",
                    "api_key": "local",
                    "model_name": script_seed_model,
                    "llm_workers": 1,
                },
                "tts": {
                    "mode": "local",
                    "local_backend": resolved_backend,
                    "device": "auto",
                    "language": "English",
                    "parallel_workers": 1,
                },
                "generation": {
                    "max_tokens": 1024,
                },
            }
            response = requests.post(f"{server.base_url}/api/config/setup", json=setup_generation, timeout=60)
            _assert_status(response, 200, "configure editor generation setup")
            threshold_response = requests.post(
                f"{server.base_url}/api/voices/settings",
                json={"value": 0},
                timeout=60,
            )
            _assert_status(threshold_response, 200, "set narrator threshold to 0 for capture")

            chunks_response = requests.get(f"{server.base_url}/api/chunks", timeout=60)
            _assert_status(chunks_response, 200, "load chunks before generation")
            target_rows = [
                dict(chunk or {})
                for chunk in (chunks_response.json() or [])
                if str((chunk or {}).get("uid") or "").strip()
                and str((chunk or {}).get("text") or "").strip()
            ]
            target_rows.sort(key=lambda chunk: int(chunk.get("id") or 0))
            if not target_rows:
                raise RuntimeError("No non-empty chunks were found before editor generation")

            if not line_ids:
                line_ids = [int(chunk.get("id") or 0) for chunk in target_rows]
                canonical_by_line_id = {int(chunk.get("id") or 0): dict(chunk) for chunk in target_rows}
                print(f"[capture] Generating editor audio for {len(line_ids)} lines")
            else:
                current_ids = [int(chunk.get("id") or 0) for chunk in target_rows]
                if current_ids != line_ids:
                    raise RuntimeError("Chunk line ids changed between capture sessions")

            session_line_ids = line_ids[session_offset:session_offset + session_size]
            if not session_line_ids:
                break
            print(
                f"[capture] Session lines: {session_line_ids[0]}-{session_line_ids[-1]} "
                f"({len(session_line_ids)} lines)"
            )

            by_line_id = {int(chunk.get("id") or 0): dict(chunk) for chunk in target_rows}

            for line_id in session_line_ids:
                row = by_line_id[line_id]
                uid = str(row.get("uid") or "").strip()
                original_text = str(row.get("text") or "")
                original_instruct = str((canonical_by_line_id.get(line_id) or {}).get("instruct") or "")
                speaker = str(row.get("speaker") or "").strip()
                try:
                    _update_chunk_fields(server.base_url, uid, {"instruct": original_instruct})
                    dictionary_response = requests.get(f"{server.base_url}/api/dictionary", timeout=30)
                    _assert_status(dictionary_response, 200, "load dictionary")
                    dictionary_entries = list((dictionary_response.json() or {}).get("entries") or [])

                    segment_texts = _segment_text_for_capture(original_text, max_words=35)
                    segment_audio_paths: List[str] = []
                    audio_ref_parts: List[str] = []
                    changed_text = len(segment_texts) > 1
                    if changed_text:
                        print(
                            f"[capture] Line {line_id}/{line_ids[-1]} ({uid}) uses segmented generation: "
                            f"{len(segment_texts)} segments"
                        )

                    try:
                        generated_any = False
                        segment_queue = list(segment_texts)
                        while segment_queue:
                            segment_text = segment_queue.pop(0)
                            if changed_text:
                                _update_chunk_fields(server.base_url, uid, {"text": segment_text})
                            try:
                                generated_chunk = _generate_chunk_once(
                                    server.base_url,
                                    uid,
                                    regenerate=generated_any,
                                    timeout_seconds=300,
                                )
                            except RuntimeError as exc:
                                if (
                                    "Timed out waiting for /api/status/audio to become idle" in str(exc)
                                    and len(segment_text.split()) > 1
                                ):
                                    _cancel_audio(server.base_url)
                                    tighter_limit = max(1, len(segment_text.split()) // 2)
                                    fallback_segments = _segment_text_for_capture(segment_text, max_words=tighter_limit)
                                    if len(fallback_segments) <= 1:
                                        raise
                                    changed_text = True
                                    print(
                                        f"[capture] Segment timeout for {uid}; retrying with "
                                        f"{len(fallback_segments)} smaller segment(s)"
                                    )
                                    segment_queue = fallback_segments + segment_queue
                                    continue
                                raise
                            audio_ref = str(generated_chunk.get("audio_path") or "").strip()
                            source_audio_abs = (
                                audio_ref if os.path.isabs(audio_ref) else os.path.join(server.layout.project_dir, audio_ref)
                            )
                            if not os.path.exists(source_audio_abs):
                                raise RuntimeError(f"Generated audio file missing: {source_audio_abs}")
                            segment_audio_paths.append(source_audio_abs)
                            audio_ref_parts.append(audio_ref)
                            generated_any = True
                    finally:
                        if changed_text:
                            _update_chunk_fields(server.base_url, uid, {"text": original_text})

                    fixture_filename = f"{line_id:04d}_{_sanitize_name(speaker)}.wav"
                    fixture_audio_abs = os.path.join(asset_dir_abs, fixture_filename)
                    _combine_audio_segments(segment_audio_paths, fixture_audio_abs)
                    sample_rate, frames, duration_seconds = _wav_info(fixture_audio_abs)
                    transformed_text = apply_dictionary_to_text(original_text, dictionary_entries)[0]

                    captured_by_line_id[line_id] = CapturedLine(
                        uid=uid,
                        line_id=line_id,
                        speaker=speaker,
                        text=original_text,
                        instruct=original_instruct,
                        transformed_text=str(transformed_text or ""),
                        source_audio_path=" | ".join(audio_ref_parts),
                        fixture_audio_path=fixture_audio_abs,
                        sample_rate=sample_rate,
                        frames=frames,
                        duration_seconds=duration_seconds,
                        audio_size=os.path.getsize(fixture_audio_abs),
                        audio_sha256=_audio_sha256(fixture_audio_abs),
                    )
                    _unload_tts_engine(server.base_url)
                except Exception as exc:
                    _cancel_audio(server.base_url)
                    failed_line_errors[line_id] = {
                        "line_id": line_id,
                        "uid": uid,
                        "speaker": speaker,
                        "text": original_text,
                        "error": str(exc),
                    }
                    print(f"[capture] FAILED line {line_id} ({uid}, {speaker}): {exc}")
                    continue

        session_offset += len(session_line_ids)
        if session_offset >= len(line_ids):
            break

    if seeded_profiles is None:
        raise RuntimeError("No voice profiles were seeded during capture")

    captured_lines = [captured_by_line_id[line_id] for line_id in line_ids if line_id in captured_by_line_id]
    if not captured_lines:
        raise RuntimeError("Capture did not produce any editor audio clips")

    failed_line_ids = [line_id for line_id in line_ids if line_id not in captured_by_line_id]
    if failed_line_ids:
        preview = ", ".join(str(line_id) for line_id in failed_line_ids[:20])
        suffix = "" if len(failed_line_ids) <= 20 else f", ... (+{len(failed_line_ids) - 20} more)"
        print(
            f"[capture] Completed with partial coverage: {len(captured_lines)}/{len(line_ids)} lines captured. "
            f"Failed line ids: {preview}{suffix}"
        )

    qwen_fixture = _build_qwen_fixture(
        lines=captured_lines,
        voice_profiles=seeded_profiles,
        qwen_fixture_path=qwen_fixture_output,
        source_book=os.path.relpath(source_book_abs, REPO_ROOT),
        tts_local_backend=resolved_backend,
    )

    qwen_fixture_abs = os.path.abspath(qwen_fixture_output)
    manifest_abs = os.path.abspath(manifest_output)
    os.makedirs(os.path.dirname(qwen_fixture_abs), exist_ok=True)
    os.makedirs(os.path.dirname(manifest_abs), exist_ok=True)

    with open(qwen_fixture_abs, "w", encoding="utf-8") as handle:
        json.dump(qwen_fixture, handle, ensure_ascii=False, indent=2)

    manifest_payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_book": os.path.relpath(source_book_abs, REPO_ROOT),
        "script_seed_fixture": os.path.relpath(script_seed_abs, REPO_ROOT),
        "voice_profile_manifest": os.path.relpath(voice_manifest_abs, REPO_ROOT),
        "tts_local_backend": resolved_backend,
        "line_count": len(captured_lines),
        "speaker_count": len({_normalize_name(line.speaker) for line in captured_lines}),
        "speakers": sorted({line.speaker for line in captured_lines}, key=lambda value: value.casefold()),
        "failed_line_count": len(failed_line_ids),
        "failed_line_ids": failed_line_ids,
        "failed_lines": [
            failed_line_errors[line_id]
            for line_id in failed_line_ids
            if line_id in failed_line_errors
        ],
        "session_size": session_size,
        "lines": [
            {
                "uid": line.uid,
                "line_id": line.line_id,
                "speaker": line.speaker,
                "text": line.text,
                "instruct": line.instruct,
                "transformed_text": line.transformed_text,
                "source_audio_path": line.source_audio_path,
                "fixture_audio_path": os.path.relpath(line.fixture_audio_path, REPO_ROOT),
                "sample_rate": line.sample_rate,
                "frames": line.frames,
                "duration_seconds": round(line.duration_seconds, 6),
                "audio_size": line.audio_size,
                "audio_sha256": line.audio_sha256,
            }
            for line in captured_lines
        ],
        "qwen_fixture_path": os.path.relpath(qwen_fixture_abs, REPO_ROOT),
        "asset_dir": os.path.relpath(asset_dir_abs, REPO_ROOT),
    }

    with open(manifest_abs, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)

    _update_worklog(
        line_count=len(captured_lines),
        speaker_count=len({_normalize_name(line.speaker) for line in captured_lines}),
        backend=resolved_backend,
        failed_count=len(failed_line_ids),
        qwen_fixture_path=qwen_fixture_abs,
        manifest_path=manifest_abs,
        asset_dir=asset_dir_abs,
    )

    return {
        "status": "ok",
        "line_count": len(captured_lines),
        "failed_line_count": len(failed_line_ids),
        "failed_line_ids": failed_line_ids,
        "speaker_count": len({_normalize_name(line.speaker) for line in captured_lines}),
        "tts_local_backend": resolved_backend,
        "qwen_fixture_path": qwen_fixture_abs,
        "asset_dir": asset_dir_abs,
        "manifest_path": manifest_abs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture editor-phase audio fixtures from real local Qwen generation")
    parser.add_argument("--source-book", default=BOOK_DEFAULT)
    parser.add_argument("--script-seed-fixture", default=SCRIPT_SEED_FIXTURE_DEFAULT)
    parser.add_argument("--voice-profile-manifest", default=VOICE_PROFILE_MANIFEST_DEFAULT)
    parser.add_argument("--qwen-output", default=QWEN_FIXTURE_DEFAULT)
    parser.add_argument("--asset-dir", default=ASSET_DIR_DEFAULT)
    parser.add_argument("--manifest-output", default=MANIFEST_DEFAULT)
    parser.add_argument("--tts-local-backend", default="auto", choices=["auto", "qwen", "mlx"])
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = capture_editor_audio_fixtures(
        source_book_path=args.source_book,
        script_seed_fixture=args.script_seed_fixture,
        voice_profile_manifest=args.voice_profile_manifest,
        qwen_fixture_output=args.qwen_output,
        asset_output_dir=args.asset_dir,
        manifest_output=args.manifest_output,
        tts_local_backend=args.tts_local_backend,
        keep_temp=bool(args.keep_temp),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
