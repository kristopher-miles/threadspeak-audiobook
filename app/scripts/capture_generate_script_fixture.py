#!/usr/bin/env python3
"""Capture non-legacy Generate Script LM Studio responses and build replay fixture."""

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
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from llm import LMStudioModelLoadService  # noqa: E402
from runtime_layout import RuntimeLayout  # noqa: E402

LMSTUDIO_DEFAULT_BASE = "http://127.0.0.1:1234"
MODEL_DEFAULT = "qwen/qwen3.5-9b"
FIXTURE_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_generate_script_test_book.json")
BOOK_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "files", "test_book.epub")
WORKLOG_PATH = os.path.join(REPO_ROOT, "wiki", "Generate-Script-Fixture-Worklog.md")


@dataclass
class CaptureResult:
    run_id: str
    chunk_calls: int
    fixture_path: str


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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        self._temp_root = ""
        self._proc: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.app_dir = ""
        self.layout: RuntimeLayout | None = None
        self.keep_temp = bool(keep_temp)

    def __enter__(self):
        self._temp_root = tempfile.mkdtemp(prefix="threadspeak_capture_script_fixture_")
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


def _lmstudio_models_payload(base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{base_url}/api/v1/models", timeout=15)
    _assert_status(response, 200, "LM Studio list models")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("LM Studio /api/v1/models did not return an object")
    return payload


def _find_model(payload: Dict[str, Any], model_name: str) -> Dict[str, Any] | None:
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    wanted = str(model_name or "").strip()
    for item in models:
        if not isinstance(item, dict):
            continue
        if wanted in {
            str(item.get("key") or "").strip(),
            str(item.get("display_name") or "").strip(),
        }:
            return item
    return None


def _wait_for_model_loaded(base_url: str, model_name: str, timeout_seconds: int = 300) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = _lmstudio_models_payload(base_url)
        model = _find_model(payload, model_name)
        if model is not None:
            loaded_instances = model.get("loaded_instances")
            if loaded_instances is None or bool(loaded_instances):
                return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for LM Studio model '{model_name}' to load")


def _ensure_target_model_loaded(base_url: str, model_name: str, api_key: str) -> Dict[str, Any]:
    payload = _lmstudio_models_payload(base_url)
    model = _find_model(payload, model_name)
    if model is None:
        available = []
        for item in payload.get("models") or []:
            if isinstance(item, dict):
                available.append(str(item.get("key") or item.get("display_name") or ""))
        raise RuntimeError(
            f"Target model '{model_name}' not found in LM Studio. Available models: {available}"
        )

    if model.get("loaded_instances"):
        return model

    service = LMStudioModelLoadService(timeout_seconds=240)
    load_payload = service.load_model(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        model_name=model_name,
        context_length=16384,
        echo_load_config=True,
    )
    status = str(load_payload.get("status") or "")
    if status != "loaded":
        raise RuntimeError(f"LM Studio model load returned unexpected status: {load_payload}")

    _wait_for_model_loaded(base_url, model_name)
    payload = _lmstudio_models_payload(base_url)
    model = _find_model(payload, model_name)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' disappeared after load")
    return model


def _safe_json_object(body: bytes) -> Dict[str, Any]:
    if not body:
        return {}
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except Exception:
        return {"raw_body": text}
    if isinstance(parsed, dict):
        return parsed
    return {"raw_body": text}


def _safe_json_event(raw_json: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except Exception:
        return {"raw": raw_json}
    if isinstance(parsed, dict):
        return parsed
    return {"raw": raw_json}


def _extract_tool_name(payload: Dict[str, Any]) -> str:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return ""
    first = tools[0]
    if not isinstance(first, dict):
        return ""
    function = first.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _build_expect_subset(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    subset: Dict[str, Any] = {}
    for key in ("model", "stream", "tool_choice", "parallel_tool_calls"):
        if key in request_payload:
            subset[key] = request_payload.get(key)
    tool_name = _extract_tool_name(request_payload)
    if tool_name:
        subset["tools"] = [{"function": {"name": tool_name}}]
    return subset


class _RecordingProxyHandler(BaseHTTPRequestHandler):
    server_version = "LMStudioCaptureProxy/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self):  # noqa: N802 - stdlib method name
        self._handle("GET")

    def do_POST(self):  # noqa: N802 - stdlib method name
        self._handle("POST")

    def _handle(self, method: str) -> None:
        proxy: "_LMStudioRecordingProxy" = self.server.proxy_server  # type: ignore[attr-defined]
        path = urlparse(self.path).path

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        request_payload = _safe_json_object(body)
        request_headers = {k: v for (k, v) in self.headers.items()}

        try:
            if method == "POST" and path == "/v1/chat/completions" and bool(request_payload.get("stream")):
                proxy.forward_stream_response(
                    request_payload=request_payload,
                    request_headers=request_headers,
                    handler=self,
                )
                return

            status, headers, payload = proxy.forward_json_response(
                method=method,
                path=path,
                request_payload=request_payload,
                request_headers=request_headers,
                raw_body=body,
            )
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Stream consumers intentionally close early once required tool args are parsed.
            return
        except Exception as exc:
            message = {"error": str(exc)}
            encoded = json.dumps(message, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                self.wfile.flush()
            except Exception:
                return


class _LMStudioRecordingProxy:
    def __init__(self, *, upstream_base_url: str):
        self._upstream_base_url = _normalize_base_url(upstream_base_url)
        self._routes: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url = ""

    @staticmethod
    def _forward_headers(request_headers: Dict[str, str]) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key in ("Authorization", "Content-Type", "Accept"):
            value = str(request_headers.get(key) or "").strip()
            if value:
                headers[key] = value
        return headers

    def _append_route(self, key: str, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._routes.setdefault(key, []).append(dict(entry))

    def _record_chat_json(
        self,
        *,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        status: int,
        content_type: str,
    ) -> None:
        entry: Dict[str, Any] = {
            "expect": _build_expect_subset(request_payload),
            "response": response_payload,
        }
        if int(status) != 200:
            entry["status"] = int(status)
        if content_type:
            entry["headers"] = {"Content-Type": content_type}
        self._append_route("POST /v1/chat/completions", entry)

    def _record_chat_stream(
        self,
        *,
        request_payload: Dict[str, Any],
        stream_events: List[Dict[str, Any]],
        status: int,
        content_type: str,
    ) -> None:
        entry: Dict[str, Any] = {
            "expect": _build_expect_subset(request_payload),
            "stream_events": list(stream_events),
        }
        if int(status) != 200:
            entry["status"] = int(status)
        if content_type:
            entry["headers"] = {"Content-Type": content_type}
        self._append_route("POST /v1/chat/completions", entry)

    def _record_models(self, *, payload: Dict[str, Any], status: int, content_type: str, path: str) -> None:
        key = f"GET {path}"
        entry: Dict[str, Any] = {"response": payload}
        if int(status) != 200:
            entry["status"] = int(status)
        if content_type:
            entry["headers"] = {"Content-Type": content_type}
        self._append_route(key, entry)

    def _upstream_url(self, path: str) -> str:
        return f"{self._upstream_base_url}{path}"

    def start(self) -> "_LMStudioRecordingProxy":
        if self._server is not None:
            return self
        host = "127.0.0.1"
        port = _find_free_port()
        server = ThreadingHTTPServer((host, port), _RecordingProxyHandler)
        server.proxy_server = self  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        self.base_url = f"http://{host}:{port}"
        return self

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "_LMStudioRecordingProxy":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def forward_json_response(
        self,
        *,
        method: str,
        path: str,
        request_payload: Dict[str, Any],
        request_headers: Dict[str, str],
        raw_body: bytes,
    ) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
        headers = self._forward_headers(request_headers)
        url = self._upstream_url(path)

        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            if raw_body and not request_payload:
                response = requests.post(url, data=raw_body, headers=headers, timeout=600)
            else:
                response = requests.post(url, json=request_payload, headers=headers, timeout=600)
        else:
            raise RuntimeError(f"Unsupported method for capture proxy: {method}")

        content_type = str(response.headers.get("Content-Type") or "").strip() or "application/json"
        payload = _safe_json_object(response.content)

        if method == "GET" and path in {"/api/v1/models", "/v1/models"}:
            self._record_models(payload=payload, status=response.status_code, content_type=content_type, path=path)
        elif method == "POST" and path == "/v1/chat/completions":
            self._record_chat_json(
                request_payload=request_payload,
                response_payload=payload,
                status=response.status_code,
                content_type=content_type,
            )

        return int(response.status_code), {"Content-Type": content_type}, payload

    def forward_stream_response(
        self,
        *,
        request_payload: Dict[str, Any],
        request_headers: Dict[str, str],
        handler: _RecordingProxyHandler,
    ) -> None:
        headers = self._forward_headers(request_headers)
        url = self._upstream_url("/v1/chat/completions")

        with requests.post(url, json=request_payload, headers=headers, stream=True, timeout=600) as response:
            status = int(response.status_code)
            content_type = str(response.headers.get("Content-Type") or "text/event-stream").strip() or "text/event-stream"

            if status >= 400:
                payload = _safe_json_object(response.content)
                self._record_chat_json(
                    request_payload=request_payload,
                    response_payload=payload,
                    status=status,
                    content_type=content_type,
                )
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                handler.send_response(status)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(encoded)))
                handler.end_headers()
                handler.wfile.write(encoded)
                handler.wfile.flush()
                return

            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Connection", "close")
            handler.end_headers()

            stream_events: List[Dict[str, Any]] = []
            seen_done = False
            client_open = True

            def _write_sse(data_value: str) -> bool:
                try:
                    handler.wfile.write(f"data: {data_value}\n\n".encode("utf-8"))
                    handler.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False

            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue

                if data == "[DONE]":
                    seen_done = True
                    if client_open:
                        client_open = _write_sse("[DONE]")
                    break

                event = _safe_json_event(data)
                stream_events.append(event)
                if client_open:
                    client_open = _write_sse(json.dumps(event, ensure_ascii=False))
                if not client_open:
                    break

            if not seen_done and client_open:
                _write_sse("[DONE]")

            self._record_chat_stream(
                request_payload=request_payload,
                stream_events=stream_events,
                status=status,
                content_type=content_type,
            )

    def snapshot_routes(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {
                key: [dict(item) for item in items]
                for key, items in self._routes.items()
            }


def _wait_for_new_mode_script_ready(
    base_url: str,
    *,
    timeout_seconds: int = 1800,
    inactivity_timeout_seconds: int = 10,
    poll_seconds: float = 0.8,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: Dict[str, Any] = {}
    last_activity_at = time.time()
    last_token: Optional[Tuple[Any, ...]] = None

    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/status/new_mode_workflow", timeout=30)
        _assert_status(response, 200, "poll new_mode_workflow")
        payload = dict(response.json() or {})
        last_payload = payload

        logs = [str(item) for item in (payload.get("logs") or [])]
        completed = tuple(sorted(str(item) for item in (payload.get("completed_stages") or [])))
        token = (
            len(logs),
            completed,
            str(payload.get("current_stage") or ""),
            bool(payload.get("running")),
            bool(payload.get("paused")),
            str(payload.get("last_error") or ""),
        )
        if token != last_token:
            last_activity_at = time.time()
            last_token = token

        for line in logs:
            lower = line.lower()
            if line.startswith("ERROR:") or "failed with return code" in lower or "failed to call llm provider" in lower:
                raise RuntimeError(f"Non-legacy workflow failed: {line}")

        if str(payload.get("last_error") or "").strip():
            raise RuntimeError(f"Non-legacy workflow failed: {payload.get('last_error')}")

        if (
            not bool(payload.get("running"))
            and not bool(payload.get("paused"))
            and "create_script" in completed
        ):
            return payload

        if (time.time() - last_activity_at) > inactivity_timeout_seconds:
            raise RuntimeError(
                "Non-legacy script workflow stalled (no state/log activity for "
                f"{inactivity_timeout_seconds}s). "
                f"Last payload: {json.dumps(last_payload, ensure_ascii=False)}"
            )

        time.sleep(poll_seconds)

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


def _ensure_models_route_present(
    routes: Dict[str, List[Dict[str, Any]]],
    *,
    lmstudio_base_url: str,
) -> None:
    if routes.get("GET /api/v1/models"):
        return
    models_payload = _lmstudio_models_payload(lmstudio_base_url)
    routes["GET /api/v1/models"] = [{"response": models_payload}]


def _build_fixture_payload(
    *,
    routes: Dict[str, List[Dict[str, Any]]],
    model_name: str,
    source_file: str,
    capture_id: str,
) -> Dict[str, Any]:
    chat_calls = len(routes.get("POST /v1/chat/completions") or [])
    return {
        "strict": True,
        "metadata": {
            "purpose": "Non-legacy Generate Script fixture captured from live LM Studio",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "run_id": capture_id,
            "model_name": model_name,
            "chat_completion_call_count": chat_calls,
            "workflow": "new_mode_workflow",
            "stages": ["process_paragraphs", "assign_dialogue", "extract_temperament", "create_script"],
            "llm_mode": "tool_streaming",
        },
        "routes": routes,
    }


def _update_worklog(*, run_id: str, chat_calls: int, fixture_path: str, model_name: str) -> None:
    if not os.path.exists(WORKLOG_PATH):
        return

    with open(WORKLOG_PATH, "r", encoding="utf-8") as handle:
        content = handle.read()

    now = datetime.now(timezone.utc).isoformat()
    rel_fixture = os.path.relpath(fixture_path, REPO_ROOT)

    content = re.sub(r"- Status: .*", "- Status: complete", content)
    content = re.sub(r"- Last run timestamp: .*", f"- Last run timestamp: {now}", content)
    content = re.sub(r"- Captured run_id: .*", f"- Captured run_id: {run_id}", content)
    content = re.sub(r"- Captured model: .*", f"- Captured model: `{model_name}`", content)
    content = re.sub(r"- Captured (?:chunk|chat) calls: .*", f"- Captured chat calls: {chat_calls}", content)
    content = re.sub(r"- Fixture path: .*", f"- Fixture path: `{rel_fixture}`", content)
    content = re.sub(
        r"- Notes: .*",
        "- Notes: captured from isolated non-legacy workflow (`/api/new_mode_workflow/start`) with llm_workers=1",
        content,
    )

    with open(WORKLOG_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def capture_generate_script_fixture(
    *,
    lmstudio_base_url: str,
    model_name: str,
    api_key: str,
    source_book_path: str,
    output_path: str,
    keep_temp: bool = False,
) -> CaptureResult:
    base = _normalize_base_url(lmstudio_base_url)
    source_book_abs = os.path.abspath(source_book_path)
    if not os.path.exists(source_book_abs):
        raise FileNotFoundError(f"Source book not found: {source_book_abs}")

    print(f"[capture] Ensuring LM Studio model is loaded: {model_name}")
    _ensure_target_model_loaded(base, model_name, api_key)

    capture_id = f"new-mode-{uuid.uuid4()}"
    with _LMStudioRecordingProxy(upstream_base_url=base) as proxy:
        print(f"[capture] Recording proxy started: {proxy.base_url}")
        with _IsolatedServer(keep_temp=keep_temp) as server:
            print(f"[capture] Isolated app started: {server.base_url}")
            setup_payload = {
                "llm": {
                    "base_url": f"{proxy.base_url}/v1",
                    "api_key": api_key,
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
            setup_response = requests.post(f"{server.base_url}/api/config/setup", json=setup_payload, timeout=60)
            _assert_status(setup_response, 200, "configure isolated setup")

            config_response = requests.get(f"{server.base_url}/api/config", timeout=30)
            _assert_status(config_response, 200, "get isolated config")
            config_payload = config_response.json()
            _sync_scripts_config(server.app_dir, setup_payload, config_payload)

            _upload_book(server.base_url, source_book_abs)
            print("[capture] Uploaded source EPUB")

            start_response = requests.post(
                f"{server.base_url}/api/new_mode_workflow/start",
                json={"process_voices": False, "generate_audio": False},
                timeout=60,
            )
            _assert_status(start_response, 200, "start non-legacy script workflow")
            print("[capture] Started non-legacy script workflow")

            status_payload = _wait_for_new_mode_script_ready(
                server.base_url,
                timeout_seconds=1800,
                inactivity_timeout_seconds=10,
                poll_seconds=0.8,
            )
            logs = [str(item) for item in (status_payload.get("logs") or [])]
            if logs:
                print("[capture] Workflow log tail:")
                for line in logs[-30:]:
                    print(line)

            routes = proxy.snapshot_routes()
            _ensure_models_route_present(routes, lmstudio_base_url=base)
            chat_calls = len(routes.get("POST /v1/chat/completions") or [])
            if chat_calls <= 0:
                raise RuntimeError("No LM Studio chat completions were captured from non-legacy workflow")

            fixture = _build_fixture_payload(
                routes=routes,
                model_name=model_name,
                source_file=os.path.relpath(source_book_abs, REPO_ROOT),
                capture_id=capture_id,
            )

            target_output = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(target_output), exist_ok=True)
            with open(target_output, "w", encoding="utf-8") as handle:
                json.dump(fixture, handle, ensure_ascii=False, indent=2)

            _update_worklog(
                run_id=capture_id,
                chat_calls=chat_calls,
                fixture_path=target_output,
                model_name=model_name,
            )

            print(f"[capture] Captured {chat_calls} LM Studio completions")
            return CaptureResult(run_id=capture_id, chunk_calls=chat_calls, fixture_path=target_output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture non-legacy Generate Script LM Studio fixture")
    parser.add_argument("--lmstudio-base-url", default=LMSTUDIO_DEFAULT_BASE)
    parser.add_argument("--model-name", default=MODEL_DEFAULT)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--source-book", default=BOOK_DEFAULT)
    parser.add_argument("--output", default=FIXTURE_DEFAULT)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = capture_generate_script_fixture(
        lmstudio_base_url=args.lmstudio_base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        source_book_path=args.source_book,
        output_path=args.output,
        keep_temp=bool(args.keep_temp),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": result.run_id,
                "chunk_calls": result.chunk_calls,
                "fixture_path": result.fixture_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
