"""Fixture-driven LM Studio compatible simulation server for E2E tests."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from .fixture_queue import ScriptedInteractionQueue, env_flag, load_fixture_payload


class _LMStudioRequestHandler(BaseHTTPRequestHandler):
    server_version = "LMStudioSim/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 - stdlib handler method name
        self._handle_request("GET")

    def do_POST(self):  # noqa: N802 - stdlib handler method name
        self._handle_request("POST")

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _handle_request(self, method: str) -> None:
        server: "LMStudioSimServer" = self.server.sim_server  # type: ignore[attr-defined]
        path = urlparse(self.path).path

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""

        payload = None
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = {"raw_body": body.decode("utf-8", errors="replace")}

        try:
            status, headers, response_payload, stream_events = server.dispatch(method, path, payload)
        except Exception as exc:
            message = {"error": str(exc)}
            encoded = json.dumps(message, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            return

        if stream_events is not None:
            self.send_response(status)
            self.send_header("Content-Type", headers.get("Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in stream_events:
                chunk = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        encoded = json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()


class LMStudioSimServer(AbstractContextManager):
    """Local HTTP server that replays fixture-defined LM Studio interactions."""

    def __init__(self, fixture_path: str, trace_path: str | None = None, trace_label: str | None = None):
        payload = load_fixture_payload(fixture_path)
        routes = payload.get("routes") or {}
        if not isinstance(routes, dict):
            raise ValueError("LM Studio fixture must include a 'routes' object")

        self._fixture_path = str(fixture_path)
        self._trace_path = str(trace_path or payload.get("trace_path") or "").strip()
        self._trace_label = str(
            trace_label
            or payload.get("trace_label")
            or payload.get("label")
            or self._fixture_path
        ).strip()
        self._trace_lock = threading.Lock()
        strict = bool(payload.get("strict", env_flag("THREADSPEAK_E2E_SIM_STRICT", default=True)))
        self._queue = ScriptedInteractionQueue(routes=routes, strict=strict)
        self._host = str(payload.get("host") or "127.0.0.1")
        self._port = int(payload.get("port") or 0)

        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url = ""

    def _trace(self, event: str, data: Dict[str, Any]) -> None:
        path = self._trace_path
        if not path:
            return
        payload = {
            "ts": time.time(),
            "event": str(event or "").strip(),
            "label": self._trace_label,
            "strict": self._queue.strict,
            "pending": self._queue.pending_counts(),
        }
        payload.update(dict(data or {}))
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._trace_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    @staticmethod
    def _find_free_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    def start(self) -> "LMStudioSimServer":
        if self._server is not None:
            return self

        port = self._port or self._find_free_port(self._host)
        server = ThreadingHTTPServer((self._host, port), _LMStudioRequestHandler)
        server.sim_server = self  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        self._server = server
        self._thread = thread
        self.base_url = f"http://{self._host}:{port}"
        self._trace(
            "server_started",
            {
                "fixture_path": self._fixture_path,
                "base_url": self.base_url,
            },
        )
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
        if server is not None:
            self._trace("server_stopped", {"base_url": self.base_url})

    def __enter__(self) -> "LMStudioSimServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def assert_all_consumed(self) -> None:
        self._queue.assert_all_consumed(context="LM Studio simulator")

    def dispatch(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, str], Dict[str, Any], Optional[list]]:
        key = f"{method} {path}"
        self._trace("request_received", {"key": key, "request": payload})

        if method == "GET" and path == "/":
            self._trace("request_default", {"key": key, "status": 200})
            return 200, {}, {"status": "ok", "name": "lmstudio-sim"}, None

        try:
            entry = self._queue.consume(key, payload)
        except Exception as exc:
            self._trace(
                "request_consume_error",
                {
                    "key": key,
                    "request": payload,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise

        if entry is None:
            if method == "GET" and path == "/v1/models":
                self._trace("request_default", {"key": key, "status": 200, "fallback": "/v1/models"})
                return 200, {}, {"object": "list", "data": []}, None
            self._trace("request_unexpected", {"key": key, "request": payload})
            raise AssertionError(f"No scripted LM Studio response for {key}")

        status = int(entry.get("status") or 200)
        headers = dict(entry.get("headers") or {})
        self._trace(
            "request_consumed",
            {
                "key": key,
                "status": status,
                "has_stream_events": "stream_events" in entry,
                "metadata": dict(entry.get("metadata") or {}),
            },
        )

        if "stream_events" in entry:
            stream_events = entry.get("stream_events") or []
            if not isinstance(stream_events, list):
                raise AssertionError(f"stream_events for {key} must be a list")
            return status, headers, {}, stream_events

        response_payload = entry.get("response")
        if response_payload is None:
            response_payload = {}
        if not isinstance(response_payload, dict):
            raise AssertionError(f"response for {key} must be a JSON object")
        return status, headers, response_payload, None
