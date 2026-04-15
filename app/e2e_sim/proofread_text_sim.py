"""Fixture-backed transcript simulator for proofread/ASR flows in E2E tests."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from .fixture_queue import env_flag, load_fixture_payload


class ProofreadTextSimProvider:
    """Lookup-based transcript fixture provider for proofreading flows."""

    ENV_FIXTURE_PATH = "THREADSPEAK_E2E_PROOFREAD_FIXTURE"
    ENV_REPORT_PATH = "THREADSPEAK_E2E_PROOFREAD_REPORT_PATH"
    ENV_TRACE_PATH = "THREADSPEAK_E2E_PROOFREAD_TRACE_PATH"
    ENV_FALLBACK_MODE = "THREADSPEAK_E2E_PROOFREAD_FALLBACK"

    def __init__(self, fixture_path: str):
        payload = load_fixture_payload(fixture_path)
        self._fixture_path = os.path.abspath(fixture_path)
        self._report_path = str(os.getenv(self.ENV_REPORT_PATH) or payload.get("report_path") or "").strip()
        self._trace_path = str(os.getenv(self.ENV_TRACE_PATH) or payload.get("trace_path") or "").strip()
        self._strict = bool(payload.get("strict", env_flag("THREADSPEAK_E2E_SIM_STRICT", default=True)))
        self._fallback_mode = str(
            os.getenv(self.ENV_FALLBACK_MODE)
            or payload.get("fallback_mode")
            or "chunk_text"
        ).strip().lower()
        if self._fallback_mode not in {"chunk_text", "fail"}:
            raise ValueError(
                f"Unsupported proofread fallback mode '{self._fallback_mode}'. "
                "Expected 'chunk_text' or 'fail'."
            )

        raw_entries = payload.get("entries") or []
        if not isinstance(raw_entries, list):
            raise ValueError("Proofread transcript fixture must contain an 'entries' list")

        self._entries_by_audio: Dict[str, Dict[str, Any]] = {}
        for entry in raw_entries:
            item = dict(entry or {})
            audio_path = self._normalize_audio_path(item.get("audio_path"))
            if not audio_path:
                raise ValueError("Each proofread fixture entry must include a non-empty audio_path")
            transcript_text = str(item.get("transcript_text") or "").strip()
            if not transcript_text:
                raise ValueError(
                    f"Fixture entry for audio_path '{audio_path}' must include non-empty transcript_text"
                )
            if audio_path in self._entries_by_audio:
                raise ValueError(f"Duplicate proofread fixture entry for audio_path '{audio_path}'")
            self._entries_by_audio[audio_path] = {
                "audio_path": audio_path,
                "transcript_text": transcript_text,
                "uid": str(item.get("uid") or "").strip(),
                "speaker": str(item.get("speaker") or "").strip(),
                "line_id": item.get("line_id"),
            }

        self._lock = threading.Lock()
        self._trace_lock = threading.Lock()
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._fallback_count = 0

        self._write_report("initialized")
        self._trace(
            "initialized",
            {
                "fixture_path": self._fixture_path,
                "entry_count": len(self._entries_by_audio),
                "strict": self._strict,
                "fallback_mode": self._fallback_mode,
            },
        )

    @classmethod
    def from_env(cls) -> Optional["ProofreadTextSimProvider"]:
        fixture_path = str(os.getenv(cls.ENV_FIXTURE_PATH) or "").strip()
        if not fixture_path:
            return None
        return cls(fixture_path)

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def fallback_mode(self) -> str:
        return self._fallback_mode

    @staticmethod
    def _normalize_audio_path(value: Any) -> str:
        normalized = str(value or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _trace(self, event: str, data: Dict[str, Any]) -> None:
        path = self._trace_path
        if not path:
            return
        payload = {
            "ts": time.time(),
            "event": str(event or "").strip(),
            "fixture_path": self._fixture_path,
            "strict": self._strict,
            "fallback_mode": self._fallback_mode,
            "lookup_count": self._lookup_count,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "fallback_count": self._fallback_count,
        }
        payload.update(dict(data or {}))
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._trace_lock:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    def _write_report(self, state: str, *, last_request: str = "", last_outcome: str = "") -> None:
        path = self._report_path
        if not path:
            return
        with self._lock:
            payload = {
                "state": state,
                "fixture_path": self._fixture_path,
                "entry_count": len(self._entries_by_audio),
                "strict": self._strict,
                "fallback_mode": self._fallback_mode,
                "lookup_count": self._lookup_count,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "fallback_count": self._fallback_count,
                "last_request": last_request,
                "last_outcome": last_outcome,
            }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def resolve_transcript(self, audio_path: str, fallback_text: Optional[str] = None) -> Optional[str]:
        normalized_path = self._normalize_audio_path(audio_path)
        fallback_value = str(fallback_text or "").strip()

        with self._lock:
            self._lookup_count += 1
            entry = self._entries_by_audio.get(normalized_path)
            if entry is not None:
                self._hit_count += 1
                transcript = str(entry.get("transcript_text") or "").strip()
                outcome = "hit"
            else:
                self._miss_count += 1
                transcript = None
                outcome = "miss"

            if transcript is None and fallback_value and self._fallback_mode == "chunk_text":
                self._fallback_count += 1
                transcript = fallback_value
                outcome = "fallback_chunk_text"

            strict_failure = (
                transcript is None
                and (self._strict or self._fallback_mode == "fail")
            )

        self._trace(
            "resolve_transcript",
            {
                "audio_path": normalized_path,
                "outcome": outcome,
                "strict_failure": strict_failure,
                "has_fallback_text": bool(fallback_value),
            },
        )
        self._write_report(
            "resolved",
            last_request=normalized_path,
            last_outcome=outcome,
        )

        if strict_failure:
            raise AssertionError(
                "No proofread transcript fixture entry for "
                f"'{normalized_path}'. "
                "Either add the fixture transcript or supply fallback-mapped chunk text."
            )
        return transcript
