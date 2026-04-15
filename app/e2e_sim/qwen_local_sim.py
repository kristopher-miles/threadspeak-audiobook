"""Fixture-driven local Qwen provider simulator used in E2E tests."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from audio_validation import (
    DEFAULT_DURATION_TOLERANCE_FACTOR,
    count_words,
    estimate_expected_duration_seconds,
)

from .fixture_queue import ScriptedInteractionQueue, env_flag, load_fixture_payload


TRUE_VALUES = {"1", "true", "yes", "on"}

_TEXT_EQUIVALENCE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def _normalize_match_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.translate(_TEXT_EQUIVALENCE_MAP)
    return " ".join(normalized.split())


def _minimum_duration_for_text(text: Any) -> float:
    normalized = str(text or "").strip()
    words = count_words(normalized)
    if words <= 0:
        return 0.0
    expected = estimate_expected_duration_seconds(word_count=words)
    return float(expected) / float(DEFAULT_DURATION_TOLERANCE_FACTOR)


def _ensure_min_duration(audio: np.ndarray, sample_rate: int, text: Any) -> np.ndarray:
    minimum_seconds = _minimum_duration_for_text(text)
    if minimum_seconds <= 0.0:
        return audio
    # Audio validation compares floating values before UI rounding. Add a
    # small headroom to avoid borderline "< min duration" failures after
    # encode/decode and duration precision drift.
    target_seconds = max(minimum_seconds + 0.2, minimum_seconds * 1.05)
    minimum_samples = int(max(1, np.ceil(target_seconds * max(1, int(sample_rate)))))
    if int(audio.shape[0]) >= minimum_samples:
        return audio
    pad_samples = max(1, minimum_samples - int(audio.shape[0]))
    return np.concatenate([audio, np.zeros((pad_samples,), dtype=np.float32)])


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in TRUE_VALUES


def _decode_wav_base64(payload: str) -> Tuple[np.ndarray, int]:
    import base64

    raw = base64.b64decode(payload)
    with wave.open(BytesIO(raw), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Only PCM16 wav fixtures are supported")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, int(sample_rate)


def _decode_wav_file(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("Only PCM16 wav fixtures are supported")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, int(sample_rate)


def _synthesize_audio(duration_ms: int, sample_rate: int, kind: str = "tone") -> np.ndarray:
    duration_ms = max(20, int(duration_ms))
    sample_count = max(1, int(sample_rate * (duration_ms / 1000.0)))
    if kind == "silence":
        return np.zeros(sample_count, dtype=np.float32)

    time_axis = np.linspace(0, duration_ms / 1000.0, sample_count, endpoint=False)
    waveform = 0.08 * np.sin(2.0 * np.pi * 220.0 * time_axis)
    return waveform.astype(np.float32)


@dataclass
class _SimPromptItem:
    ref_code: np.ndarray
    ref_text: str


class _QwenLocalSimModel:
    def __init__(self, provider: "QwenLocalSimProvider", model_kind: str):
        self._provider = provider
        self._model_kind = model_kind

    def generate_custom_voice(self, **kwargs):
        return self._provider.generate_custom_voice(self._model_kind, kwargs)

    def create_voice_clone_prompt(self, **kwargs):
        return self._provider.create_voice_clone_prompt(kwargs)

    def generate_voice_clone(self, **kwargs):
        return self._provider.generate_voice_clone(self._model_kind, kwargs)

    def generate_voice_design(self, **kwargs):
        return self._provider.generate_voice_design(kwargs)

    @staticmethod
    def _tokenize_texts(texts: List[str]) -> List[List[int]]:
        return [[ord(char) % 127 for char in str(text)] for text in (texts or [])]


class QwenLocalSimProvider:
    """Fixture-backed simulator for local Qwen model API surface."""

    ENV_FIXTURE_PATH = "THREADSPEAK_E2E_QWEN_FIXTURE"
    ENV_REPORT_PATH = "THREADSPEAK_E2E_QWEN_REPORT_PATH"
    ENV_TRACE_PATH = "THREADSPEAK_E2E_QWEN_TRACE_PATH"

    def __init__(self, fixture_path: str):
        payload = load_fixture_payload(fixture_path)
        self._fixture_path = os.path.abspath(fixture_path)
        self._fixture_dir = os.path.dirname(self._fixture_path)
        methods = payload.get("methods") or {}
        if not isinstance(methods, dict):
            raise ValueError("Qwen simulator fixture must include a 'methods' object")
        self._has_scripted_custom_voice = bool(list(methods.get("generate_custom_voice") or []))
        self._has_scripted_clone_prompt = bool(list(methods.get("create_voice_clone_prompt") or []))
        unordered_methods = payload.get("unordered_methods")
        if not isinstance(unordered_methods, list):
            unordered_methods = ["generate_voice_clone"]

        strict = bool(payload.get("strict", env_flag("THREADSPEAK_E2E_SIM_STRICT", default=True)))
        self._queue = ScriptedInteractionQueue(
            routes=methods,
            strict=strict,
            unordered_keys=[str(item) for item in unordered_methods if str(item).strip()],
        )
        self._default_sample_rate = int(payload.get("sample_rate") or 24000)
        self._default_duration_ms = int(payload.get("default_duration_ms") or 250)
        self._models: Dict[str, _QwenLocalSimModel] = {}
        self._report_path = str(os.getenv(self.ENV_REPORT_PATH) or "").strip()
        self._trace_path = str(os.getenv(self.ENV_TRACE_PATH) or payload.get("trace_path") or "").strip()
        self._report_lock = threading.Lock()
        self._trace_lock = threading.Lock()
        self._clone_replay_cache: Dict[str, Dict[str, Any]] = {}
        self._clone_audio_replay_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._clone_entry_by_text: Dict[str, Dict[str, Any]] = {}
        self._clone_entries_ordered: List[Dict[str, Any]] = [dict(item or {}) for item in list(methods.get("generate_voice_clone") or [])]
        self._clone_texts_ordered: List[str] = []
        for raw_entry in list(methods.get("generate_voice_clone") or []):
            entry = dict(raw_entry or {})
            expect = dict(entry.get("expect") or {})
            text_key = str(expect.get("text") or "").strip()
            if text_key and text_key not in self._clone_entry_by_text:
                self._clone_entry_by_text[text_key] = entry
            normalized_key = _normalize_match_text(text_key)
            if normalized_key and normalized_key not in self._clone_entry_by_text:
                self._clone_entry_by_text[normalized_key] = entry
        for entry in self._clone_entries_ordered:
            expect = dict(entry.get("expect") or {})
            self._clone_texts_ordered.append(_normalize_match_text(expect.get("text") or ""))
        self._write_report("initialized")
        self._trace("initialized", {"fixture_path": self._fixture_path})

    @classmethod
    def from_env(cls) -> Optional["QwenLocalSimProvider"]:
        fixture_path = str(os.getenv(cls.ENV_FIXTURE_PATH) or "").strip()
        if not fixture_path:
            return None
        return cls(fixture_path)

    def get_model(self, model_kind: str) -> _QwenLocalSimModel:
        key = str(model_kind or "").strip() or "custom_voice"
        model = self._models.get(key)
        if model is None:
            model = _QwenLocalSimModel(self, key)
            self._models[key] = model
        return model

    def assert_all_consumed(self) -> None:
        self._queue.assert_all_consumed(context="Qwen local simulator")

    def _consume(self, method_key: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._trace("request_received", {"method": method_key, "request": payload})
        try:
            entry = self._queue.consume(method_key, payload)
        except Exception as exc:
            self._trace(
                "request_consume_error",
                {
                    "method": method_key,
                    "request": payload,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise
        self._trace(
            "request_consumed",
            {
                "method": method_key,
                "metadata": dict((entry or {}).get("metadata") or {}),
            },
        )
        self._write_report(f"consumed:{method_key}")
        return entry

    def _trace(self, event: str, data: Dict[str, Any]) -> None:
        path = self._trace_path
        if not path:
            return
        payload = {
            "ts": time.time(),
            "event": str(event or "").strip(),
            "strict": self._queue.strict,
            "pending": self._queue.pending_counts(),
            "fixture_path": self._fixture_path,
        }
        payload.update(dict(data or {}))
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._trace_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    def _write_report(self, state: str) -> None:
        path = self._report_path
        if not path:
            return
        with self._report_lock:
            pending = self._queue.pending_counts()
            report = {
                "state": state,
                "pending": pending,
                "strict": self._queue.strict,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)

    def _resolve_audio_specs(
        self,
        entry: Optional[Dict[str, Any]],
        count: int,
    ) -> Tuple[List[np.ndarray], int]:
        entry = entry or {}
        sample_rate = int(entry.get("sample_rate") or self._default_sample_rate)

        if entry.get("audio_wav_base64"):
            audio, sr = _decode_wav_base64(str(entry["audio_wav_base64"]))
            self._trace(
                "audio_resolved",
                {"count": max(1, count), "sample_rate": int(sr), "source": "audio_wav_base64"},
            )
            return [audio for _ in range(max(1, count))], sr
        if entry.get("audio_wav_path"):
            audio, sr = _decode_wav_file(self._resolve_fixture_path(str(entry["audio_wav_path"])))
            self._trace(
                "audio_resolved",
                {"count": max(1, count), "sample_rate": int(sr), "source": "audio_wav_path"},
            )
            return [audio for _ in range(max(1, count))], sr

        specs: List[Dict[str, Any]] = []
        if isinstance(entry.get("audios"), list):
            specs = [dict(item or {}) for item in entry.get("audios")]
        elif isinstance(entry.get("audio"), dict):
            specs = [dict(entry["audio"]) for _ in range(max(1, count))]

        if not specs:
            specs = [{} for _ in range(max(1, count))]

        if len(specs) < count:
            specs.extend([copy.deepcopy(specs[-1]) for _ in range(count - len(specs))])

        outputs: List[np.ndarray] = []
        resolved_sources: List[str] = []
        for spec in specs[:count]:
            wav_path = str(spec.get("wav_path") or "").strip()
            if wav_path:
                audio, wav_sr = _decode_wav_file(self._resolve_fixture_path(wav_path))
                outputs.append(audio)
                sample_rate = int(wav_sr)
                resolved_sources.append(f"wav_path:{wav_path}")
                continue
            kind = str(spec.get("kind") or "tone").strip().lower()
            duration_ms = int(spec.get("duration_ms") or self._default_duration_ms)
            outputs.append(_synthesize_audio(duration_ms, sample_rate, kind=kind))
            resolved_sources.append(f"synth:{kind}:{duration_ms}ms")

        self._trace(
            "audio_resolved",
            {
                "count": len(outputs),
                "sample_rate": int(sample_rate),
                "sources": resolved_sources,
            },
        )

        return outputs, sample_rate

    def _apply_duration_floor(
        self,
        audios: List[np.ndarray],
        sample_rate: int,
        texts: List[Any],
        *,
        method: str,
    ) -> Tuple[List[np.ndarray], int]:
        if not audios:
            return audios, sample_rate
        if not texts:
            return audios, sample_rate
        padded = list(audios)
        adjusted = 0
        for idx, audio in enumerate(padded):
            text_idx = min(idx, len(texts) - 1)
            before_samples = int(audio.shape[0])
            after = _ensure_min_duration(audio, sample_rate, texts[text_idx])
            if int(after.shape[0]) > before_samples:
                adjusted += 1
                padded[idx] = after
        if adjusted > 0:
            self._trace(
                "audio_duration_floor_applied",
                {
                    "method": method,
                    "adjusted_clips": adjusted,
                    "sample_rate": int(sample_rate),
                },
            )
        return padded, sample_rate

    def _resolve_fixture_path(self, path_value: str) -> str:
        target = str(path_value or "").strip()
        if not target:
            raise ValueError("audio fixture path is required")
        if os.path.isabs(target):
            resolved = target
        else:
            resolved = os.path.abspath(os.path.join(self._fixture_dir, target))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Qwen simulator audio fixture not found: {resolved}")
        return resolved

    def generate_custom_voice(self, model_kind: str, kwargs: Dict[str, Any]):
        text = kwargs.get("text")
        texts = text if isinstance(text, list) else [text]
        payload = {
            "model_kind": model_kind,
            "text": text,
            "language": kwargs.get("language"),
            "speaker": kwargs.get("speaker"),
            "instruct": kwargs.get("instruct"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
        }
        if self._has_scripted_custom_voice:
            entry = self._consume("generate_custom_voice", payload)
        else:
            entry = {}
            self._trace(
                "request_dynamic",
                {
                    "method": "generate_custom_voice",
                    "request": payload,
                },
            )
        audios, sr = self._resolve_audio_specs(entry, len(texts))
        return self._apply_duration_floor(audios, sr, texts, method="generate_custom_voice")

    def create_voice_clone_prompt(self, kwargs: Dict[str, Any]):
        ref_text = str(kwargs.get("ref_text") or "")
        payload = {
            "ref_text": ref_text,
            "has_ref_audio": kwargs.get("ref_audio") is not None,
        }
        if self._has_scripted_clone_prompt:
            entry = self._consume("create_voice_clone_prompt", payload) or {}
        else:
            entry = {}
            self._trace(
                "request_dynamic",
                {
                    "method": "create_voice_clone_prompt",
                    "request": payload,
                },
            )
        token_count = int(entry.get("prompt_tokens") or max(1, len(ref_text) // 4))
        ref_code = np.zeros((token_count,), dtype=np.int32)
        prompt_ref_text = str(entry.get("prompt_ref_text") or ref_text)
        return [_SimPromptItem(ref_code=ref_code, ref_text=prompt_ref_text)]

    def generate_voice_clone(self, model_kind: str, kwargs: Dict[str, Any]):
        text = kwargs.get("text")
        texts = text if isinstance(text, list) else [text]
        match_text: Any
        if isinstance(text, list) and len(text) == 1:
            match_text = text[0]
        else:
            match_text = text
        payload: Dict[str, Any] = {
            "model_kind": model_kind,
            "text": match_text,
            "has_prompt": kwargs.get("voice_clone_prompt") is not None,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "has_instruct_ids": kwargs.get("instruct_ids") is not None,
        }
        replay_key = json.dumps(
            {
                "text": _normalize_match_text(payload.get("text")),
                "has_prompt": bool(payload.get("has_prompt")),
                "has_instruct_ids": bool(payload.get("has_instruct_ids")),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        try:
            entry = self._consume("generate_voice_clone", payload)
            self._clone_replay_cache[replay_key] = dict(entry or {})
        except AssertionError:
            replay_audio = self._clone_audio_replay_cache.get(replay_key)
            if replay_audio is not None:
                audio, sr = replay_audio
                self._trace(
                    "request_audio_replayed",
                    {
                        "method": "generate_voice_clone",
                        "request": payload,
                    },
                )
                replay_audios = [audio for _ in range(max(1, len(texts)))]
                return self._apply_duration_floor(
                    replay_audios,
                    int(sr),
                    texts,
                    method="generate_voice_clone",
                )
            cached = self._clone_replay_cache.get(replay_key)
            if cached is not None:
                entry = dict(cached)
                self._trace(
                    "request_replayed",
                    {
                        "method": "generate_voice_clone",
                        "request": payload,
                    },
                )
            else:
                text_key = str(match_text or "").strip()
                mapped = self._clone_entry_by_text.get(text_key)
                if mapped is None:
                    mapped = self._clone_entry_by_text.get(_normalize_match_text(text_key))
                if mapped is not None:
                    entry = dict(mapped)
                    self._clone_replay_cache[replay_key] = dict(entry)
                    self._trace(
                        "request_text_mapped",
                        {
                            "method": "generate_voice_clone",
                            "request": payload,
                        },
                    )
                else:
                    normalized_target = _normalize_match_text(text_key)

                    sequence_entries: List[Dict[str, Any]] = []
                    total_entries = len(self._clone_entries_ordered)
                    for start in range(total_entries):
                        if not self._clone_texts_ordered[start]:
                            continue
                        combined = self._clone_texts_ordered[start]
                        end = start
                        while True:
                            if combined == normalized_target:
                                sequence_entries = self._clone_entries_ordered[start : end + 1]
                                break
                            end += 1
                            if end >= total_entries:
                                break
                            next_text = self._clone_texts_ordered[end]
                            if not next_text:
                                break
                            combined = f"{combined} {next_text}"
                            if len(combined) > len(normalized_target) + 8:
                                break
                        if sequence_entries:
                            break

                    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text_key) if part.strip()]
                    if not sequence_entries and len(parts) > 1 and all(part in self._clone_entry_by_text for part in parts):
                        sequence_entries = [self._clone_entry_by_text[part] for part in parts]

                    if sequence_entries:
                        composed_audio: List[np.ndarray] = []
                        composed_sr = self._default_sample_rate
                        for part_entry in sequence_entries:
                            audios, sr = self._resolve_audio_specs(part_entry, 1)
                            composed_sr = int(sr)
                            composed_audio.append(audios[0])
                        if composed_audio:
                            gap = np.zeros(max(1, int(0.14 * composed_sr)), dtype=np.float32)
                            combined = composed_audio[0]
                            for segment in composed_audio[1:]:
                                combined = np.concatenate([combined, gap, segment])
                            self._clone_audio_replay_cache[replay_key] = (combined, composed_sr)
                            self._trace(
                                "request_text_composed",
                                {
                                    "method": "generate_voice_clone",
                                    "request": payload,
                                    "parts": parts if parts else [],
                                    "sequence_length": len(sequence_entries),
                                },
                            )
                            composed = [combined for _ in range(max(1, len(texts)))]
                            return self._apply_duration_floor(
                                composed,
                                composed_sr,
                                texts,
                                method="generate_voice_clone",
                            )
                    raise
        audios, sr = self._resolve_audio_specs(entry, len(texts))
        return self._apply_duration_floor(audios, sr, texts, method="generate_voice_clone")

    def generate_voice_design(self, kwargs: Dict[str, Any]):
        text = kwargs.get("text")
        texts = text if isinstance(text, list) else [text]
        payload = {
            "text": text,
            "language": kwargs.get("language"),
            "instruct": kwargs.get("instruct"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
        }
        entry = self._consume("generate_voice_design", payload)
        audios, sr = self._resolve_audio_specs(entry, len(texts))
        return self._apply_duration_floor(audios, sr, texts, method="generate_voice_design")
