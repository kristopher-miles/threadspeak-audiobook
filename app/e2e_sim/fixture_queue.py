"""Fixture-backed scripted interaction queues for E2E simulations."""

from __future__ import annotations

import copy
import json
import os
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


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


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.translate(_TEXT_EQUIVALENCE_MAP)
    return " ".join(normalized.split())


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in TRUE_VALUES


def load_fixture_payload(path: str) -> Dict[str, Any]:
    target = str(path or "").strip()
    if not target:
        raise ValueError("fixture path is required")
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("fixture payload must be a JSON object")
    return payload


def _partial_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if not _partial_match(expected_value, actual[key]):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return False
        for index, expected_item in enumerate(expected):
            if not _partial_match(expected_item, actual[index]):
                return False
        return True

    if isinstance(expected, str) and isinstance(actual, str):
        if expected == actual:
            return True
        return _normalize_text(expected) == _normalize_text(actual)

    return expected == actual


@dataclass
class ConsumedInteraction:
    key: str
    expect: Any
    request: Any
    response: Any


class ScriptedInteractionQueue:
    """Thread-safe per-key FIFO queue with optional strict request matching."""

    def __init__(
        self,
        *,
        routes: Dict[str, List[Dict[str, Any]]],
        strict: bool = True,
        unordered_keys: Optional[List[str]] = None,
    ):
        self._strict = bool(strict)
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[Dict[str, Any]]] = {
            str(key): deque([dict(item or {}) for item in (items or [])])
            for key, items in (routes or {}).items()
        }
        self._unordered_keys = {str(key).strip() for key in (unordered_keys or []) if str(key).strip()}
        self._consumed: List[ConsumedInteraction] = []

    @property
    def strict(self) -> bool:
        return self._strict

    def pending_counts(self) -> Dict[str, int]:
        with self._lock:
            return {key: len(queue) for key, queue in self._queues.items() if queue}

    def consume(self, key: str, request_payload: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        channel = str(key or "").strip()
        if not channel:
            raise ValueError("interaction key is required")

        with self._lock:
            queue = self._queues.get(channel)
            if not queue:
                if self._strict:
                    raise AssertionError(f"Unexpected interaction for '{channel}' (no scripted entries remain)")
                return None

            entry = dict(queue[0])
            expected = entry.get("expect")
            if expected is not None and not _partial_match(expected, request_payload):
                if channel in self._unordered_keys and len(queue) > 1:
                    matched_index = -1
                    matched_entry: Optional[Dict[str, Any]] = None
                    for idx, candidate in enumerate(queue):
                        candidate_expected = (candidate or {}).get("expect")
                        if candidate_expected is None or _partial_match(candidate_expected, request_payload):
                            matched_index = idx
                            matched_entry = dict(candidate or {})
                            break
                    if matched_entry is not None and matched_index >= 0:
                        del queue[matched_index]
                        if not queue:
                            self._queues.pop(channel, None)
                        consumed = ConsumedInteraction(
                            key=channel,
                            expect=copy.deepcopy(matched_entry.get("expect")),
                            request=copy.deepcopy(request_payload),
                            response=copy.deepcopy(matched_entry.get("response")),
                        )
                        self._consumed.append(consumed)
                        return matched_entry
                if self._strict:
                    raise AssertionError(
                        f"Interaction mismatch for '{channel}'. expected subset={expected!r}, actual={request_payload!r}"
                    )
                return None

            queue.popleft()
            if not queue:
                self._queues.pop(channel, None)

            consumed = ConsumedInteraction(
                key=channel,
                expect=copy.deepcopy(expected),
                request=copy.deepcopy(request_payload),
                response=copy.deepcopy(entry.get("response")),
            )
            self._consumed.append(consumed)
            return entry

    def assert_all_consumed(self, *, context: str = "scenario") -> None:
        pending = self.pending_counts()
        if pending:
            raise AssertionError(f"Unconsumed scripted interactions for {context}: {pending}")

    def consumed(self) -> List[ConsumedInteraction]:
        with self._lock:
            return list(self._consumed)
