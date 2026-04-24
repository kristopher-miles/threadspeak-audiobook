"""LM Studio runtime coordinator for single-load preflight across workers."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .model_load_service import LMStudioModelLoadService
from .models import LLMRuntimeConfig
from .tool_capability_service import ToolCapabilityService

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on some platforms
    fcntl = None


@dataclass
class _RuntimePreflightState:
    in_flight: bool = False
    ready_until: float = 0.0
    fallback_until: float = 0.0
    last_result: Dict[str, Any] = field(default_factory=dict)


class LMStudioRuntimeCoordinator:
    """Coordinate LM Studio model preflight so workers do not race model loads."""

    def __init__(
        self,
        *,
        probe_timeout_seconds: int = 2,
        load_timeout_seconds: int = 240,
        wait_timeout_seconds: int = 240,
        poll_interval_seconds: float = 0.5,
        ready_ttl_seconds: float = 30.0,
        fallback_ttl_seconds: float = 5.0,
        wait_for_inflight_seconds: float = 300.0,
        lock_dir: Optional[str] = None,
        service_factory: Optional[Callable[[int], LMStudioModelLoadService]] = None,
    ):
        self._probe_timeout_seconds = max(1, int(probe_timeout_seconds))
        self._load_timeout_seconds = max(1, int(load_timeout_seconds))
        self._wait_timeout_seconds = max(1, int(wait_timeout_seconds))
        self._poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self._ready_ttl_seconds = max(0.0, float(ready_ttl_seconds))
        self._fallback_ttl_seconds = max(0.0, float(fallback_ttl_seconds))
        self._wait_for_inflight_seconds = max(1.0, float(wait_for_inflight_seconds))
        self._lock_dir = lock_dir or os.path.join(tempfile.gettempdir(), "threadspeak-lmstudio-locks")
        self._service_factory = service_factory

        self._state_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._states: Dict[Tuple[str, str], _RuntimePreflightState] = {}
        self._origin_thread_locks: Dict[str, threading.Lock] = {}

    def ensure_ready(self, runtime: LLMRuntimeConfig) -> Dict[str, Any]:
        """Ensure LM Studio has exactly one loaded target model before worker fan-out."""
        base_url = str(getattr(runtime, "base_url", "") or "").strip()
        model_name = str(getattr(runtime, "model_name", "") or "").strip()
        if not base_url or not model_name:
            return {"status": "skipped", "reason": "missing_runtime"}
        if ToolCapabilityService.is_openrouter_url(base_url):
            return {"status": "skipped", "reason": "openrouter"}

        origin = ToolCapabilityService.normalize_lm_studio_origin(base_url)
        if not origin:
            return {"status": "skipped", "reason": "missing_origin"}

        key = self._runtime_key(runtime, origin)
        now = time.time()

        with self._state_condition:
            state = self._states.setdefault(key, _RuntimePreflightState())
            if state.ready_until > now:
                return dict(state.last_result)
            if state.fallback_until > now:
                return dict(state.last_result)

            if state.in_flight:
                wait_deadline = now + self._wait_for_inflight_seconds
                while state.in_flight and time.time() < wait_deadline:
                    self._state_condition.wait(timeout=0.2)
                now = time.time()
                if state.ready_until > now:
                    return dict(state.last_result)
                if state.fallback_until > now:
                    return dict(state.last_result)
                if state.in_flight:
                    result = {
                        "status": "fallback",
                        "reason": "wait_timeout",
                        "message": "Timed out waiting for LM Studio preflight lease.",
                    }
                    state.last_result = result
                    state.fallback_until = now + self._fallback_ttl_seconds
                    return dict(result)

            state.in_flight = True

        result: Dict[str, Any]
        try:
            result = self._prepare_with_cross_process_lock(runtime, origin)
        except Exception as exc:  # pragma: no cover - safety net
            result = {
                "status": "fallback",
                "reason": "preflight_error",
                "message": str(exc),
            }
        finally:
            with self._state_condition:
                state = self._states.setdefault(key, _RuntimePreflightState())
                state.in_flight = False
                state.last_result = dict(result)
                now = time.time()
                if result.get("status") == "prepared":
                    state.ready_until = now + self._ready_ttl_seconds
                    state.fallback_until = 0.0
                elif result.get("status") == "fallback":
                    state.fallback_until = now + self._fallback_ttl_seconds
                    state.ready_until = 0.0
                else:
                    state.ready_until = 0.0
                    state.fallback_until = 0.0
                self._state_condition.notify_all()
        return dict(result)

    def _prepare_with_cross_process_lock(self, runtime: LLMRuntimeConfig, origin: str) -> Dict[str, Any]:
        with self._origin_lock(origin):
            return self._prepare_once(runtime, origin)

    def _prepare_once(self, runtime: LLMRuntimeConfig, origin: str) -> Dict[str, Any]:
        model_name = str(runtime.model_name or "").strip()
        api_key = str(runtime.api_key or "")

        models_payload = self._list_models(origin=origin, api_key=api_key)
        models = models_payload.get("models")
        if not isinstance(models, list):
            return {
                "status": "skipped",
                "reason": "non_lmstudio_endpoint",
                "message": "Provider does not expose LM Studio model state endpoint.",
            }

        total_loaded, target_loaded = self._loaded_instance_counts(models, model_name)
        if total_loaded == 1 and target_loaded == 1:
            return {
                "status": "prepared",
                "reason": "already_loaded",
                "origin": origin,
                "model_name": model_name,
                "total_loaded_instances": total_loaded,
            }

        service = self._new_service(self._load_timeout_seconds)
        try:
            service.unload_all_models(base_url=origin, api_key=api_key)
            payload = service.load_model(
                base_url=origin,
                api_key=api_key,
                model_name=model_name,
            )
        except Exception as exc:
            return {
                "status": "fallback",
                "reason": "load_failed",
                "message": f"LM Studio preflight failed: {exc}",
            }

        status = str((payload or {}).get("status") or "").strip().lower()
        if status and status != "loaded":
            return {
                "status": "fallback",
                "reason": "unexpected_load_status",
                "message": f"LM Studio returned unexpected load status: {status}",
            }

        if not self._wait_for_target_loaded(origin=origin, api_key=api_key, model_name=model_name):
            return {
                "status": "fallback",
                "reason": "wait_timeout",
                "message": f"Timed out waiting for LM Studio model '{model_name}' to finish loading.",
            }

        return {
            "status": "prepared",
            "reason": "loaded",
            "origin": origin,
            "model_name": model_name,
            "total_loaded_instances": 1,
        }

    def _wait_for_target_loaded(self, *, origin: str, api_key: str, model_name: str) -> bool:
        deadline = time.time() + self._wait_timeout_seconds
        while time.time() < deadline:
            try:
                payload = self._list_models(origin=origin, api_key=api_key)
            except Exception:
                time.sleep(self._poll_interval_seconds)
                continue

            models = payload.get("models") if isinstance(payload, dict) else None
            if isinstance(models, list):
                total_loaded, target_loaded = self._loaded_instance_counts(models, model_name)
                if total_loaded == 1 and target_loaded == 1:
                    return True
            time.sleep(self._poll_interval_seconds)
        return False

    def _list_models(self, *, origin: str, api_key: str) -> Dict[str, Any]:
        service = self._new_service(self._probe_timeout_seconds)
        return service.list_models(base_url=origin, api_key=api_key)

    def _new_service(self, timeout_seconds: int) -> LMStudioModelLoadService:
        if self._service_factory is not None:
            return self._service_factory(int(timeout_seconds))
        return LMStudioModelLoadService(timeout_seconds=int(timeout_seconds))

    @staticmethod
    def _loaded_instance_counts(models: list, model_name: str) -> Tuple[int, int]:
        total_loaded = 0
        target_loaded = 0
        for model in models:
            if not isinstance(model, dict):
                continue
            loaded_instances = [
                entry
                for entry in (model.get("loaded_instances") or [])
                if isinstance(entry, dict) and str(entry.get("id") or "").strip()
            ]
            loaded_count = len(loaded_instances)
            total_loaded += loaded_count
            if ToolCapabilityService.model_name_matches_lm_studio(model, model_name):
                target_loaded += loaded_count
        return total_loaded, target_loaded

    @staticmethod
    def _runtime_key(runtime: LLMRuntimeConfig, origin: str) -> Tuple[str, str]:
        return (
            str(origin or "").strip().lower(),
            str(getattr(runtime, "model_name", "") or "").strip(),
        )

    @contextlib.contextmanager
    def _origin_lock(self, origin: str):
        os.makedirs(self._lock_dir, exist_ok=True)
        lock_name = hashlib.sha1(str(origin or "").encode("utf-8")).hexdigest()
        lock_path = os.path.join(self._lock_dir, f"{lock_name}.lock")

        with self._state_lock:
            thread_lock = self._origin_thread_locks.setdefault(origin, threading.Lock())

        with thread_lock:
            handle = open(lock_path, "a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                handle.close()
