import tempfile
import threading
import unittest

from llm.lmstudio_runtime_coordinator import LMStudioRuntimeCoordinator
from llm.models import LLMRuntimeConfig


class _FakeLMStudioService:
    def __init__(self, *, model_name: str, target_loaded_instances: int, other_loaded_instances: int):
        self._model_name = model_name
        self._target_loaded_instances = int(target_loaded_instances)
        self._other_loaded_instances = int(other_loaded_instances)
        self._lock = threading.Lock()
        self.list_calls = 0
        self.unload_calls = 0
        self.load_calls = 0
        self.last_load_kwargs = None

    def list_models(self, *, base_url: str, api_key: str):
        with self._lock:
            self.list_calls += 1
            target_instances = [
                {"id": f"{self._model_name}-instance-{index + 1}"}
                for index in range(self._target_loaded_instances)
            ]
            other_instances = [
                {"id": f"other-instance-{index + 1}"}
                for index in range(self._other_loaded_instances)
            ]
        return {
            "models": [
                {
                    "key": self._model_name,
                    "display_name": self._model_name,
                    "loaded_instances": target_instances,
                },
                {
                    "key": "other-model",
                    "display_name": "other-model",
                    "loaded_instances": other_instances,
                },
            ]
        }

    def unload_all_models(self, *, base_url: str, api_key: str):
        with self._lock:
            self.unload_calls += 1
            self._target_loaded_instances = 0
            self._other_loaded_instances = 0
        return {"status": "ok", "total_loaded_instances": 0, "unloaded_instance_ids": []}

    def load_model(self, *, base_url: str, api_key: str, model_name: str, **kwargs):
        with self._lock:
            self.load_calls += 1
            self.last_load_kwargs = dict(kwargs)
            self._target_loaded_instances = 1
            self._other_loaded_instances = 0
        return {"status": "loaded", "type": "llm", "model": model_name}


class _FailingListService:
    def list_models(self, *, base_url: str, api_key: str):
        raise RuntimeError("endpoint offline")


class _NonLMService:
    def list_models(self, *, base_url: str, api_key: str):
        return {"data": []}


class LMStudioRuntimeCoordinatorTests(unittest.TestCase):
    def test_prepare_collapses_parallel_calls_to_single_load(self):
        runtime = LLMRuntimeConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="local",
            model_name="target-model",
        )
        service = _FakeLMStudioService(
            model_name="target-model",
            target_loaded_instances=0,
            other_loaded_instances=2,
        )

        with tempfile.TemporaryDirectory() as lock_dir:
            coordinator = LMStudioRuntimeCoordinator(
                service_factory=lambda _timeout: service,
                lock_dir=lock_dir,
                wait_timeout_seconds=5,
                poll_interval_seconds=0.01,
                ready_ttl_seconds=30,
            )

            results = []
            errors = []
            result_lock = threading.Lock()

            def _worker():
                try:
                    result = coordinator.ensure_ready(runtime)
                    with result_lock:
                        results.append(result)
                except Exception as exc:  # pragma: no cover - defensive
                    with result_lock:
                        errors.append(exc)

            threads = [threading.Thread(target=_worker) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 5)
            self.assertTrue(all(result.get("status") == "prepared" for result in results))
            self.assertEqual(service.unload_calls, 1)
            self.assertEqual(service.load_calls, 1)
            self.assertEqual(service.last_load_kwargs, {})

    def test_prepare_skips_when_target_already_loaded_once(self):
        runtime = LLMRuntimeConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="local",
            model_name="target-model",
        )
        service = _FakeLMStudioService(
            model_name="target-model",
            target_loaded_instances=1,
            other_loaded_instances=0,
        )

        with tempfile.TemporaryDirectory() as lock_dir:
            coordinator = LMStudioRuntimeCoordinator(
                service_factory=lambda _timeout: service,
                lock_dir=lock_dir,
                wait_timeout_seconds=5,
                poll_interval_seconds=0.01,
                ready_ttl_seconds=30,
            )
            result = coordinator.ensure_ready(runtime)

        self.assertEqual(result.get("status"), "prepared")
        self.assertEqual(result.get("reason"), "already_loaded")
        self.assertEqual(service.unload_calls, 0)
        self.assertEqual(service.load_calls, 0)

    def test_prepare_falls_back_when_probe_fails(self):
        runtime = LLMRuntimeConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="local",
            model_name="target-model",
        )
        with tempfile.TemporaryDirectory() as lock_dir:
            coordinator = LMStudioRuntimeCoordinator(
                service_factory=lambda _timeout: _FailingListService(),
                lock_dir=lock_dir,
                wait_timeout_seconds=2,
                poll_interval_seconds=0.01,
                fallback_ttl_seconds=1,
            )
            result = coordinator.ensure_ready(runtime)

        self.assertEqual(result.get("status"), "fallback")
        self.assertEqual(result.get("reason"), "preflight_error")

    def test_prepare_skips_non_lmstudio_payload(self):
        runtime = LLMRuntimeConfig(
            base_url="http://127.0.0.1:1234/v1",
            api_key="local",
            model_name="target-model",
        )
        with tempfile.TemporaryDirectory() as lock_dir:
            coordinator = LMStudioRuntimeCoordinator(
                service_factory=lambda _timeout: _NonLMService(),
                lock_dir=lock_dir,
                wait_timeout_seconds=2,
                poll_interval_seconds=0.01,
            )
            result = coordinator.ensure_ready(runtime)

        self.assertEqual(result.get("status"), "skipped")
        self.assertEqual(result.get("reason"), "non_lmstudio_endpoint")


if __name__ == "__main__":
    unittest.main()
