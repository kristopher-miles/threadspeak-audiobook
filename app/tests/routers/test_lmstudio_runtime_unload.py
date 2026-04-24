import asyncio
import unittest
from unittest import mock

from fastapi import BackgroundTasks

from api import shared as shared_module
from api.routers import editor_audio_router, voices_router


class LMStudioUnloadHelperTests(unittest.TestCase):
    def test_unload_helper_skips_when_llm_base_missing(self):
        with mock.patch.object(shared_module, "_read_runtime_llm_settings", return_value={}):
            result = shared_module._attempt_lmstudio_unload_all_models("proofread")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "missing_base_url")

    def test_unload_helper_skips_for_openrouter(self):
        with mock.patch.object(
            shared_module,
            "_read_runtime_llm_settings",
            return_value={"base_url": "https://openrouter.ai/api/v1", "api_key": "sk-test"},
        ):
            result = shared_module._attempt_lmstudio_unload_all_models("render_pending")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "openrouter")

    def test_unload_helper_handles_unreachable_target(self):
        with (
            mock.patch.object(
                shared_module,
                "_read_runtime_llm_settings",
                return_value={"base_url": "http://127.0.0.1:1234/v1", "api_key": "local"},
            ),
            mock.patch.object(shared_module, "LMStudioModelLoadService") as service_cls,
        ):
            service_cls.return_value.unload_all_models.side_effect = RuntimeError("offline")
            result = shared_module._attempt_lmstudio_unload_all_models("voices_bulk_generation")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "unreachable")


class LMStudioUnloadRouterWiringTests(unittest.TestCase):
    def test_voices_preflight_endpoint_calls_central_helper(self):
        with mock.patch.object(
            voices_router,
            "_attempt_lmstudio_unload_all_models",
            return_value={"status": "ok", "reason": "unloaded"},
        ) as unload_mock:
            result = asyncio.run(voices_router.preflight_unload_lmstudio_models_for_voice_generation())

        self.assertEqual(result["status"], "ok")
        unload_mock.assert_called_once_with("voices_bulk_generation")

    def test_generate_batch_does_not_call_legacy_unload_helper(self):
        request = editor_audio_router.BatchGenerateRequest(indices=["chunk-1"])
        with (
            mock.patch.object(editor_audio_router, "_attempt_lmstudio_unload_all_models") as unload_mock,
            mock.patch.object(editor_audio_router, "_resolve_batch_target_rows", return_value=[{"uid": "chunk-1", "text": "line"}]),
            mock.patch.object(editor_audio_router, "_raise_if_generation_voice_issue"),
            mock.patch.object(editor_audio_router, "_load_audio_worker_settings", return_value={"workers": 2}),
            mock.patch.object(editor_audio_router, "_enqueue_audio_job", return_value={"status": "started"}),
        ):
            result = asyncio.run(editor_audio_router.generate_batch_endpoint(request, BackgroundTasks()))

        self.assertEqual(result["status"], "started")
        self.assertEqual(result["workers"], 2)
        unload_mock.assert_not_called()

    def test_generate_batch_fast_does_not_call_legacy_unload_helper(self):
        request = editor_audio_router.BatchGenerateRequest(indices=["chunk-1"])
        with (
            mock.patch.object(editor_audio_router, "_attempt_lmstudio_unload_all_models") as unload_mock,
            mock.patch.object(editor_audio_router, "_resolve_batch_target_rows", return_value=[{"uid": "chunk-1", "text": "line"}]),
            mock.patch.object(editor_audio_router, "_raise_if_generation_voice_issue"),
            mock.patch.object(editor_audio_router, "_load_audio_worker_settings", return_value={"batch_seed": 7, "batch_size": 4}),
            mock.patch.object(editor_audio_router, "_enqueue_audio_job", return_value={"status": "started"}),
        ):
            result = asyncio.run(editor_audio_router.generate_batch_fast_endpoint(request, BackgroundTasks()))

        self.assertEqual(result["status"], "started")
        self.assertEqual(result["batch_seed"], 7)
        self.assertEqual(result["batch_size"], 4)
        unload_mock.assert_not_called()

    def test_proofread_does_not_call_legacy_unload_helper(self):
        request = editor_audio_router.ProofreadRequest(chapter=None, threshold=0.75)
        with (
            mock.patch.object(editor_audio_router, "_attempt_lmstudio_unload_all_models") as unload_mock,
            mock.patch.object(editor_audio_router, "_any_project_task_running", return_value=None),
            mock.patch.object(editor_audio_router, "_start_task_run", return_value="proofread-run"),
        ):
            result = asyncio.run(editor_audio_router.start_proofread(request, BackgroundTasks()))

        self.assertEqual(result["status"], "started")
        self.assertEqual(result["run_id"], "proofread-run")
        unload_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
