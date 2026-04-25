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
        request = editor_audio_router.BatchGenerateRequest(indices=["chunk-1"], neutral_narrator=True)
        with (
            mock.patch.object(editor_audio_router, "_attempt_lmstudio_unload_all_models") as unload_mock,
            mock.patch.object(editor_audio_router, "_resolve_batch_target_rows", return_value=[{"uid": "chunk-1", "text": "line"}]),
            mock.patch.object(editor_audio_router, "_raise_if_generation_voice_issue"),
            mock.patch.object(editor_audio_router, "_load_audio_worker_settings", return_value={"workers": 2}),
            mock.patch.object(editor_audio_router, "_enqueue_audio_job", return_value={"status": "started"}) as enqueue_mock,
        ):
            result = asyncio.run(editor_audio_router.generate_batch_endpoint(request, BackgroundTasks()))

        self.assertEqual(result["status"], "started")
        self.assertEqual(result["workers"], 2)
        unload_mock.assert_not_called()
        self.assertTrue(enqueue_mock.call_args.kwargs["neutral_narrator"])

    def test_generate_batch_fast_does_not_call_legacy_unload_helper(self):
        request = editor_audio_router.BatchGenerateRequest(indices=["chunk-1"], neutral_narrator=True)
        with (
            mock.patch.object(editor_audio_router, "_attempt_lmstudio_unload_all_models") as unload_mock,
            mock.patch.object(editor_audio_router, "_resolve_batch_target_rows", return_value=[{"uid": "chunk-1", "text": "line"}]),
            mock.patch.object(editor_audio_router, "_raise_if_generation_voice_issue"),
            mock.patch.object(editor_audio_router, "_load_audio_worker_settings", return_value={"batch_seed": 7, "batch_size": 4}),
            mock.patch.object(editor_audio_router, "_enqueue_audio_job", return_value={"status": "started"}) as enqueue_mock,
        ):
            result = asyncio.run(editor_audio_router.generate_batch_fast_endpoint(request, BackgroundTasks()))

        self.assertEqual(result["status"], "started")
        self.assertEqual(result["batch_seed"], 7)
        self.assertEqual(result["batch_size"], 4)
        unload_mock.assert_not_called()
        self.assertTrue(enqueue_mock.call_args.kwargs["neutral_narrator"])

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


class EditorAudioVoiceDesignUnloadTests(unittest.TestCase):
    def setUp(self):
        self.original_audio_current_job = shared_module.audio_current_job

    def tearDown(self):
        shared_module.audio_current_job = self.original_audio_current_job

    @staticmethod
    def _job(kind="parallel"):
        return {
            "id": 99,
            "corr_id": "audio-test",
            "kind": kind,
            "uids": ["chunk-1"],
            "pending_uids": ["chunk-1"],
            "generation_pending_uids": ["chunk-1"],
            "pending_finalize_uids": [],
            "generation_finished": False,
            "processed_clips": 0,
            "error_clips": 0,
            "finalizer_failures": 0,
            "retry_uids": [],
            "run_token": "run-token",
        }

    def test_audio_job_runner_unloads_voice_design_before_parallel_generation(self):
        job = self._job("parallel")
        done_event = shared_module.threading.Event()
        result_holder = {}
        manager = mock.Mock()
        manager.generate_chunks_parallel.return_value = {"completed": ["chunk-1"], "failed": [], "cancelled": 0}
        settings = {"workers": 2, "batch_seed": -1, "batch_size": 2, "batch_group_by_type": False, "tts_cfg": {}}

        with (
            mock.patch.object(shared_module, "project_manager", manager),
            mock.patch.object(shared_module, "audio_current_job", job),
            mock.patch.object(shared_module, "_append_audio_log"),
            mock.patch.object(shared_module, "_append_audio_log_locked"),
            mock.patch.object(shared_module, "record_audio_perf"),
        ):
            shared_module._audio_job_runner(job, settings, job["run_token"], result_holder, done_event)

        self.assertTrue(done_event.is_set())
        self.assertEqual(result_holder["results"], {"completed": ["chunk-1"], "failed": [], "cancelled": 0})
        manager.unload_voice_design_model.assert_called_once()
        manager.generate_chunks_parallel.assert_called_once()
        self.assertLess(
            manager.method_calls.index(mock.call.unload_voice_design_model()),
            manager.method_calls.index(mock.call.generate_chunks_parallel(
                ["chunk-1"],
                2,
                mock.ANY,
                cancel_check=mock.ANY,
                item_callback=mock.ANY,
                generation_token="run-token",
                item_started_callback=mock.ANY,
                neutral_narrator=False,
            )),
        )

    def test_audio_job_runner_unloads_voice_design_before_batch_generation(self):
        job = self._job("batch_fast")
        done_event = shared_module.threading.Event()
        result_holder = {}
        manager = mock.Mock()
        manager.generate_chunks_batch.return_value = {"completed": ["chunk-1"], "failed": [], "cancelled": 0}
        settings = {"workers": 2, "batch_seed": 7, "batch_size": 4, "batch_group_by_type": True, "tts_cfg": {}}

        with (
            mock.patch.object(shared_module, "project_manager", manager),
            mock.patch.object(shared_module, "audio_current_job", job),
            mock.patch.object(shared_module, "_append_audio_log"),
            mock.patch.object(shared_module, "_append_audio_log_locked"),
            mock.patch.object(shared_module, "record_audio_perf"),
        ):
            shared_module._audio_job_runner(job, settings, job["run_token"], result_holder, done_event)

        self.assertTrue(done_event.is_set())
        self.assertEqual(result_holder["results"], {"completed": ["chunk-1"], "failed": [], "cancelled": 0})
        manager.unload_voice_design_model.assert_called_once()
        manager.generate_chunks_batch.assert_called_once()
        self.assertLess(
            manager.method_calls.index(mock.call.unload_voice_design_model()),
            manager.method_calls.index(mock.call.generate_chunks_batch(
                ["chunk-1"],
                7,
                4,
                mock.ANY,
                batch_group_by_type=True,
                cancel_check=mock.ANY,
                item_callback=mock.ANY,
                generation_token="run-token",
                item_started_callback=mock.ANY,
                log_callback=mock.ANY,
                neutral_narrator=False,
            )),
        )

    def test_audio_job_runner_fails_before_generation_when_voice_design_unload_fails(self):
        job = self._job("parallel")
        done_event = shared_module.threading.Event()
        result_holder = {}
        manager = mock.Mock()
        manager.unload_voice_design_model.side_effect = RuntimeError("designer still loaded")
        settings = {"workers": 2, "batch_seed": -1, "batch_size": 2, "batch_group_by_type": False, "tts_cfg": {}}

        with (
            mock.patch.object(shared_module, "project_manager", manager),
            mock.patch.object(shared_module, "audio_current_job", job),
            mock.patch.object(shared_module, "_append_audio_log") as append_log,
            mock.patch.object(shared_module, "_append_audio_log_locked"),
            mock.patch.object(shared_module, "record_audio_perf"),
        ):
            shared_module._audio_job_runner(job, settings, job["run_token"], result_holder, done_event)

        self.assertTrue(done_event.is_set())
        self.assertIn("designer still loaded", result_holder["error"])
        manager.generate_chunks_parallel.assert_not_called()
        self.assertTrue(any("designer still loaded" in str(call.args[0]) for call in append_log.call_args_list))


if __name__ == "__main__":
    unittest.main()
