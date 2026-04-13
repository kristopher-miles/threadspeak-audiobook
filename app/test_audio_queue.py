import copy
import threading
import time
import unittest

import app as app_module


class AudioQueueMetricsTests(unittest.TestCase):
    def setUp(self):
        with app_module.audio_queue_lock:
            self._backup_process_audio = copy.deepcopy(app_module.process_state["audio"])
            self._backup_audio_queue = copy.deepcopy(app_module.audio_queue)
            self._backup_audio_current_job = copy.deepcopy(app_module.audio_current_job)
            self._backup_audio_cancel_event_set = app_module.audio_cancel_event.is_set()

    def tearDown(self):
        with app_module.audio_queue_lock:
            app_module.process_state["audio"] = self._backup_process_audio
            app_module.audio_queue[:] = self._backup_audio_queue
            app_module.audio_current_job = self._backup_audio_current_job
            if self._backup_audio_cancel_event_set:
                app_module.audio_cancel_event.set()
            else:
                app_module.audio_cancel_event.clear()

    def test_refresh_preserves_sample_buffer_for_tracker_updates(self):
        with app_module.audio_queue_lock:
            app_module.audio_queue[:] = []
            app_module.audio_current_job = {
                "id": 1,
                "kind": "parallel",
                "status": "running",
                "label": "Test job",
                "scope": "custom",
                "uids": ["u3"],
                "total_chunks": 1,
                "total_words": 6,
                "remaining_words": 6,
                "pending_uids": ["u3"],
                "processed_clips": 0,
                "error_clips": 0,
                "queued_at": 0.0,
                "started_at": 0.0,
                "finished_at": None,
                "last_output_at": None,
            }
            app_module.process_state["audio"]["metrics"] = app_module._new_audio_metrics()
            app_module.process_state["audio"]["heartbeat"] = app_module._new_audio_heartbeat_state()

            # This matches the normal queue refresh path used by /api/status/audio.
            app_module._refresh_audio_process_state_locked(persist=False)

            job = app_module.audio_current_job
            app_module._record_audio_sample_locked(job, "u3", 2.0, 6, 6, True)

            metrics = app_module.process_state["audio"]["metrics"]
            self.assertEqual(metrics["processed_clips"], 1)
            self.assertEqual(metrics["successful_clips"], 1)
            self.assertEqual(metrics["error_clips"], 0)
            self.assertEqual(len(metrics["samples"]), 1)
            self.assertEqual(job["pending_uids"], [])
            self.assertEqual(job["remaining_words"], 0)
            self.assertGreater(metrics["words_per_minute"], 0)
            self.assertEqual(metrics["estimated_remaining_seconds"], 0.0)

    def test_abandon_audio_job_clears_active_job_and_resets_generating_chunks(self):
        reset_calls = []

        def fake_reset_generating_chunks(indices=None, generation_token=None, target_status="pending"):
            reset_calls.append({
                "indices": list(indices or []),
                "generation_token": generation_token,
                "target_status": target_status,
            })
            return 3

        original_reset = app_module.project_manager.reset_generating_chunks
        app_module.project_manager.reset_generating_chunks = fake_reset_generating_chunks
        try:
            with app_module.audio_queue_lock:
                job = {
                    "id": 7,
                    "kind": "parallel",
                    "status": "running",
                    "label": "Test job",
                    "scope": "custom",
                    "uids": ["u10", "u11", "u12"],
                    "pending_uids": ["u10", "u11", "u12"],
                    "total_chunks": 3,
                    "total_words": 30,
                    "remaining_words": 30,
                    "processed_clips": 0,
                    "error_clips": 0,
                    "queued_at": 0.0,
                    "started_at": 1.0,
                    "finished_at": None,
                    "last_output_at": None,
                    "run_token": "run-123",
                }
                app_module.audio_queue[:] = []
                app_module.audio_current_job = job
                app_module.process_state["audio"]["cancel"] = True
                app_module.process_state["audio"]["recent_jobs"] = []
                app_module.process_state["audio"]["logs"] = []

                abandoned = app_module._abandon_audio_job_locked(
                    job,
                    "run-123",
                    "User requested cancellation",
                    status="cancelled",
                )

                self.assertTrue(abandoned)
                self.assertIsNone(app_module.audio_current_job)
                self.assertFalse(app_module.process_state["audio"]["cancel"])
                self.assertEqual(len(reset_calls), 1)
                self.assertEqual(reset_calls[0]["indices"], ["u10", "u11", "u12"])
                self.assertEqual(reset_calls[0]["generation_token"], "run-123")
                self.assertEqual(app_module.process_state["audio"]["recent_jobs"][0]["status"], "cancelled")
                self.assertIn("reset 3 generating chunk(s)", app_module.process_state["audio"]["logs"][-1])
        finally:
            app_module.project_manager.reset_generating_chunks = original_reset

    def test_restored_audio_job_runs_only_pending_indices(self):
        captured = {}

        def fake_generate_chunks_batch(indices, batch_seed, batch_size, progress_callback=None,
                                       batch_group_by_type=False, cancel_check=None,
                                       item_callback=None, generation_token=None,
                                       item_started_callback=None):
            captured["indices"] = list(indices)
            captured["generation_token"] = generation_token
            return {"completed": [], "failed": [], "cancelled": 0}

        original_generate = app_module.project_manager.generate_chunks_batch
        app_module.project_manager.generate_chunks_batch = fake_generate_chunks_batch
        done_event = threading.Event()
        job = {
            "id": 42,
            "corr_id": "audio-00042-test",
            "kind": "batch_fast",
            "uids": ["u0", "u1", "u2", "u3"],
            "pending_uids": ["u2", "u3"],
            "total_chunks": 4,
            "total_words": 40,
            "remaining_words": 20,
            "processed_clips": 2,
            "error_clips": 0,
            "status": "running",
            "label": "Restored job",
            "scope": "custom",
            "run_token": "run-restored",
            "queued_at": 0.0,
        }
        result_holder = {}

        try:
            with app_module.audio_queue_lock:
                app_module.audio_current_job = job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()

            app_module._audio_job_runner(
                job,
                {
                    "batch_seed": -1,
                    "batch_size": 2,
                    "batch_group_by_type": False,
                    "workers": 2,
                },
                "run-restored",
                result_holder,
                done_event,
            )

            self.assertTrue(done_event.is_set())
            self.assertEqual(captured["indices"], ["u2", "u3"])
            self.assertEqual(captured["generation_token"], "run-restored")
        finally:
            app_module.project_manager.generate_chunks_batch = original_generate
            with app_module.audio_queue_lock:
                app_module.audio_current_job = self._backup_audio_current_job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()

    def test_parallel_success_waits_for_finalize_before_counting_metrics(self):
        callbacks = {}

        def fake_register_audio_finalization_listener(
            generation_token,
            *,
            submission_callback=None,
            item_callback=None,
            activity_callback=None,
        ):
            callbacks["submission"] = submission_callback
            callbacks["item"] = item_callback
            callbacks["activity"] = activity_callback

        def fake_unregister_audio_finalization_listener(_generation_token):
            callbacks.clear()

        def fake_generate_chunks_parallel(indices, max_workers=2, progress_callback=None,
                                          cancel_check=None, item_callback=None,
                                          generation_token=None, item_started_callback=None):
            self.assertEqual(indices, ["u3"])
            self.assertEqual(generation_token, "run-finalize")
            self.assertIsNotNone(item_callback)
            self.assertIsNotNone(item_started_callback)
            self.assertEqual(job["processed_clips"], 0)
            self.assertEqual(job["remaining_words"], 6)

            item_started_callback("u3", time.time() - 60.0)
            item_callback("u3", True, 2.0, 6, 6)
            self.assertEqual(job["processed_clips"], 0)
            self.assertEqual(job["remaining_words"], 6)

            callbacks["submission"]("u3", {"id": 99})
            callbacks["item"]("u3", True, 0.02, 6, 6, {})

            self.assertEqual(job["processed_clips"], 1)
            self.assertEqual(job["remaining_words"], 0)
            return {"completed": ["u3"], "failed": [], "cancelled": 0}

        original_generate = app_module.project_manager.generate_chunks_parallel
        original_register = app_module.project_manager.register_audio_finalization_listener
        original_unregister = app_module.project_manager.unregister_audio_finalization_listener
        done_event = threading.Event()
        job = {
            "id": 9,
            "corr_id": "audio-00009-finalize",
            "kind": "parallel",
            "uids": ["u3"],
            "pending_uids": ["u3"],
            "generation_pending_uids": ["u3"],
            "pending_finalize_uids": [],
            "total_chunks": 1,
            "total_words": 6,
            "remaining_words": 6,
            "processed_clips": 0,
            "error_clips": 0,
            "status": "running",
            "label": "Finalize metrics job",
            "scope": "custom",
            "run_token": "run-finalize",
            "queued_at": 0.0,
            "started_at": 1.0,
            "last_output_at": None,
            "last_generation_activity_at": 1.0,
            "last_finalize_activity_at": None,
        }
        result_holder = {}

        try:
            app_module.project_manager.generate_chunks_parallel = fake_generate_chunks_parallel
            app_module.project_manager.register_audio_finalization_listener = fake_register_audio_finalization_listener
            app_module.project_manager.unregister_audio_finalization_listener = fake_unregister_audio_finalization_listener

            with app_module.audio_queue_lock:
                app_module.audio_current_job = job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()
                app_module.process_state["audio"]["metrics"] = app_module._new_audio_metrics()

            app_module._audio_job_runner(
                job,
                {
                    "workers": 2,
                    "batch_seed": -1,
                    "batch_size": 2,
                    "batch_group_by_type": False,
                },
                "run-finalize",
                result_holder,
                done_event,
            )

            self.assertTrue(done_event.is_set())
            metrics = app_module.process_state["audio"]["metrics"]
            self.assertEqual(metrics["processed_clips"], 1)
            self.assertEqual(metrics["successful_clips"], 1)
            self.assertEqual(len(metrics["samples"]), 1)
            self.assertGreater(metrics["total_elapsed_seconds"], 1.5)
            self.assertLess(metrics["total_elapsed_seconds"], 5.0)
            self.assertEqual(job["pending_uids"], [])
            self.assertEqual(job["pending_finalize_uids"], [])
        finally:
            app_module.project_manager.generate_chunks_parallel = original_generate
            app_module.project_manager.register_audio_finalization_listener = original_register
            app_module.project_manager.unregister_audio_finalization_listener = original_unregister
            with app_module.audio_queue_lock:
                app_module.audio_current_job = self._backup_audio_current_job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()

if __name__ == "__main__":
    unittest.main()
