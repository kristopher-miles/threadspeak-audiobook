import copy
import unittest

import app as app_module


class AudioQueueMetricsTests(unittest.TestCase):
    def setUp(self):
        with app_module.audio_queue_lock:
            self._backup_process_audio = copy.deepcopy(app_module.process_state["audio"])
            self._backup_audio_queue = copy.deepcopy(app_module.audio_queue)
            self._backup_audio_current_job = copy.deepcopy(app_module.audio_current_job)

    def tearDown(self):
        with app_module.audio_queue_lock:
            app_module.process_state["audio"] = self._backup_process_audio
            app_module.audio_queue[:] = self._backup_audio_queue
            app_module.audio_current_job = self._backup_audio_current_job

    def test_refresh_preserves_sample_buffer_for_tracker_updates(self):
        with app_module.audio_queue_lock:
            app_module.audio_queue[:] = []
            app_module.audio_current_job = {
                "id": 1,
                "kind": "parallel",
                "status": "running",
                "label": "Test job",
                "scope": "custom",
                "indices": [3],
                "total_chunks": 1,
                "total_words": 6,
                "remaining_words": 6,
                "pending_indices": [3],
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
            app_module._record_audio_sample_locked(job, 3, 2.0, 6, 6, True)

            metrics = app_module.process_state["audio"]["metrics"]
            self.assertEqual(metrics["processed_clips"], 1)
            self.assertEqual(metrics["successful_clips"], 1)
            self.assertEqual(metrics["error_clips"], 0)
            self.assertEqual(len(metrics["samples"]), 1)
            self.assertEqual(job["pending_indices"], [])
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
                    "indices": [10, 11, 12],
                    "pending_indices": [10, 11, 12],
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
                self.assertEqual(reset_calls[0]["indices"], [10, 11, 12])
                self.assertEqual(reset_calls[0]["generation_token"], "run-123")
                self.assertEqual(app_module.process_state["audio"]["recent_jobs"][0]["status"], "cancelled")
                self.assertIn("reset 3 generating chunk(s)", app_module.process_state["audio"]["logs"][-1])
        finally:
            app_module.project_manager.reset_generating_chunks = original_reset


if __name__ == "__main__":
    unittest.main()
