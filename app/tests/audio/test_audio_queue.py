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
                                       item_started_callback=None, log_callback=None,
                                       neutral_narrator=False):
            captured["indices"] = list(indices)
            captured["generation_token"] = generation_token
            if log_callback:
                log_callback("Sub-batch 1/1 [clone speaker='Jordan'] active 15.0s; chunk_ids=[2, 3], uids=[u2, u3], text_chars=[120, 160], total_chars=280")
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
            self.assertTrue(
                any("Sub-batch 1/1 [clone speaker='Jordan'] active 15.0s" in entry for entry in app_module.process_state["audio"]["logs"])
            )
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
                                          generation_token=None, item_started_callback=None,
                                          neutral_narrator=False):
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

    def test_recompute_uses_cumulative_project_average_instead_of_rolling_window(self):
        with app_module.audio_queue_lock:
            app_module.process_state["audio"]["metrics"] = app_module._new_audio_metrics()
            app_module.audio_queue[:] = [{"remaining_words": 300}]
            app_module.audio_current_job = {"remaining_words": 200}

            metrics = app_module.process_state["audio"]["metrics"]
            metrics["rolling_seconds"] = 2.0
            metrics["rolling_input_words"] = 400
            metrics["rolling_output_words"] = 400
            metrics["total_elapsed_seconds"] = 100.0
            metrics["total_input_words"] = 1000
            metrics["total_output_words"] = 1000

            app_module._recompute_audio_metrics_locked()

            self.assertAlmostEqual(metrics["words_per_minute"], 600.0)
            self.assertAlmostEqual(metrics["estimated_remaining_seconds"], 50.0)

    def test_enqueue_preserves_project_metrics_when_restarting_from_idle(self):
        original_get_chunk_raw = app_module.project_manager.get_chunk_raw
        original_persist = app_module._persist_audio_queue_state_locked

        try:
            app_module.project_manager.get_chunk_raw = lambda ref: {
                "uid": str(ref),
                "text": "word " * 60,
            }
            app_module._persist_audio_queue_state_locked = lambda: None

            with app_module.audio_queue_lock:
                app_module.audio_queue[:] = []
                app_module.audio_current_job = None
                app_module.process_state["audio"]["metrics"] = app_module._new_audio_metrics()
                metrics = app_module.process_state["audio"]["metrics"]
                metrics["processed_clips"] = 10
                metrics["successful_clips"] = 10
                metrics["total_elapsed_seconds"] = 60.0
                metrics["total_input_words"] = 600
                metrics["total_output_words"] = 600
                app_module.process_state["audio"]["logs"] = ["existing log"]
                app_module.process_state["audio"]["recent_jobs"] = [{"id": 999}]

            result = app_module._enqueue_audio_job("parallel", ["u1"], label="Next pass")

            with app_module.audio_queue_lock:
                metrics = app_module.process_state["audio"]["metrics"]
                self.assertEqual(metrics["processed_clips"], 10)
                self.assertEqual(metrics["total_elapsed_seconds"], 60.0)
                self.assertEqual(metrics["total_input_words"], 600)
                self.assertAlmostEqual(metrics["words_per_minute"], 600.0)
                self.assertAlmostEqual(metrics["estimated_remaining_seconds"], 6.0)
                self.assertAlmostEqual(result["estimated_remaining_seconds"], 6.0)
        finally:
            app_module.project_manager.get_chunk_raw = original_get_chunk_raw
            app_module._persist_audio_queue_state_locked = original_persist

    def test_effective_parallel_workers_clamps_local_mlx_to_one(self):
        class FakeEngine:
            mode = "local"

            @property
            def local_backend(self):
                return "mlx"

        original_get_engine = app_module.project_manager.get_engine
        try:
            app_module.project_manager.get_engine = lambda: FakeEngine()
            workers = app_module._effective_parallel_workers(
                {"workers": 4, "tts_cfg": {"mode": "local", "local_backend": "auto"}}
            )
            self.assertEqual(workers, 1)
        finally:
            app_module.project_manager.get_engine = original_get_engine

    def test_normalize_restored_audio_job_runtime_requeues_stale_generation_finished_job(self):
        original_list_finalize = app_module.project_manager.list_audio_finalize_tasks
        try:
            app_module.project_manager.list_audio_finalize_tasks = lambda generation_token=None, statuses=None: []
            normalized = app_module._normalize_restored_audio_job_runtime(
                {
                    "run_token": "stale-run",
                    "generation_finished": True,
                    "generation_pending_uids": [],
                    "pending_finalize_uids": ["u1"],
                },
                {"pending_uids": ["u1"]},
            )
            self.assertEqual(normalized["generation_pending_uids"], ["u1"])
            self.assertEqual(normalized["pending_finalize_uids"], [])
            self.assertFalse(normalized["generation_finished"])
            self.assertIsNone(normalized["run_token"])
        finally:
            app_module.project_manager.list_audio_finalize_tasks = original_list_finalize

    def test_serialize_audio_job_reconciles_stale_pending_finalize_uids(self):
        original_list_finalize = app_module.project_manager.list_audio_finalize_tasks
        original_get_chunks = app_module.project_manager.get_chunks_by_uids
        try:
            app_module.project_manager.list_audio_finalize_tasks = lambda generation_token=None, statuses=None: []
            app_module.project_manager.get_chunks_by_uids = lambda uids: [
                {
                    "uid": "u1",
                    "id": 0,
                    "status": "done",
                }
            ]
            serialized = app_module._serialize_audio_job(
                {
                    "id": 17,
                    "corr_id": "audio-00017-stale-finalize",
                    "kind": "batch_fast",
                    "status": "running",
                    "label": "Stale finalize",
                    "scope": "custom",
                    "uids": ["u1"],
                    "pending_uids": ["u1"],
                    "generation_pending_uids": [],
                    "pending_finalize_uids": ["u1"],
                    "total_chunks": 1,
                    "total_words": 6,
                    "remaining_words": 6,
                    "processed_clips": 0,
                    "error_clips": 0,
                    "generation_finished": True,
                    "run_token": "stale-finalize-run",
                }
            )
            self.assertEqual(serialized["pending_uids"], [])
            self.assertEqual(serialized["pending_finalize_uids"], [])
        finally:
            app_module.project_manager.list_audio_finalize_tasks = original_list_finalize
            app_module.project_manager.get_chunks_by_uids = original_get_chunks

    def test_refresh_audio_process_state_uses_compact_job_summary(self):
        original_get_chunks = app_module.project_manager.get_chunks_by_uids
        original_get_coverage = app_module.project_manager.get_audio_coverage_summary
        try:
            def fail_get_chunks(_uids):
                raise AssertionError("refresh summary should not resolve full UID ordinals or reconcile all chunk rows")

            app_module.project_manager.get_chunks_by_uids = fail_get_chunks
            app_module.project_manager.get_audio_coverage_summary = lambda: {
                "total_clips": 3,
                "valid_clips": 1,
                "invalid_clips": 2,
                "percentage": 33,
            }

            with app_module.audio_queue_lock:
                app_module.audio_queue[:] = []
                app_module.audio_current_job = {
                    "id": 23,
                    "corr_id": "audio-00023-compact",
                    "kind": "batch_fast",
                    "status": "running",
                    "label": "Compact summary",
                    "scope": "project",
                    "scope_mode": "project",
                    "chapter": None,
                    "uids": ["u1", "u2", "u3"],
                    "pending_uids": ["u2", "u3"],
                    "generation_pending_uids": ["u2"],
                    "pending_finalize_uids": ["u3"],
                    "retry_uids": ["u3"],
                    "total_chunks": 3,
                    "total_words": 18,
                    "remaining_words": 12,
                    "processed_clips": 1,
                    "error_clips": 0,
                    "generation_finished": False,
                    "finalized_clips": 1,
                    "finalizer_failures": 0,
                    "recovery_count": 0,
                    "queued_at": 0.0,
                    "started_at": 1.0,
                    "finished_at": None,
                    "last_output_at": None,
                    "last_generation_activity_at": 2.0,
                    "last_finalize_activity_at": 3.0,
                    "run_token": "run-compact",
                }

                app_module._refresh_audio_process_state_locked(persist=False)
                summary = app_module.process_state["audio"]["current_job"]

            self.assertNotIn("uids", summary)
            self.assertNotIn("indices", summary)
            self.assertEqual(summary["pending_chunks"], 2)
            self.assertEqual(summary["generation_pending_chunks"], 1)
            self.assertEqual(summary["pending_finalize_chunks"], 1)
            self.assertEqual(summary["retry_chunks"], 1)
            self.assertEqual(summary["total_chunks"], 3)
        finally:
            app_module.project_manager.get_chunks_by_uids = original_get_chunks
            app_module.project_manager.get_audio_coverage_summary = original_get_coverage

    def test_audio_job_runner_skips_generation_when_only_finalization_remains(self):
        original_parallel = app_module.project_manager.generate_chunks_parallel
        original_batch = app_module.project_manager.generate_chunks_batch
        original_register = app_module.project_manager.register_audio_finalization_listener
        original_unregister = app_module.project_manager.unregister_audio_finalization_listener
        done_event = threading.Event()
        result_holder = {}
        job = {
            "id": 21,
            "corr_id": "audio-00021-finalize-only",
            "kind": "batch_fast",
            "uids": ["u9"],
            "pending_uids": ["u9"],
            "generation_pending_uids": [],
            "pending_finalize_uids": ["u9"],
            "generation_finished": True,
            "total_chunks": 1,
            "total_words": 6,
            "remaining_words": 6,
            "processed_clips": 0,
            "error_clips": 0,
            "status": "running",
            "label": "Finalize only",
            "scope": "custom",
            "run_token": "run-finalize-only",
            "queued_at": 0.0,
            "started_at": 1.0,
        }

        def fail_generate(*args, **kwargs):
            raise AssertionError("generation should not rerun when only finalization remains")

        try:
            app_module.project_manager.generate_chunks_parallel = fail_generate
            app_module.project_manager.generate_chunks_batch = fail_generate
            app_module.project_manager.register_audio_finalization_listener = lambda *args, **kwargs: None
            app_module.project_manager.unregister_audio_finalization_listener = lambda *args, **kwargs: None

            with app_module.audio_queue_lock:
                app_module.audio_current_job = job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()

            app_module._audio_job_runner(
                job,
                {
                    "workers": 4,
                    "batch_seed": -1,
                    "batch_size": 4,
                    "batch_group_by_type": False,
                    "tts_cfg": {"mode": "local", "local_backend": "auto"},
                },
                "run-finalize-only",
                result_holder,
                done_event,
            )

            self.assertTrue(done_event.is_set())
            self.assertEqual(result_holder["results"], {"completed": [], "failed": [], "cancelled": 0})
        finally:
            app_module.project_manager.generate_chunks_parallel = original_parallel
            app_module.project_manager.generate_chunks_batch = original_batch
            app_module.project_manager.register_audio_finalization_listener = original_register
            app_module.project_manager.unregister_audio_finalization_listener = original_unregister
            with app_module.audio_queue_lock:
                app_module.audio_current_job = self._backup_audio_current_job
                app_module.process_state["audio"]["cancel"] = False
                app_module.audio_cancel_event.clear()

if __name__ == "__main__":
    unittest.main()
