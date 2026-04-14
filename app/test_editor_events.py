import unittest
import asyncio

import app as app_module
from api.routers import editor_audio_router as router_module


class EditorEventsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with app_module.audio_queue_lock:
            self._backup_audio_state = dict(app_module.process_state["audio"])
            self._backup_current_job = app_module.audio_current_job
            app_module.process_state["audio"] = {
                **app_module.process_state["audio"],
                "running": False,
                "queue": [],
                "current_job": None,
                "recent_jobs": [],
                "logs": [],
                "metrics": app_module._new_audio_metrics(),
                "heartbeat": app_module._new_audio_heartbeat_state(),
            }
            app_module.audio_current_job = None

    def tearDown(self):
        with app_module.audio_queue_lock:
            app_module.process_state["audio"] = self._backup_audio_state
            app_module.audio_current_job = self._backup_current_job

    async def test_editor_events_streams_broker_messages(self):
        response = await router_module.editor_events(scope_mode="project")
        iterator = response.body_iterator

        first = await iterator.__anext__()
        second = await iterator.__anext__()
        self.assertIn("event: chapter_list_changed", first)
        self.assertIn("event: audio_status", second)
        self.assertIn('"audio_coverage"', second)

        router_module.chunk_event_broker.publish(
            "chunk_upsert",
            {"uid": "chunk-1", "chapter": "Chapter 1"},
        )
        matched = None
        for _ in range(5):
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
            if "event: chunk_upsert" in chunk:
                matched = chunk
                break
        self.assertIsNotNone(matched)
        self.assertIn('"uid": "chunk-1"', matched)

        await iterator.aclose()


if __name__ == "__main__":
    unittest.main()
