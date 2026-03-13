import copy
import importlib.util
import os
import tempfile
import unittest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SPEC = importlib.util.spec_from_file_location("alexandria_app_module", MODULE_PATH)
app_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(app_module)


class ProcessingWorkflowTests(unittest.TestCase):
    def setUp(self):
        with app_module.processing_workflow_lock:
            self._backup_workflow = copy.deepcopy(app_module.process_state["processing_workflow"])
        self._temp_dir = tempfile.TemporaryDirectory()
        self._workflow_state_path = os.path.join(self._temp_dir.name, "processing_workflow_state.json")
        self._backup_workflow_state_path = app_module.PROCESSING_WORKFLOW_STATE_PATH
        app_module.PROCESSING_WORKFLOW_STATE_PATH = self._workflow_state_path

    def tearDown(self):
        with app_module.processing_workflow_lock:
            app_module.process_state["processing_workflow"] = self._backup_workflow
        app_module.PROCESSING_WORKFLOW_STATE_PATH = self._backup_workflow_state_path
        self._temp_dir.cleanup()

    def test_stage_sequence_respects_toggles(self):
        self.assertEqual(
            app_module._processing_workflow_stage_sequence({"process_voices": True, "generate_audio": False}),
            ["script", "review", "sanity", "repair", "voices"],
        )
        self.assertEqual(
            app_module._processing_workflow_stage_sequence({"process_voices": False, "generate_audio": True}),
            ["script", "review", "sanity", "repair", "audio"],
        )

    def test_pause_request_marks_state(self):
        with app_module.processing_workflow_lock:
            app_module.process_state["processing_workflow"] = app_module._new_processing_workflow_state() | {
                "running": True,
                "current_stage": "voices",
            }
            requested = app_module._request_processing_workflow_pause_locked()
            state = copy.deepcopy(app_module.process_state["processing_workflow"])

        self.assertTrue(requested)
        self.assertTrue(state["pause_requested"])
        self.assertTrue(any("Pause requested" in line for line in state["logs"]))

    def test_mark_stage_complete_updates_completed_stages(self):
        with app_module.processing_workflow_lock:
            app_module.process_state["processing_workflow"] = app_module._new_processing_workflow_state() | {
                "running": True,
                "options": {"process_voices": True, "generate_audio": True},
            }
            app_module._mark_processing_workflow_stage_complete("script")
            state = copy.deepcopy(app_module.process_state["processing_workflow"])

        self.assertEqual(state["completed_stages"], ["script"])
        self.assertEqual(state["current_stage"], "script")
        self.assertTrue(any("Generate Annotated Script completed." in line for line in state["logs"]))


if __name__ == "__main__":
    unittest.main()
