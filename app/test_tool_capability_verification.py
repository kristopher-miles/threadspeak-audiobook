import asyncio
import unittest
from unittest import mock

from api.routers import config_router


class ToolCapabilityVerificationTests(unittest.TestCase):
    def test_openrouter_supported_model_match(self):
        with mock.patch.object(config_router, "_request_json", return_value={
            "data": [{"id": "openai/gpt-4.1"}, {"id": "anthropic/claude-sonnet-4"}]
        }):
            result = config_router.verify_tool_capability(
                "https://openrouter.ai/api/v1",
                "sk-test",
                "anthropic/claude-sonnet-4",
            )
        self.assertEqual(result["status"], "supported")
        self.assertTrue(result["supported"])
        self.assertEqual(result["provider"], "openrouter")

    def test_openrouter_absent_model_is_unsupported(self):
        with mock.patch.object(config_router, "_request_json", return_value={
            "data": [{"id": "openai/gpt-4.1"}]
        }):
            result = config_router.verify_tool_capability(
                "https://openrouter.ai/api/v1",
                "sk-test",
                "meta-llama/llama-3",
            )
        self.assertEqual(result["status"], "unsupported")
        self.assertFalse(result["supported"])

    def test_openrouter_failure_is_unknown(self):
        with mock.patch.object(config_router, "_request_json", side_effect=RuntimeError("boom")):
            result = config_router.verify_tool_capability(
                "https://openrouter.ai/api/v1",
                "sk-test",
                "openai/gpt-4.1",
            )
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["supported"])
        self.assertIn("Could not verify", result["message"])

    def test_lm_studio_supported_model_by_key(self):
        payload = {
            "models": [{
                "key": "qwen3-tool",
                "display_name": "Qwen 3 Tool",
                "capabilities": {"trained_for_tool_use": True},
                "loaded_instances": [],
            }]
        }
        with mock.patch.object(config_router, "_request_json", return_value=payload):
            result = config_router.verify_tool_capability(
                "http://localhost:1234/v1",
                "local",
                "qwen3-tool",
            )
        self.assertEqual(result["status"], "supported")
        self.assertTrue(result["supported"])
        self.assertEqual(result["provider"], "lmstudio")

    def test_lm_studio_unsupported_model_by_display_name(self):
        payload = {
            "models": [{
                "key": "gemma-3",
                "display_name": "Gemma 3",
                "capabilities": {"trained_for_tool_use": False},
                "loaded_instances": [],
            }]
        }
        with mock.patch.object(config_router, "_request_json", return_value=payload):
            result = config_router.verify_tool_capability(
                "http://127.0.0.1:1234/v1",
                "",
                "Gemma 3",
            )
        self.assertEqual(result["status"], "unsupported")
        self.assertFalse(result["supported"])

    def test_lm_studio_matches_loaded_instance_id(self):
        payload = {
            "models": [{
                "key": "downloaded-key",
                "display_name": "Downloaded Model",
                "capabilities": {"trained_for_tool_use": True},
                "loaded_instances": [{"id": "currently-loaded-id"}],
            }]
        }
        with mock.patch.object(config_router, "_request_json", return_value=payload):
            result = config_router.verify_tool_capability(
                "http://localhost:1234/v1",
                "",
                "currently-loaded-id",
            )
        self.assertEqual(result["status"], "supported")

    def test_lm_studio_failure_is_unknown(self):
        with mock.patch.object(config_router, "_request_json", side_effect=RuntimeError("offline")):
            result = config_router.verify_tool_capability(
                "http://localhost:1234/v1",
                "",
                "qwen3-tool",
            )
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["supported"])
        self.assertIn("Could not verify", result["message"])

    def test_lm_studio_model_load_posts_expected_payload(self):
        with mock.patch.object(config_router, "_post_json", return_value={"status": "loaded"}) as mock_post:
            result = config_router.load_lmstudio_model(
                base_url="http://localhost:1234/v1",
                api_key="local",
                model_name="qwen/qwen3.5-9b",
                context_length=8192,
                flash_attention=True,
                echo_load_config=True,
            )
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(mock_post.call_count, 1)
        call_args = mock_post.call_args[0]
        self.assertEqual(call_args[0], "http://localhost:1234/api/v1/models/load")
        self.assertEqual(call_args[1]["model"], "qwen/qwen3.5-9b")
        self.assertEqual(call_args[1]["context_length"], 8192)
        self.assertTrue(call_args[1]["flash_attention"])
        self.assertTrue(call_args[1]["echo_load_config"])

    def test_lm_studio_model_load_endpoint_uses_saved_defaults(self):
        with (
            mock.patch.object(
                config_router,
                "_read_saved_llm_config",
                return_value={
                    "base_url": "http://127.0.0.1:1234/v1",
                    "api_key": "local",
                    "model_name": "saved/model",
                },
            ),
            mock.patch.object(config_router, "load_lmstudio_model", return_value={"status": "loaded"}) as load_mock,
        ):
            result = asyncio.run(
                config_router.load_lmstudio_model_endpoint(
                    config_router.LMStudioModelLoadRequest(context_length=4096)
                )
            )
        self.assertEqual(result["status"], "loaded")
        load_mock.assert_called_once()
        kwargs = load_mock.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(kwargs["api_key"], "local")
        self.assertEqual(kwargs["model_name"], "saved/model")
        self.assertEqual(kwargs["context_length"], 4096)

    def test_lm_studio_model_load_endpoint_requires_model(self):
        with mock.patch.object(config_router, "_read_saved_llm_config", return_value={"base_url": "http://127.0.0.1:1234/v1"}):
            with self.assertRaises(config_router.HTTPException) as ctx:
                asyncio.run(
                    config_router.load_lmstudio_model_endpoint(
                        config_router.LMStudioModelLoadRequest()
                    )
                )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Model name is required", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
