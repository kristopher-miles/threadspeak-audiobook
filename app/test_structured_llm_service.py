import unittest
from types import SimpleNamespace

from llm.contracts import SCRIPT_ENTRIES_CONTRACT
from llm.models import ChatCompletionResult, LLMRuntimeConfig, ToolCapabilityResult
from llm.structured_service import StructuredLLMService


class StructuredLLMServiceTests(unittest.TestCase):
    def test_extract_tool_arguments_from_tool_calls(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    arguments='{"voice":"Warm and measured"}'
                                )
                            )
                        ],
                        reasoning_content="",
                    )
                )
            ]
        )

        payload = StructuredLLMService._extract_tool_arguments(response)
        self.assertEqual(payload, {"voice": "Warm and measured"})

    def test_extract_tool_arguments_from_reasoning_content_tags(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[],
                        reasoning_content=(
                            "<tool_call>\n"
                            "<function=submit_script_entries>\n"
                            "<parameter=entries>\n"
                            '[{"speaker":"NARRATOR","text":"Hello","instruct":"Calm."}]'
                            "\n</parameter>\n"
                            "</function>\n"
                            "</tool_call>"
                        ),
                    )
                )
            ]
        )

        payload = StructuredLLMService._extract_tool_arguments(response)
        self.assertIsInstance(payload, dict)
        self.assertIn("entries", payload)
        self.assertIsInstance(payload["entries"], list)
        self.assertEqual(payload["entries"][0]["speaker"], "NARRATOR")

    def test_extract_tool_arguments_returns_none_when_missing(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[], reasoning_content=""))]
        )
        self.assertIsNone(StructuredLLMService._extract_tool_arguments(response))

    def test_run_retries_empty_length_tool_response_with_larger_token_budget(self):
        class _FakeCapabilityService:
            def verify_tool_capability(self, base_url, api_key, model_name):
                return ToolCapabilityResult(
                    status="supported",
                    provider="lmstudio",
                    message="supported",
                )

        def _raw_with_tool_arguments(arguments_json: str):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=arguments_json)
                                )
                            ],
                            reasoning_content="",
                        )
                    )
                ]
            )

        def _raw_without_tool_arguments():
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[],
                            reasoning_content="",
                        )
                    )
                ]
            )

        class _FakeChatService:
            def __init__(self):
                self.calls = []
                self._responses = [
                    ChatCompletionResult(
                        text="",
                        finish_reason="length",
                        raw_response=_raw_without_tool_arguments(),
                    ),
                    ChatCompletionResult(
                        text="",
                        finish_reason="tool_calls",
                        raw_response=_raw_with_tool_arguments(
                            '{"entries":[{"speaker":"NARRATOR","text":"Hello","instruct":"Calm"}]}'
                        ),
                    ),
                ]

            def complete(self, *, client, model_name, params):
                self.calls.append(params)
                return self._responses.pop(0)

        chat_service = _FakeChatService()
        service = StructuredLLMService(
            chat_service=chat_service,
            capability_service=_FakeCapabilityService(),
        )

        result = service.run(
            client=object(),
            runtime=LLMRuntimeConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="local",
                model_name="google/gemma-4-26b-a4b",
            ),
            messages=[{"role": "user", "content": "Test"}],
            contract=SCRIPT_ENTRIES_CONTRACT,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(result.mode, "tool")
        self.assertTrue(result.tool_call_observed)
        self.assertIsInstance(result.parsed, list)
        self.assertEqual(len(result.parsed), 1)
        self.assertEqual(result.parsed[0]["speaker"], "NARRATOR")
        self.assertEqual(len(chat_service.calls), 2)
        self.assertEqual(chat_service.calls[0].max_tokens, 512)
        self.assertEqual(chat_service.calls[1].max_tokens, 1024)


if __name__ == "__main__":
    unittest.main()
