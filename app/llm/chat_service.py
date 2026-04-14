"""Non-streamed chat-completions service helpers."""

from typing import Any, Dict

from .errors import LLMResponseParseError, LLMTransportError
from .models import ChatCompletionParams, ChatCompletionResult


class ChatCompletionService:
    """Execute non-stream chat completions with consistent request/response mapping."""

    def complete(self, *, client: Any, model_name: str, params: ChatCompletionParams) -> ChatCompletionResult:
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": params.messages,
        }

        optional_fields = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "presence_penalty": params.presence_penalty,
            "max_tokens": params.max_tokens,
            "extra_body": params.extra_body,
            "tools": params.tools,
            "tool_choice": params.tool_choice,
            "parallel_tool_calls": params.parallel_tool_calls,
            "stream": True if params.stream else None,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value

        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMTransportError(f"Failed to call LLM provider: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        text = getattr(message, "content", "") if message is not None else ""
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise LLMResponseParseError("LLM response content was not a string")

        usage = getattr(response, "usage", None)
        return ChatCompletionResult(
            text=text,
            finish_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
            raw_response=response,
        )
