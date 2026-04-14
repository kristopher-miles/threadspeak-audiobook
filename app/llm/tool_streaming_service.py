"""Streaming helper for forced tool-call workflows."""

import json
import re
from typing import Any, Dict, List, Optional

from .errors import LLMTransportError
from .models import ToolStreamResult


class ToolStreamingService:
    """Stream chat-completions tool calls and extract the first parseable tool args."""

    def stream_required_tool_call(
        self,
        *,
        client: Any,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.1,
        tool_choice: Any = "required",
        parallel_tool_calls: bool = False,
        reasoning_parameter_name: Optional[str] = None,
    ) -> ToolStreamResult:
        tool_call_args = ""
        reasoning_content = ""
        text_content = ""

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:
            raise LLMTransportError(f"Failed to start streamed LLM request: {exc}") from exc

        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content += rc
                content = getattr(delta, "content", None)
                if content:
                    text_content += content

                delta_tool_calls = getattr(delta, "tool_calls", None) or []
                if delta_tool_calls:
                    function = getattr(delta_tool_calls[0], "function", None)
                    frag = getattr(function, "arguments", None) if function is not None else None
                    if frag:
                        tool_call_args += frag

                parsed = self._try_parse_json(tool_call_args)
                if parsed is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    return ToolStreamResult(
                        parsed_arguments=parsed,
                        raw_payload=tool_call_args,
                        text_content=text_content,
                        reasoning_content=reasoning_content,
                    )
        except Exception as exc:
            raise LLMTransportError(f"Streamed LLM request failed: {exc}") from exc

        parsed = self._try_parse_json(tool_call_args)
        if parsed is not None:
            return ToolStreamResult(
                parsed_arguments=parsed,
                raw_payload=tool_call_args,
                text_content=text_content,
                reasoning_content=reasoning_content,
            )

        if reasoning_parameter_name and reasoning_content:
            extracted = self._extract_reasoning_parameter(reasoning_content, reasoning_parameter_name)
            if extracted:
                return ToolStreamResult(
                    parsed_arguments={reasoning_parameter_name: extracted},
                    raw_payload=reasoning_content,
                    text_content=text_content,
                    reasoning_content=reasoning_content,
                )

        return ToolStreamResult(
            parsed_arguments=None,
            raw_payload=tool_call_args or reasoning_content or text_content,
            text_content=text_content,
            reasoning_content=reasoning_content,
        )

    @staticmethod
    def _try_parse_json(payload: str) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _extract_reasoning_parameter(reasoning_content: str, parameter_name: str) -> str:
        pattern = rf"<parameter={re.escape(parameter_name)}>\s*(.*?)\s*</parameter>"
        match = re.search(pattern, reasoning_content, re.DOTALL | re.IGNORECASE)
        return (match.group(1).strip() if match else "")
