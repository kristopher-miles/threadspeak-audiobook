"""Adaptive structured-output LLM service with per-model tool/json strategy caching."""

from __future__ import annotations

import json
import threading
import re
from typing import Any, Dict, List, Optional

from .chat_service import ChatCompletionService
from .models import (
    ChatCompletionParams,
    LLMRuntimeConfig,
    StructuredLLMResult,
    StructuredOutputContract,
)
from .tool_capability_service import ToolCapabilityService
from .tool_streaming_service import ToolStreamingService


class StructuredLLMService:
    """Route structured requests through tool-calling or JSON fallback per model."""

    def __init__(
        self,
        *,
        chat_service: Optional[ChatCompletionService] = None,
        tool_streaming_service: Optional[ToolStreamingService] = None,
        capability_service: Optional[ToolCapabilityService] = None,
    ):
        self._chat_service = chat_service or ChatCompletionService()
        self._tool_streaming_service = tool_streaming_service or ToolStreamingService()
        self._capability_service = capability_service or ToolCapabilityService()
        self._strategy_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._strategy_cache = {}

    def run(
        self,
        *,
        client: Any,
        runtime: LLMRuntimeConfig,
        messages: List[Dict[str, Any]],
        contract: StructuredOutputContract,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        use_streaming_tool: bool = False,
        reasoning_parameter_name: Optional[str] = None,
    ) -> StructuredLLMResult:
        sanitized_messages = self._sanitize_messages(messages)
        key = self._strategy_cache_key(runtime)
        strategy = self._cached_strategy(key)

        if strategy is None:
            capability = self._capability_service.verify_tool_capability(
                runtime.base_url,
                runtime.api_key,
                runtime.model_name,
            )
            if capability.status == "supported":
                strategy = "tool"
                self._set_cached_strategy(key, strategy)
            elif capability.status == "unsupported":
                strategy = "json"
                self._set_cached_strategy(key, strategy)
            else:
                strategy = "unknown"

        if strategy in ("tool", "unknown"):
            tool_result = self._run_tool_mode(
                client=client,
                runtime=runtime,
                messages=sanitized_messages,
                contract=contract,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body=extra_body,
                use_streaming_tool=use_streaming_tool,
                reasoning_parameter_name=reasoning_parameter_name,
            )
            if tool_result is not None:
                self._set_cached_strategy(key, "tool")
                return tool_result

            # Unknown policy and runtime failure fallback: try JSON and cache fallback.
            json_result = self._run_json_mode(
                client=client,
                runtime=runtime,
                messages=sanitized_messages,
                contract=contract,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body=extra_body,
            )
            self._set_cached_strategy(key, "json")
            return json_result

        json_result = self._run_json_mode(
            client=client,
            runtime=runtime,
            messages=sanitized_messages,
            contract=contract,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
        )
        return json_result

    def _run_tool_mode(
        self,
        *,
        client: Any,
        runtime: LLMRuntimeConfig,
        messages: List[Dict[str, Any]],
        contract: StructuredOutputContract,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        presence_penalty: Optional[float],
        extra_body: Optional[Dict[str, Any]],
        use_streaming_tool: bool,
        reasoning_parameter_name: Optional[str],
    ) -> Optional[StructuredLLMResult]:
        tool_messages = self._with_instruction(messages, contract.tool_instruction)
        tool_spec = [{
            "type": "function",
            "function": {
                "name": contract.tool_name,
                "description": f"Return structured data for {contract.name}.",
                "parameters": contract.tool_schema,
            },
        }]

        if use_streaming_tool:
            try:
                stream_result = self._tool_streaming_service.stream_required_tool_call(
                    client=client,
                    model_name=runtime.model_name,
                    messages=tool_messages,
                    tools=tool_spec,
                    tool_choice="required",
                    parallel_tool_calls=False,
                    temperature=0.1 if temperature is None else float(temperature),
                    max_tokens=int(max_tokens or 512),
                    reasoning_parameter_name=reasoning_parameter_name,
                )
            except Exception:
                return None

            parsed = self._unwrap_contract_payload(contract, stream_result.parsed_arguments)
            if self._matches_contract(parsed, contract):
                return StructuredLLMResult(
                    mode="tool",
                    parsed=parsed,
                    text=self._payload_to_text(parsed),
                    raw_payload=stream_result.raw_payload,
                    tool_call_observed=True,
                    finish_reason=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                )
            return None

        try:
            completion = self._chat_service.complete(
                client=client,
                model_name=runtime.model_name,
                params=ChatCompletionParams(
                    messages=tool_messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    tools=tool_spec,
                    tool_choice="required",
                    parallel_tool_calls=False,
                ),
            )
        except Exception:
            return None

        args_payload = self._extract_tool_arguments(completion.raw_response)
        parsed = self._unwrap_contract_payload(contract, args_payload)
        if self._matches_contract(parsed, contract):
            return StructuredLLMResult(
                mode="tool",
                parsed=parsed,
                text=self._payload_to_text(parsed),
                raw_payload=self._payload_to_text(args_payload),
                tool_call_observed=True,
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        # Some models ignore tool-calling and still return valid JSON text.
        parsed_text = self._parse_json_like_payload(completion.text, contract.root_type)
        if self._matches_contract(parsed_text, contract):
            return StructuredLLMResult(
                mode="tool",
                parsed=parsed_text,
                text=self._payload_to_text(parsed_text),
                raw_payload=completion.text,
                tool_call_observed=False,
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        return None

    def _run_json_mode(
        self,
        *,
        client: Any,
        runtime: LLMRuntimeConfig,
        messages: List[Dict[str, Any]],
        contract: StructuredOutputContract,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        presence_penalty: Optional[float],
        extra_body: Optional[Dict[str, Any]],
    ) -> StructuredLLMResult:
        json_messages = self._with_instruction(messages, contract.json_instruction)
        completion = self._chat_service.complete(
            client=client,
            model_name=runtime.model_name,
            params=ChatCompletionParams(
                messages=json_messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ),
        )

        parsed = self._parse_json_like_payload(completion.text, contract.root_type)
        if not self._matches_contract(parsed, contract):
            parsed = None

        return StructuredLLMResult(
            mode="json",
            parsed=parsed,
            text=self._payload_to_text(parsed) if parsed is not None else (completion.text or ""),
            raw_payload=completion.text or "",
            tool_call_observed=False,
            finish_reason=completion.finish_reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )

    @staticmethod
    def _with_instruction(messages: List[Dict[str, Any]], instruction: str) -> List[Dict[str, Any]]:
        if not messages:
            return [{"role": "user", "content": instruction.strip()}]

        normalized = [dict(message or {}) for message in messages]
        index = len(normalized) - 1
        content = str(normalized[index].get("content") or "").strip()
        normalized[index]["content"] = f"{content}\n\n{instruction.strip()}" if content else instruction.strip()
        return normalized

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for message in messages or []:
            normalized = dict(message or {})
            content = normalized.get("content")
            if isinstance(content, str):
                normalized["content"] = StructuredLLMService._sanitize_prompt_text(content)
            sanitized.append(normalized)
        return sanitized

    @staticmethod
    def _sanitize_prompt_text(text: str) -> str:
        if not text:
            return ""
        value = str(text)
        line_patterns = (
            r"(?im)^\s*You MUST call the identify_(?:dialogue|sentiment) tool[^\n]*\n?",
            r"(?im)^\s*Output ONLY valid JSON arrays[^\n]*\n?",
            r"(?im)^\s*OUTPUT FORMAT:\s*A JSON array[^\n]*\n?",
            r"(?im)^\s*Return your final answer as JSON[^\n]*\n?",
            r"(?im)^\s*Make sure the value of \"voice\"[^\n]*\n?",
            r"(?im)^\s*Reply with JSON like\s*\{\"result\":\"TRUE\"\}\s*or\s*\{\"result\":\"FALSE\"\}[^\n]*\n?",
        )
        for pattern in line_patterns:
            value = re.sub(pattern, "", value)
        return value.strip()

    @staticmethod
    def _extract_tool_arguments(raw_response: Any) -> Optional[Dict[str, Any]]:
        choices = getattr(raw_response, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return None
        function = getattr(tool_calls[0], "function", None)
        if function is None:
            return None
        arguments = getattr(function, "arguments", None)
        if not arguments:
            return None

        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _payload_to_text(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _unwrap_contract_payload(contract: StructuredOutputContract, payload: Any) -> Any:
        if payload is None:
            return None
        if contract.unwrap_field and isinstance(payload, dict):
            return payload.get(contract.unwrap_field)
        return payload

    @staticmethod
    def _matches_contract(payload: Any, contract: StructuredOutputContract) -> bool:
        if contract.root_type == "array":
            return isinstance(payload, list)
        if contract.root_type == "object":
            return isinstance(payload, dict)
        return False

    @staticmethod
    def _parse_json_like_payload(text: str, expected_root_type: str) -> Optional[Any]:
        raw = (text or "").strip()
        if not raw:
            return None

        direct = StructuredLLMService._try_json_load(raw)
        if StructuredLLMService._matches_root(direct, expected_root_type):
            return direct

        if "```" in raw:
            fence_payload = StructuredLLMService._extract_fenced_code(raw)
            if fence_payload:
                parsed = StructuredLLMService._try_json_load(fence_payload)
                if StructuredLLMService._matches_root(parsed, expected_root_type):
                    return parsed

        bracketed = StructuredLLMService._extract_first_json_root(raw, expected_root_type)
        if bracketed:
            parsed = StructuredLLMService._try_json_load(bracketed)
            if StructuredLLMService._matches_root(parsed, expected_root_type):
                return parsed

        return None

    @staticmethod
    def _try_json_load(payload: str) -> Optional[Any]:
        try:
            return json.loads(payload)
        except Exception:
            return None

    @staticmethod
    def _matches_root(payload: Any, root_type: str) -> bool:
        if root_type == "array":
            return isinstance(payload, list)
        if root_type == "object":
            return isinstance(payload, dict)
        return False

    @staticmethod
    def _extract_fenced_code(text: str) -> Optional[str]:
        start = text.find("```")
        if start < 0:
            return None
        end = text.find("```", start + 3)
        if end < 0:
            return None
        block = text[start + 3:end]
        block = block.strip()
        if block.lower().startswith("json"):
            block = block[4:].strip()
        return block or None

    @staticmethod
    def _extract_first_json_root(text: str, root_type: str) -> Optional[str]:
        opener = "[" if root_type == "array" else "{"
        closer = "]" if root_type == "array" else "}"
        start = text.find(opener)
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    @staticmethod
    def _strategy_cache_key(runtime: LLMRuntimeConfig) -> str:
        return f"{runtime.base_url}|{runtime.model_name}"

    def _cached_strategy(self, cache_key: str) -> Optional[str]:
        with self._cache_lock:
            return self._strategy_cache.get(cache_key)

    def _set_cached_strategy(self, cache_key: str, strategy: str) -> None:
        with self._cache_lock:
            self._strategy_cache[cache_key] = strategy
