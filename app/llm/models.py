"""Typed request/response models for LLM integration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Normalized runtime settings for OpenAI-compatible endpoints."""

    base_url: str
    api_key: str
    model_name: str
    timeout: Optional[float] = 600.0
    llm_workers: int = 1

    @staticmethod
    def normalize_base_url(base_url: str, default_base_url: str) -> str:
        url = str(base_url or default_base_url).strip().rstrip("/")
        if not url:
            url = str(default_base_url or "http://localhost:11434/v1").strip().rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    @classmethod
    def from_dict(
        cls,
        llm_config: Optional[Dict[str, Any]],
        *,
        default_base_url: str = "http://localhost:11434/v1",
        default_api_key: str = "local",
        default_model_name: str = "local-model",
        default_timeout: Optional[float] = 600.0,
        default_workers: int = 1,
    ) -> "LLMRuntimeConfig":
        cfg = dict(llm_config or {})

        timeout = cfg.get("timeout", default_timeout)
        if timeout is None:
            timeout_value: Optional[float] = None
        else:
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError):
                timeout_value = default_timeout

        try:
            workers = max(1, int(cfg.get("llm_workers", default_workers) or default_workers))
        except (TypeError, ValueError):
            workers = max(1, int(default_workers or 1))

        return cls(
            base_url=cls.normalize_base_url(cfg.get("base_url", ""), default_base_url),
            api_key=str(cfg.get("api_key", default_api_key) or default_api_key),
            model_name=str(cfg.get("model_name", default_model_name) or default_model_name),
            timeout=timeout_value,
            llm_workers=workers,
        )

    @classmethod
    def from_app_config(
        cls,
        config: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMRuntimeConfig":
        return cls.from_dict((config or {}).get("llm") or {}, **kwargs)


@dataclass(frozen=True)
class ChatCompletionParams:
    """Provider-agnostic parameters for a chat-completions request."""

    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    stream: bool = False


@dataclass(frozen=True)
class ChatCompletionResult:
    """Normalized non-stream response fields consumed across the app."""

    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw_response: Optional[Any] = None


@dataclass(frozen=True)
class ToolStreamResult:
    """Parsed result of a streamed tool-call response."""

    parsed_arguments: Optional[Dict[str, Any]] = None
    raw_payload: str = ""
    text_content: str = ""
    reasoning_content: str = ""


@dataclass(frozen=True)
class ToolCapabilityResult:
    """Normalized capability verification response for setup UI and APIs."""

    status: str
    provider: str
    message: str

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "supported": self.supported,
            "provider": self.provider,
            "message": self.message,
        }


@dataclass(frozen=True)
class StructuredOutputContract:
    """Structured output contract for adaptive tool/json calls."""

    name: str
    root_type: str  # "object" or "array"
    tool_name: str
    tool_schema: Dict[str, Any]
    json_instruction: str
    tool_instruction: str
    unwrap_field: Optional[str] = None


@dataclass(frozen=True)
class StructuredLLMResult:
    """Normalized structured-result payload from adaptive LLM execution."""

    mode: str  # "tool" or "json"
    parsed: Optional[Any]
    text: str
    raw_payload: str
    tool_call_observed: Optional[bool] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
