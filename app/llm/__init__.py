"""Unified LLM integration package for API and pipeline call sites."""

from .chat_service import ChatCompletionService
from .client_factory import LLMClientFactory
from .contracts import (
    ATTRIBUTION_DECISION_CONTRACT,
    DIALOGUE_SPEAKER_CONTRACT,
    SCRIPT_ENTRIES_CONTRACT,
    SENTIMENT_MOOD_CONTRACT,
    VOICE_DESCRIPTION_CONTRACT,
)
from .errors import LLMResponseParseError, LLMServiceError, LLMTransportError
from .model_load_service import LMStudioModelLoadService
from .models import (
    ChatCompletionParams,
    ChatCompletionResult,
    LLMRuntimeConfig,
    StructuredLLMResult,
    StructuredOutputContract,
    ToolCapabilityResult,
    ToolStreamResult,
)
from .structured_service import StructuredLLMService
from .tool_capability_service import ToolCapabilityService
from .tool_streaming_service import ToolStreamingService

__all__ = [
    "ATTRIBUTION_DECISION_CONTRACT",
    "ChatCompletionParams",
    "ChatCompletionResult",
    "ChatCompletionService",
    "DIALOGUE_SPEAKER_CONTRACT",
    "LLMClientFactory",
    "LMStudioModelLoadService",
    "LLMRuntimeConfig",
    "LLMResponseParseError",
    "LLMServiceError",
    "LLMTransportError",
    "SCRIPT_ENTRIES_CONTRACT",
    "SENTIMENT_MOOD_CONTRACT",
    "StructuredLLMResult",
    "StructuredLLMService",
    "StructuredOutputContract",
    "ToolCapabilityResult",
    "ToolCapabilityService",
    "ToolStreamResult",
    "ToolStreamingService",
    "VOICE_DESCRIPTION_CONTRACT",
]
