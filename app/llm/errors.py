"""Shared exception types for LLM API integration."""


class LLMServiceError(Exception):
    """Base class for recoverable LLM service errors."""


class LLMTransportError(LLMServiceError):
    """Raised when an outbound request to the LLM provider fails."""


class LLMResponseParseError(LLMServiceError):
    """Raised when provider responses cannot be parsed into expected structures."""
