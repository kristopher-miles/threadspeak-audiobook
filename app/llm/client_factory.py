"""Factory for OpenAI-compatible API clients."""

from typing import Any, Optional, Type

from openai import OpenAI

from .models import LLMRuntimeConfig


class LLMClientFactory:
    """Create configured OpenAI-compatible clients from normalized runtime settings."""

    def __init__(self, client_cls: Optional[Type[Any]] = None):
        self._client_cls = client_cls or OpenAI

    def create_client(self, runtime: LLMRuntimeConfig):
        kwargs = {
            "base_url": runtime.base_url,
            "api_key": runtime.api_key,
        }
        if runtime.timeout is not None:
            kwargs["timeout"] = runtime.timeout
        return self._client_cls(**kwargs)

    def create_from_app_config(self, config: dict, **runtime_kwargs: Any):
        runtime = LLMRuntimeConfig.from_app_config(config, **runtime_kwargs)
        return self.create_client(runtime), runtime
