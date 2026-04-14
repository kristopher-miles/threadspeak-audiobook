"""Capability verification for tool-calling support across supported providers."""

from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from .models import ToolCapabilityResult


class ToolCapabilityService:
    """Verify whether configured models support tool-calling semantics."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 5,
        request_json_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ):
        self._timeout_seconds = timeout_seconds
        self._request_json_fn = request_json_fn or self._request_json

    def verify_tool_capability(self, base_url: str, api_key: str, model_name: str) -> ToolCapabilityResult:
        if self.is_openrouter_url(base_url):
            return self.verify_openrouter_tool_capability(base_url, api_key, model_name)
        return self.verify_lm_studio_tool_capability(base_url, api_key, model_name)

    def verify_openrouter_tool_capability(self, base_url: str, api_key: str, model_name: str) -> ToolCapabilityResult:
        model = (model_name or "").strip()
        if not model:
            return self._result("unknown", "openrouter", "Enter a model name to verify tool calling.")

        try:
            payload = self._request_json_fn("https://openrouter.ai/api/v1/models?supported_parameters=tools", api_key)
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                return self._result("unknown", "openrouter", "OpenRouter returned an unexpected model list.")

            tool_models = {
                str(item.get("id") or "").strip()
                for item in models
                if isinstance(item, dict)
            }
            if model in tool_models:
                return self._result("supported", "openrouter", "This OpenRouter model supports tool calling.")
            return self._result(
                "unsupported",
                "openrouter",
                "OpenRouter does not list this model as supporting tool calling.",
            )
        except Exception as exc:
            return self._result("unknown", "openrouter", f"Could not verify OpenRouter model: {exc}")

    def verify_lm_studio_tool_capability(self, base_url: str, api_key: str, model_name: str) -> ToolCapabilityResult:
        model = (model_name or "").strip()
        if not model:
            return self._result("unknown", "lmstudio", "Enter a model name to verify tool calling.")

        origin = self.normalize_lm_studio_origin(base_url)
        if not origin:
            return self._result("unknown", "lmstudio", "Enter an LM Studio base URL to verify tool calling.")

        try:
            payload = self._request_json_fn(f"{origin}/api/v1/models", api_key)
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                return self._result("unknown", "lmstudio", "LM Studio returned an unexpected model list.")

            for item in models:
                if not isinstance(item, dict) or not self.model_name_matches_lm_studio(item, model):
                    continue
                capabilities = item.get("capabilities") or {}
                if capabilities.get("trained_for_tool_use") is True:
                    return self._result("supported", "lmstudio", "This LM Studio model is trained for tool use.")
                if capabilities.get("trained_for_tool_use") is False:
                    return self._result(
                        "unsupported",
                        "lmstudio",
                        "LM Studio reports this model is not trained for tool use.",
                    )
                return self._result(
                    "unknown",
                    "lmstudio",
                    "LM Studio did not report tool-use capability for this model.",
                )
            return self._result("unknown", "lmstudio", "LM Studio did not return a matching model.")
        except Exception as exc:
            return self._result("unknown", "lmstudio", f"Could not verify LM Studio model: {exc}")

    @staticmethod
    def auth_headers(api_key: str) -> Dict[str, str]:
        key = (api_key or "").strip()
        if not key or key.lower() == "local":
            return {}
        return {"Authorization": f"Bearer {key}"}

    def _request_json(self, url: str, api_key: str) -> Dict[str, Any]:
        response = requests.get(url, headers=self.auth_headers(api_key), timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def is_openrouter_url(base_url: str) -> bool:
        try:
            return "openrouter.ai" in (urlparse(base_url).netloc or "").lower()
        except ValueError:
            return False

    @staticmethod
    def normalize_lm_studio_origin(base_url: str) -> str:
        raw = (base_url or "").strip().rstrip("/")
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def model_name_matches_lm_studio(model: Dict[str, Any], model_name: str) -> bool:
        wanted = (model_name or "").strip()
        if not wanted:
            return False

        candidates = {
            str(model.get("key") or "").strip(),
            str(model.get("display_name") or "").strip(),
        }
        for instance in model.get("loaded_instances") or []:
            if isinstance(instance, dict):
                candidates.add(str(instance.get("id") or "").strip())
        return wanted in candidates

    @staticmethod
    def _result(status: str, provider: str, message: str) -> ToolCapabilityResult:
        return ToolCapabilityResult(status=status, provider=provider, message=message)
