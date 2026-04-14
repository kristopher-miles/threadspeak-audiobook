"""LM Studio model-load helpers for forcing runtime load configuration."""

from typing import Any, Callable, Dict, Optional

import requests

from .tool_capability_service import ToolCapabilityService


class LMStudioModelLoadService:
    """Load an LM Studio model with explicit runtime configuration."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 120,
        post_json_fn: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None,
    ):
        self._timeout_seconds = timeout_seconds
        self._post_json_fn = post_json_fn or self._post_json

    def load_model(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        context_length: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        flash_attention: Optional[bool] = None,
        num_experts: Optional[int] = None,
        offload_kv_cache_to_gpu: Optional[bool] = None,
        echo_load_config: bool = False,
    ) -> Dict[str, Any]:
        model = str(model_name or "").strip()
        if not model:
            raise ValueError("model_name is required")

        origin = ToolCapabilityService.normalize_lm_studio_origin(base_url)
        if not origin:
            raise ValueError("base_url is required")

        payload: Dict[str, Any] = {"model": model}
        optional_fields = {
            "context_length": context_length,
            "eval_batch_size": eval_batch_size,
            "flash_attention": flash_attention,
            "num_experts": num_experts,
            "offload_kv_cache_to_gpu": offload_kv_cache_to_gpu,
            "echo_load_config": True if echo_load_config else None,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value

        return self._post_json_fn(f"{origin}/api/v1/models/load", payload, api_key)

    @staticmethod
    def auth_headers(api_key: str) -> Dict[str, str]:
        key = str(api_key or "").strip()
        if not key or key.lower() == "local":
            return {}
        return {"Authorization": f"Bearer {key}"}

    def _post_json(self, url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        response = requests.post(
            url,
            headers=self.auth_headers(api_key),
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
