"""Test-only E2E simulation helpers.

These modules provide deterministic HTTP and local-backend simulators used by
integration tests. They are activated explicitly via env vars and are not used
by default runtime paths.
"""

from .fixture_queue import ScriptedInteractionQueue, load_fixture_payload
from .lmstudio_server import LMStudioSimServer
from .qwen_local_sim import QwenLocalSimProvider

__all__ = [
    "ScriptedInteractionQueue",
    "load_fixture_payload",
    "LMStudioSimServer",
    "QwenLocalSimProvider",
]
