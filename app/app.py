import functools
import os
import sys
from stdio_utils import configure_utf8_stdio

configure_utf8_stdio()

import types
import api.shared as _shared
import api.routers.config_router as _config_router
import api.routers.workflow_router as _workflow_router
import api.routers.voices_router as _voices_router
import api.routers.editor_audio_router as _editor_audio_router
import api.routers.scripts_router as _scripts_router
import api.routers.voice_designer_router as _voice_designer_router
import api.routers.clone_voices_router as _clone_voices_router
import api.routers.lora_router as _lora_router
import api.routers.dataset_builder_router as _dataset_builder_router
import api.routers.emotions_router as _emotions_router
import api.routers.model_downloads_router as _model_downloads_router
from api.main import app

_MODULES = (
    _shared,
    _config_router,
    _workflow_router,
    _voices_router,
    _editor_audio_router,
    _scripts_router,
    _voice_designer_router,
    _clone_voices_router,
    _lora_router,
    _dataset_builder_router,
    _emotions_router,
    _model_downloads_router,
)

_EXPORT_OWNERS = {}


def _sync_compat_overrides():
    """Propagate app.py monkeypatches to decomposed modules before each call."""
    module_globals = globals()
    for _name, _value in module_globals.items():
        if _name.startswith("__"):
            continue
        if getattr(_value, "__compat_wrapper__", False):
            _value = getattr(_value, "__compat_target__", _value)
        for _module in _MODULES:
            if hasattr(_module, _name):
                setattr(_module, _name, _value)


def _wrap_callable(name, owner_module, fn):
    @functools.wraps(fn)
    def _compat_wrapper(*args, **kwargs):
        _sync_compat_overrides()
        result = getattr(owner_module, name)(*args, **kwargs)
        for _export_name, _export_owner in _EXPORT_OWNERS.items():
            _current = globals().get(_export_name)
            if getattr(_current, "__compat_wrapper__", False):
                continue
            try:
                globals()[_export_name] = getattr(_export_owner, _export_name)
            except AttributeError:
                pass
        return result

    _compat_wrapper.__compat_wrapper__ = True
    _compat_wrapper.__compat_target__ = fn
    return _compat_wrapper


for _module in _MODULES:
    for _name, _value in vars(_module).items():
        if _name.startswith("__"):
            continue
        if _name not in _EXPORT_OWNERS:
            _EXPORT_OWNERS[_name] = _module
        globals()[_name] = _value

for _name, _owner in list(_EXPORT_OWNERS.items()):
    _value = globals().get(_name)
    if isinstance(_value, types.FunctionType):
        globals()[_name] = _wrap_callable(_name, _owner, _value)

# Preserve normal import behavior where this module exists in sys.modules.
_this_module = sys.modules.get(__name__)
if _this_module is not None and isinstance(_this_module, types.ModuleType):
    _this_module.__dict__.update(globals())


class _SuppressOKAccessFilter(logging.Filter):
    """Drop 2xx/3xx access-log lines; keep 4xx/5xx and everything else."""
    def filter(self, record):
        try:
            # uvicorn access records have args = (client, method, path, http_ver, status_code)
            return int(record.args[4]) >= 400
        except (TypeError, IndexError, ValueError):
            return True


if __name__ == "__main__":
    import logging
    import uvicorn

    log_level_env = (os.getenv("LOG_LEVEL") or "default").strip().lower()
    if log_level_env != "full":
        logging.getLogger("uvicorn.access").addFilter(_SuppressOKAccessFilter())

    share_local_raw = (os.getenv("PINOKIO_SHARE_LOCAL") or "").strip().lower()
    share_local_enabled = share_local_raw in {"1", "true", "yes", "on"}

    host = "0.0.0.0" if share_local_enabled else "127.0.0.1"
    port = 4200
    pinokio_port = (os.getenv("PINOKIO_SHARE_LOCAL_PORT") or "").strip()
    if pinokio_port:
        try:
            candidate_port = int(pinokio_port)
            if 1 <= candidate_port <= 65535:
                port = candidate_port
        except ValueError:
            pass

    print(
        "[Threadspeak] startup bind settings: "
        f"PINOKIO_SHARE_LOCAL={os.getenv('PINOKIO_SHARE_LOCAL')!r}, "
        f"PINOKIO_SHARE_LOCAL_PORT={os.getenv('PINOKIO_SHARE_LOCAL_PORT')!r}, "
        f"host={host}, port={port}",
        flush=True,
    )

    uvicorn.run(app, host=host, port=port)
