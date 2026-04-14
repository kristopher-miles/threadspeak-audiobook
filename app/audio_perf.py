import json
import os
import threading
import time

from runtime_layout import LAYOUT


_AUDIO_PERF_ENV = "THREADSPEAK_AUDIO_PERF"
_AUDIO_PERF_LOCK = threading.Lock()
_ENABLED_VALUES = {"1", "true", "yes", "on"}


def audio_perf_enabled():
    value = str(os.getenv(_AUDIO_PERF_ENV, "")).strip().lower()
    return value in _ENABLED_VALUES


def audio_perf_log_path():
    LAYOUT.ensure_base_dirs()
    return os.path.join(LAYOUT.logs_dir, "audio_perf.jsonl")


def record_audio_perf(event, **payload):
    if not audio_perf_enabled():
        return

    record = {
        "ts": time.time(),
        "event": str(event or "").strip() or "unknown",
        "thread": threading.current_thread().name,
    }
    for key, value in payload.items():
        try:
            json.dumps(value)
            record[key] = value
        except TypeError:
            record[key] = repr(value)

    line = json.dumps(record, sort_keys=True)
    path = audio_perf_log_path()
    with _AUDIO_PERF_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
