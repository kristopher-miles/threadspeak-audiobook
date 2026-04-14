import json
import os

from script_provider import open_project_script_store


def infer_project_root(*paths: str | None) -> str | None:
    for path in paths:
        normalized = str(path or "").strip()
        if not normalized:
            continue
        return os.path.abspath(os.path.dirname(normalized) or ".")
    return None


def import_project_document_from_path(project_root: str, key: str, path: str, *, reason: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    store = open_project_script_store(project_root)
    try:
        store.replace_project_document(key, payload, reason=reason, wait=True)
    finally:
        store.stop()
    return payload
