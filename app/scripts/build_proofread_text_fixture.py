#!/usr/bin/env python3
"""Build a proofread transcript fixture from captured non-legacy E2E artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(APP_DIR)

QWEN_FULL_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_full_e2e_test_book.json")
EDITOR_MANIFEST_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "editor_audio_test_book_manifest.json")
OUTPUT_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "proofread_text_test_book.json")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _normalize_audio_path(path: Any) -> str:
    value = str(path or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _manifest_line_by_uid(manifest_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lines = manifest_payload.get("lines") or []
    if not isinstance(lines, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in lines:
        row = dict(item or {})
        uid = str(row.get("uid") or "").strip()
        if uid and uid not in out:
            out[uid] = row
    return out


def _coalesce_transcript(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_proofread_fixture(
    *,
    qwen_full_fixture: str,
    editor_manifest: str,
    output_path: str,
    strict: bool = True,
    fallback_mode: str = "chunk_text",
) -> Dict[str, Any]:
    qwen_abs = os.path.abspath(qwen_full_fixture)
    manifest_abs = os.path.abspath(editor_manifest)
    output_abs = os.path.abspath(output_path)
    for path in (qwen_abs, manifest_abs):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    qwen_payload = _read_json(qwen_abs)
    manifest_payload = _read_json(manifest_abs)
    line_by_uid = _manifest_line_by_uid(manifest_payload)

    methods = qwen_payload.get("methods") or {}
    if not isinstance(methods, dict):
        raise ValueError("Qwen full fixture must include a methods object")
    clone_entries = methods.get("generate_voice_clone") or []
    if not isinstance(clone_entries, list):
        raise ValueError("Qwen full fixture methods.generate_voice_clone must be a list")

    entries = []
    seen_audio_paths: Dict[str, str] = {}
    for idx, raw_entry in enumerate(clone_entries):
        entry = dict(raw_entry or {})
        metadata = dict(entry.get("metadata") or {})
        expect = dict(entry.get("expect") or {})

        audio_path = _normalize_audio_path(metadata.get("source_audio_path"))
        if not audio_path:
            raise ValueError(
                f"Entry {idx} missing metadata.source_audio_path in {qwen_abs}"
            )

        uid = str(metadata.get("uid") or "").strip()
        manifest_line = line_by_uid.get(uid) if uid else None
        transcript_text = _coalesce_transcript(
            (manifest_line or {}).get("transformed_text"),
            (manifest_line or {}).get("text"),
            expect.get("text"),
            metadata.get("text"),
        )
        if not transcript_text:
            raise ValueError(
                f"Unable to derive transcript text for entry {idx} ({audio_path})"
            )

        existing = seen_audio_paths.get(audio_path)
        if existing is not None:
            if existing != transcript_text:
                raise ValueError(
                    f"Conflicting transcript text for audio_path '{audio_path}': "
                    f"{existing!r} vs {transcript_text!r}"
                )
            continue

        seen_audio_paths[audio_path] = transcript_text
        entries.append(
            {
                "audio_path": audio_path,
                "transcript_text": transcript_text,
                "uid": uid,
                "speaker": str(metadata.get("speaker") or "").strip(),
                "line_id": metadata.get("line_id"),
            }
        )

    fixture_payload = {
        "strict": bool(strict),
        "fallback_mode": str(fallback_mode or "chunk_text").strip().lower() or "chunk_text",
        "metadata": {
            "purpose": "Proofread transcript fixture for non-legacy E2E replay",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_qwen_fixture": os.path.relpath(qwen_abs, REPO_ROOT),
            "source_editor_manifest": os.path.relpath(manifest_abs, REPO_ROOT),
            "entry_count": len(entries),
        },
        "entries": entries,
    }

    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    with open(output_abs, "w", encoding="utf-8") as handle:
        json.dump(fixture_payload, handle, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "output_path": output_abs,
        "entry_count": len(entries),
        "strict": bool(strict),
        "fallback_mode": fixture_payload["fallback_mode"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build proofread transcript fixture for E2E test mode")
    parser.add_argument("--qwen-full", default=QWEN_FULL_DEFAULT)
    parser.add_argument("--editor-manifest", default=EDITOR_MANIFEST_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--fallback-mode", default="chunk_text", choices=["chunk_text", "fail"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_proofread_fixture(
        qwen_full_fixture=args.qwen_full,
        editor_manifest=args.editor_manifest,
        output_path=args.output,
        strict=bool(args.strict),
        fallback_mode=str(args.fallback_mode),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

