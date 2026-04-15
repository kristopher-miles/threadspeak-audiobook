#!/usr/bin/env python3
"""Build full E2E harness fixtures by composing phase fixtures."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(APP_DIR)

SCRIPT_LM_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_generate_script_test_book.json")
VOICE_LM_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "lmstudio_voice_profiles_test_book.json")
VOICE_QWEN_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_voice_profiles_test_book.json")
EDITOR_QWEN_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_editor_audio_test_book.json")
VOICE_MANIFEST_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "voice_profiles_test_book_manifest.json")
OUTPUT_QWEN_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "qwen_local_full_e2e_test_book.json")
OUTPUT_HARNESS_DEFAULT = os.path.join(APP_DIR, "test_fixtures", "e2e_sim", "full_e2e_test_book_harness.json")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _methods(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    methods = payload.get("methods") or {}
    if not isinstance(methods, dict):
        raise ValueError("Qwen fixture methods must be an object")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, value in methods.items():
        if not isinstance(value, list):
            raise ValueError(f"Qwen fixture methods[{key}] must be a list")
        out[str(key)] = [dict(item or {}) for item in value]
    return out


def _unique_speakers_in_order(editor_methods: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for entry in editor_methods.get("generate_voice_clone") or []:
        speaker = str(((entry.get("metadata") or {}).get("speaker") or "").strip())
        if not speaker:
            continue
        if speaker in seen:
            continue
        seen.add(speaker)
        ordered.append(speaker)
    return ordered


def _build_prompt_entries(
    *,
    voice_manifest: Dict[str, Any],
    speaker_order: Sequence[str],
) -> List[Dict[str, Any]]:
    profiles = [dict(item or {}) for item in (voice_manifest.get("voice_profiles") or [])]
    profile_by_speaker = {
        str(item.get("speaker") or "").strip(): item
        for item in profiles
        if str(item.get("speaker") or "").strip()
    }

    prompt_entries: List[Dict[str, Any]] = []
    for speaker in speaker_order:
        profile = profile_by_speaker.get(speaker) or {}
        ref_text = str(profile.get("generated_ref_text") or profile.get("sample_text") or "").strip()
        if not ref_text:
            raise ValueError(f"Voice manifest missing generated_ref_text/sample_text for speaker '{speaker}'")
        prompt_entries.append(
            {
                "expect": {
                    "has_ref_audio": True,
                    "ref_text": ref_text,
                },
                "prompt_ref_text": ref_text,
                "prompt_tokens": max(1, len(ref_text) // 4),
                "metadata": {
                    "speaker": speaker,
                    "ref_audio": str(profile.get("ref_audio") or "").strip(),
                },
            }
        )
    return prompt_entries


def build_full_harness(
    *,
    script_lm_fixture: str,
    voice_lm_fixture: str,
    voice_qwen_fixture: str,
    editor_qwen_fixture: str,
    voice_manifest_path: str,
    output_qwen_fixture: str,
    output_harness_manifest: str,
) -> Dict[str, Any]:
    script_lm_abs = os.path.abspath(script_lm_fixture)
    voice_lm_abs = os.path.abspath(voice_lm_fixture)
    voice_qwen_abs = os.path.abspath(voice_qwen_fixture)
    editor_qwen_abs = os.path.abspath(editor_qwen_fixture)
    voice_manifest_abs = os.path.abspath(voice_manifest_path)
    output_qwen_abs = os.path.abspath(output_qwen_fixture)
    output_harness_abs = os.path.abspath(output_harness_manifest)

    for path in (script_lm_abs, voice_lm_abs, voice_qwen_abs, editor_qwen_abs, voice_manifest_abs):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    script_lm = _read_json(script_lm_abs)
    voice_lm = _read_json(voice_lm_abs)
    voice_qwen = _read_json(voice_qwen_abs)
    editor_qwen = _read_json(editor_qwen_abs)
    voice_manifest = _read_json(voice_manifest_abs)

    voice_methods = _methods(voice_qwen)
    editor_methods = _methods(editor_qwen)
    _ = voice_manifest  # Manifest remains a provenance artifact for the combined harness.

    combined_methods: Dict[str, List[Dict[str, Any]]] = {
        "generate_voice_design": list(voice_methods.get("generate_voice_design") or []),
        # Clone-prompt generation can vary by runtime path; keep this dynamic in simulator.
        "create_voice_clone_prompt": [],
        "generate_voice_clone": list(editor_methods.get("generate_voice_clone") or []),
    }

    source_file = (
        str(((script_lm.get("metadata") or {}).get("source_file") or "").strip())
        or str(((voice_lm.get("metadata") or {}).get("source_file") or "").strip())
    )

    combined_qwen = {
        "strict": True,
        "unordered_methods": ["generate_voice_clone"],
        "metadata": {
            "purpose": "Combined Qwen fixture for full e2e replay (voice design + editor audio)",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file,
            "component_fixtures": {
                "voice_qwen": os.path.relpath(voice_qwen_abs, REPO_ROOT),
                "editor_qwen": os.path.relpath(editor_qwen_abs, REPO_ROOT),
                "voice_manifest": os.path.relpath(voice_manifest_abs, REPO_ROOT),
            },
            "counts": {
                "generate_voice_design": len(combined_methods["generate_voice_design"]),
                "create_voice_clone_prompt": len(combined_methods["create_voice_clone_prompt"]),
                "generate_voice_clone": len(combined_methods["generate_voice_clone"]),
            },
        },
        "methods": combined_methods,
    }

    os.makedirs(os.path.dirname(output_qwen_abs), exist_ok=True)
    with open(output_qwen_abs, "w", encoding="utf-8") as handle:
        json.dump(combined_qwen, handle, ensure_ascii=False, indent=2)

    harness_manifest = {
        "version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "phases": {
            "generate_script": {
                "lm_fixture": os.path.relpath(script_lm_abs, REPO_ROOT),
                "model_name": str(((script_lm.get("metadata") or {}).get("model_name") or "").strip()),
            },
            "voice_profiles": {
                "lm_fixture": os.path.relpath(voice_lm_abs, REPO_ROOT),
                "model_name": str(((voice_lm.get("metadata") or {}).get("model_name") or "").strip()),
            },
            "editor_audio": {
                "qwen_fixture": os.path.relpath(output_qwen_abs, REPO_ROOT),
            },
        },
        "component_fixtures": {
            "script_lm": os.path.relpath(script_lm_abs, REPO_ROOT),
            "voice_lm": os.path.relpath(voice_lm_abs, REPO_ROOT),
            "voice_qwen": os.path.relpath(voice_qwen_abs, REPO_ROOT),
            "editor_qwen": os.path.relpath(editor_qwen_abs, REPO_ROOT),
            "voice_manifest": os.path.relpath(voice_manifest_abs, REPO_ROOT),
        },
    }

    os.makedirs(os.path.dirname(output_harness_abs), exist_ok=True)
    with open(output_harness_abs, "w", encoding="utf-8") as handle:
        json.dump(harness_manifest, handle, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "qwen_fixture": output_qwen_abs,
        "harness_manifest": output_harness_abs,
        "counts": combined_qwen["metadata"]["counts"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full E2E harness fixtures from phase fixtures")
    parser.add_argument("--script-lm", default=SCRIPT_LM_DEFAULT)
    parser.add_argument("--voice-lm", default=VOICE_LM_DEFAULT)
    parser.add_argument("--voice-qwen", default=VOICE_QWEN_DEFAULT)
    parser.add_argument("--editor-qwen", default=EDITOR_QWEN_DEFAULT)
    parser.add_argument("--voice-manifest", default=VOICE_MANIFEST_DEFAULT)
    parser.add_argument("--output-qwen", default=OUTPUT_QWEN_DEFAULT)
    parser.add_argument("--output-harness", default=OUTPUT_HARNESS_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_full_harness(
        script_lm_fixture=args.script_lm,
        voice_lm_fixture=args.voice_lm,
        voice_qwen_fixture=args.voice_qwen,
        editor_qwen_fixture=args.editor_qwen,
        voice_manifest_path=args.voice_manifest,
        output_qwen_fixture=args.output_qwen,
        output_harness_manifest=args.output_harness,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
