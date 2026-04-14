#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from runtime_layout import LAYOUT
from script_provider import open_project_script_store


PERSISTENT_FILE_MAP = {
    "state.json": LAYOUT.state_path,
    "audio_queue_state.json": LAYOUT.audio_queue_state_path,
    "audio_cancel_tombstone.json": LAYOUT.audio_cancel_tombstone_path,
    "processing_workflow_state.json": LAYOUT.processing_workflow_state_path,
    "new_mode_workflow_state.json": LAYOUT.new_mode_workflow_state_path,
    "script_generation_checkpoint.json": LAYOUT.script_generation_checkpoint_path,
    "script_review_checkpoint.json": LAYOUT.script_review_checkpoint_path,
    "chunks.sqlite3": LAYOUT.chunks_db_path,
    "chunks.queue.log": LAYOUT.chunks_queue_log_path,
    "voice_state.audit.jsonl": LAYOUT.voice_audit_log_path,
    "script_repair_trace.jsonl": LAYOUT.script_repair_trace_path,
    "cloned_audiobook.mp3": LAYOUT.audiobook_path,
    "optimized_audiobook.zip": LAYOUT.optimized_export_path,
    "audacity_export.zip": LAYOUT.audacity_export_path,
    "audiobook.m4b": LAYOUT.m4b_path,
    "m4b_cover.jpg": LAYOUT.m4b_cover_path,
}

PERSISTENT_DIR_MAP = {
    "uploads": LAYOUT.uploads_dir,
    "voicelines": LAYOUT.voicelines_dir,
    "clone_voices": LAYOUT.clone_voices_dir,
    "designed_voices": LAYOUT.designed_voices_dir,
    "scripts": LAYOUT.script_snapshots_dir,
    "saved_projects": LAYOUT.project_archives_dir,
    "backups": LAYOUT.backups_dir,
    "dataset_builder": LAYOUT.dataset_builder_dir,
    "lora_datasets": LAYOUT.lora_datasets_dir,
    "lora_models": LAYOUT.lora_models_dir,
}

PROMPT_FILE_MAP = {
    "default_prompts.txt": LAYOUT.prompt_default_path,
    "review_prompts.txt": LAYOUT.prompt_review_path,
    "attribution_prompts.txt": LAYOUT.prompt_attribution_path,
    "voice_prompt.txt": LAYOUT.prompt_voice_path,
    "dialogue_identification_system_prompt.txt": LAYOUT.prompt_dialogue_path,
    "temperament_extraction_system_prompt.txt": LAYOUT.prompt_temperament_path,
}

PROJECT_DOCUMENT_MAP = {
    "paragraphs.json": "paragraphs",
    "script_sanity_check.json": "script_sanity_result",
}


def _copy_file(source: str, target: str, *, overwrite: bool) -> bool:
    if not os.path.isfile(source):
        return False
    if os.path.exists(target) and not overwrite:
        return False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)
    return True


def _copy_dir(source: str, target: str, *, overwrite: bool) -> bool:
    if not os.path.isdir(source):
        return False
    if os.path.exists(target) and overwrite:
        shutil.rmtree(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=not overwrite)
    return True


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_script_document(store, legacy_root: str) -> bool:
    path = os.path.join(legacy_root, "annotated_script.json")
    if not os.path.isfile(path):
        return False
    payload = _load_json(path)
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    dictionary = payload.get("dictionary", []) if isinstance(payload, dict) else []
    sanity_cache = payload.get("sanity_cache") if isinstance(payload, dict) else None
    store.replace_script_document(
        entries=entries or [],
        dictionary=dictionary or [],
        sanity_cache=sanity_cache,
        reason="legacy_runtime_migration",
        rebuild_chunks=True,
        wait=True,
    )
    return True


def _import_chunks(store, legacy_root: str) -> bool:
    path = os.path.join(legacy_root, "chunks.json")
    if not os.path.isfile(path):
        return False
    payload = _load_json(path)
    if not isinstance(payload, list):
        return False
    store.replace_chunks(payload, reason="legacy_runtime_migration", wait=True)
    return True


def _import_voice_config(store, legacy_root: str) -> bool:
    path = os.path.join(legacy_root, "voice_config.json")
    if not os.path.isfile(path):
        return False
    payload = _load_json(path)
    rows = [
        {"speaker": speaker, "config": dict(config or {})}
        for speaker, config in dict(payload or {}).items()
    ]
    store.replace_voice_profiles(rows, reason="legacy_runtime_migration", wait=True)
    return True


def _import_project_documents(store, legacy_root: str) -> list[str]:
    imported = []
    for filename, document_key in PROJECT_DOCUMENT_MAP.items():
        path = os.path.join(legacy_root, filename)
        if not os.path.isfile(path):
            continue
        payload = _load_json(path)
        store.replace_project_document(
            document_key,
            payload,
            reason="legacy_runtime_migration",
            wait=True,
        )
        imported.append(filename)
    return imported


def _import_transcription_cache(store, legacy_root: str) -> bool:
    path = os.path.join(legacy_root, "transcription_cache.json")
    if not os.path.isfile(path):
        return False
    payload = _load_json(path)
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return False
    store.clear_transcription_cache(reason="legacy_runtime_migration", wait=True)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        store.store_transcription_cache(entry, reason="legacy_runtime_migration", wait=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy repo-root project state into the DB-only runtime layout."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing migrated files/directories.")
    parser.add_argument(
        "--legacy-root",
        default=LAYOUT.legacy_root_dir,
        help="Directory containing legacy repo-root project files.",
    )
    args = parser.parse_args()

    legacy_root = os.path.abspath(args.legacy_root)
    LAYOUT.ensure_base_dirs()
    migrated = []

    for legacy_name, target in PROMPT_FILE_MAP.items():
        source = os.path.join(legacy_root, legacy_name)
        if _copy_file(source, target, overwrite=args.overwrite):
            migrated.append((legacy_name, os.path.relpath(target, legacy_root)))

    for legacy_name, target in PERSISTENT_FILE_MAP.items():
        source = os.path.join(legacy_root, legacy_name)
        if _copy_file(source, target, overwrite=args.overwrite):
            migrated.append((legacy_name, os.path.relpath(target, legacy_root)))

    for legacy_name, target in PERSISTENT_DIR_MAP.items():
        source = os.path.join(legacy_root, legacy_name)
        if _copy_dir(source, target, overwrite=args.overwrite):
            migrated.append((legacy_name, os.path.relpath(target, legacy_root)))

    store = open_project_script_store(LAYOUT.project_dir)
    try:
        if _import_script_document(store, legacy_root):
            migrated.append(("annotated_script.json", "db/script_entries"))
        if _import_chunks(store, legacy_root):
            migrated.append(("chunks.json", "db/chunks"))
        if _import_voice_config(store, legacy_root):
            migrated.append(("voice_config.json", "db/voice_profiles"))
        if _import_transcription_cache(store, legacy_root):
            migrated.append(("transcription_cache.json", "db/transcription_cache_entries"))
        for legacy_name in _import_project_documents(store, legacy_root):
            migrated.append((legacy_name, f"db/project_documents/{PROJECT_DOCUMENT_MAP[legacy_name]}"))
    finally:
        store.stop(flush=True)

    print("Migrated entries:", flush=True)
    if not migrated:
        print("  none", flush=True)
        return 0

    for source_name, target_name in migrated:
        print(f"  {source_name} -> {target_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
