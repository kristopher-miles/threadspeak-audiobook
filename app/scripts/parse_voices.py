import json
import os

from stdio_utils import configure_utf8_stdio
from runtime_layout import LAYOUT
from script_provider import open_project_script_store

configure_utf8_stdio()


def main():
    store = open_project_script_store(LAYOUT.project_dir)
    try:
        script_document = store.load_script_document()
        if not (script_document.get("entries") or []):
            print("Error: no script entries found. Please generate the script first.")
            return 1

        summary = store.get_voice_summary()
        voices = list(summary.get("voices") or [])
        existing = store.load_voice_config()
        missing_rows = [
            {"speaker": speaker, "config": {}}
            for speaker in voices
            if str(speaker or "").strip() and speaker not in existing
        ]
        if missing_rows:
            store.upsert_voice_profiles(missing_rows, reason="parse_voices_seed_profiles", wait=True)
        print(f"Found {len(voices)} unique voices: {', '.join(voices)}")
        print("Voice state is stored in SQLite; no project voice file was written.")
    finally:
        store.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
