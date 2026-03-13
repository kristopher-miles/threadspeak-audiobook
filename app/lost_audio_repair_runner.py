import json
import os
import sys

from project import ProjectManager


def main():
    if len(sys.argv) < 3:
        print("Usage: lost_audio_repair_runner.py <root_dir> <use_asr>", flush=True)
        return 2

    root_dir = sys.argv[1]
    use_asr = sys.argv[2] == "1"
    manager = ProjectManager(root_dir)

    def progress_callback(update):
        message = str((update or {}).get("message") or "").strip()
        if message:
            print(message, flush=True)

    result = manager.repair_lost_audio_links(use_asr=use_asr, progress_callback=progress_callback)
    print(
        json.dumps(
            {
                "status": "ok",
                "preserved": result.get("preserved", 0),
                "relinked": result.get("relinked", 0),
                "asr_relinked": result.get("asr_relinked", 0),
                "invalid_candidates": result.get("invalid_candidates", 0),
                "unmatched_files": result.get("unmatched_files", 0),
                "total_candidates": result.get("total_candidates", 0),
                "asr_errors": result.get("asr_errors", []),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
