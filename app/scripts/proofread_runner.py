import json
import sys

from stdio_utils import configure_utf8_stdio
from project import ProjectManager

configure_utf8_stdio()

TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__:"


def emit_progress(payload):
    print(f"{TASK_PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)


def main():
    if len(sys.argv) < 4:
        print("Usage: proofread_runner.py <root_dir> <threshold> <chapter|__ALL__>", flush=True)
        return 2

    root_dir = sys.argv[1]
    threshold = float(sys.argv[2])
    chapter_arg = sys.argv[3]
    chapter = None if chapter_arg == "__ALL__" else chapter_arg

    manager = ProjectManager(root_dir)
    result = manager.proofread_chunks(
        chapter=chapter,
        threshold=threshold,
        progress_callback=emit_progress,
    )
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
