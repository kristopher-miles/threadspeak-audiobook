#!/usr/bin/env python3
"""Create and manage an isolated Threadspeak debug clone.

This harness exists for fast, repeatable UI debugging against real project data.
It can:

1. clone the current repository into a temporary workspace,
2. start an isolated app server on a free port,
3. print the URL/paths needed for browser automation, and
4. tear everything down and delete the temp clone.

Typical usage:

    app/env/bin/python app/scripts/isolated_debug_app.py start
    app/env/bin/python app/scripts/isolated_debug_app.py stop --manifest /tmp/.../isolated-debug.json

For commands that should always clean up when they finish, use `exec`:

    app/env/bin/python app/scripts/isolated_debug_app.py exec -- python -c "print('debug')"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_PREFIX = "threadspeak-debug-"
DEFAULT_MANIFEST_NAME = "isolated-debug.json"
DEFAULT_READY_TIMEOUT_SECONDS = 30.0
DEFAULT_READY_POLL_SECONDS = 0.2
COPY_EXCLUDES = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
    "env",
    "app/env",
    "node_modules",
    "cache",
    "logs",
    "app/.pytest_cache",
    "app/__pycache__",
    ".audio_queue_state.json.*.tmp",
    ".chunks.json.*.tmp",
    ".new_mode_workflow_state.json.*.tmp",
    "merge_audio_*",
    "temp_batch_*.wav",
)


class HarnessError(RuntimeError):
    """Raised when the isolated harness cannot complete an operation."""


def _ownership_path(source_root: Path) -> Path:
    digest = hashlib.sha1(str(source_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"threadspeak-isolated-debug-{digest}.owner.json"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _release_harness_ownership(source_root: Path, *, manifest_path: str | None = None) -> None:
    owner_path = _ownership_path(source_root)
    payload = _read_json_file(owner_path)
    if payload is None:
        owner_path.unlink(missing_ok=True)
        return
    if manifest_path is not None and str(payload.get("manifest_path") or "").strip() != str(manifest_path):
        return
    owner_path.unlink(missing_ok=True)


def _claim_harness_ownership(source_root: Path) -> Path:
    owner_path = _ownership_path(source_root)
    stale_payload = _read_json_file(owner_path)
    if stale_payload is not None:
        stale_pid = int(stale_payload.get("pid") or 0)
        stale_manifest = str(stale_payload.get("manifest_path") or "").strip()
        stale_port = stale_payload.get("port")
        if _pid_is_running(stale_pid):
            stop_hint = (
                f"app/env/bin/python app/scripts/isolated_debug_app.py stop --manifest {stale_manifest}"
                if stale_manifest else
                f"kill {stale_pid}"
            )
            raise HarnessError(
                "another isolated debug server is already running"
                f" (pid={stale_pid}, port={stale_port or 'unknown'}).\n"
                f"Stop the previous server before spawning a new one.\n"
                f"Recommended cleanup: {stop_hint}"
            )
        owner_path.unlink(missing_ok=True)

    reservation = {
        "owner_pid": os.getpid(),
        "source_root": str(source_root.resolve()),
        "claimed_at": time.time(),
    }
    try:
        fd = os.open(owner_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return _claim_harness_ownership(source_root)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(reservation, handle, indent=2, sort_keys=True)
    return owner_path


def _update_harness_ownership(owner_path: Path, payload: dict) -> None:
    owner_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _python_for_clone(source_root: Path) -> Path:
    preferred = source_root / "app" / "env" / "bin" / "python"
    if preferred.exists():
        return preferred
    return Path(sys.executable)


def _ignore_copy(_dir: str, names: Iterable[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in {".git", ".pytest_cache", "__pycache__", ".DS_Store"}:
            ignored.add(name)
    return ignored


def _remove_excluded_paths(clone_root: Path) -> None:
    for relative_path in COPY_EXCLUDES:
        for target in clone_root.glob(relative_path):
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.exists():
                shutil.rmtree(target, ignore_errors=True)


def _copy_repo(source_root: Path, clone_root: Path) -> None:
    clone_root.mkdir(parents=True, exist_ok=True)

    if sys.platform == "darwin":
        clone_command = ["cp", "-cR", f"{source_root}/.", str(clone_root)]
        if subprocess.run(clone_command, check=False).returncode == 0:
            _remove_excluded_paths(clone_root)
            return

    if sys.platform.startswith("linux"):
        clone_command = ["cp", "-a", "--reflink=auto", f"{source_root}/.", str(clone_root)]
        if subprocess.run(clone_command, check=False).returncode == 0:
            _remove_excluded_paths(clone_root)
            return

    rsync = shutil.which("rsync")
    if rsync:
        command = [rsync, "-a", f"{source_root}/", f"{clone_root}/"]
        for pattern in COPY_EXCLUDES:
            command.extend(["--exclude", pattern])
        subprocess.run(command, check=True)
        return

    shutil.copytree(
        source_root,
        clone_root,
        dirs_exist_ok=True,
        ignore=_ignore_copy,
    )
    shutil.rmtree(clone_root / "app" / "env", ignore_errors=True)


def _wait_until_ready(url: str, process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    probe_url = f"{url}/api/status/audio"
    while time.time() < deadline:
        if process.poll() is not None:
            raise HarnessError(f"server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(probe_url, timeout=0.5) as response:
                if 200 <= int(getattr(response, "status", 0) or 0) < 300:
                    return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError, ValueError):
            pass
        time.sleep(DEFAULT_READY_POLL_SECONDS)
    raise HarnessError(f"server did not become ready within {timeout_seconds:.1f}s")


def _manifest_path(clone_root: Path) -> Path:
    return clone_root / DEFAULT_MANIFEST_NAME


def _read_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HarnessError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessError(f"manifest is invalid JSON: {path}") from exc


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tail_log(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _terminate_process_group(pid: int, timeout_seconds: float = 5.0) -> None:
    if pid <= 0:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return


def _create_clone_root(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp"))


def _start_harness(source_root: Path, port: int | None, prefix: str, ready_timeout_seconds: float) -> dict:
    owner_path = _claim_harness_ownership(source_root)
    clone_root = _create_clone_root(prefix)
    manifest_path = _manifest_path(clone_root)
    log_path = clone_root / "server.log"
    selected_port = int(port or _find_free_port())
    url = f"http://127.0.0.1:{selected_port}"
    python_path = _python_for_clone(source_root)

    try:
        _copy_repo(source_root, clone_root)
        app_cwd = clone_root / "app"
        env = os.environ.copy()
        env["PINOKIO_SHARE_LOCAL_PORT"] = str(selected_port)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")

        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                [str(python_path), "app.py"],
                cwd=app_cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        payload = {
            "created_at": time.time(),
            "clone_root": str(clone_root),
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
            "ownership_path": str(owner_path),
            "pid": int(process.pid),
            "port": selected_port,
            "python_path": str(python_path),
            "source_root": str(source_root),
            "url": url,
        }
        _write_manifest(manifest_path, payload)
        _update_harness_ownership(owner_path, payload)
        _wait_until_ready(url, process, ready_timeout_seconds)
        return payload
    except BaseException as exc:
        log_tail = _tail_log(log_path)
        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
            _terminate_process_group(int(manifest.get("pid") or 0))
        shutil.rmtree(clone_root, ignore_errors=True)
        _release_harness_ownership(source_root, manifest_path=str(manifest_path))
        if log_tail:
            raise HarnessError(f"{exc}\n\nRecent server log:\n{log_tail}") from exc
        raise


def _stop_harness(manifest_path: Path, keep_files: bool = False) -> dict:
    payload = _read_manifest(manifest_path)
    clone_root = Path(payload["clone_root"])
    log_path = Path(payload["log_path"])
    pid = int(payload.get("pid") or 0)
    source_root = Path(payload.get("source_root") or REPO_ROOT)
    stop_error = None

    try:
        _terminate_process_group(pid)
    except Exception as exc:  # pragma: no cover - defensive cleanup path
        stop_error = str(exc)
    finally:
        _release_harness_ownership(source_root, manifest_path=str(manifest_path))

    result = {
        "clone_root": str(clone_root),
        "log_path": str(log_path),
        "removed_files": not keep_files,
        "stopped_pid": pid,
    }
    if stop_error:
        result["stop_error"] = stop_error
    if keep_files:
        return result

    shutil.rmtree(clone_root, ignore_errors=True)
    return result


def _run_exec_mode(args: argparse.Namespace) -> int:
    if not args.command:
        raise HarnessError("exec mode requires a command after '--'")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise HarnessError("exec mode requires a command after '--'")

    payload = _start_harness(
        source_root=REPO_ROOT,
        port=args.port,
        prefix=args.prefix,
        ready_timeout_seconds=args.ready_timeout_seconds,
    )
    manifest_path = Path(payload["manifest_path"])
    env = os.environ.copy()
    env["THREADSPEAK_DEBUG_URL"] = payload["url"]
    env["THREADSPEAK_DEBUG_CLONE_ROOT"] = payload["clone_root"]
    env["THREADSPEAK_DEBUG_MANIFEST"] = payload["manifest_path"]
    env["THREADSPEAK_DEBUG_LOG_PATH"] = payload["log_path"]

    exit_code = 0
    try:
        completed = subprocess.run(command, env=env, check=False)
        exit_code = int(completed.returncode)
        return exit_code
    finally:
        _stop_harness(manifest_path, keep_files=args.keep_files)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    start_parser = subparsers.add_parser("start", help="Create an isolated clone and start its server.")
    start_parser.add_argument("--port", type=int, help="Port for the isolated server.")
    start_parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefix for the temp clone directory.")
    start_parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help="How long to wait for the isolated server to become ready.",
    )

    stop_parser = subparsers.add_parser("stop", help="Stop an isolated clone and remove its files.")
    stop_parser.add_argument("--manifest", required=True, help="Path to the manifest returned by 'start'.")
    stop_parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Stop the server but keep the cloned files for inspection.",
    )

    exec_parser = subparsers.add_parser(
        "exec",
        help="Start an isolated clone, run a command with debug env vars, then always clean up.",
    )
    exec_parser.add_argument("--port", type=int, help="Port for the isolated server.")
    exec_parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefix for the temp clone directory.")
    exec_parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help="How long to wait for the isolated server to become ready.",
    )
    exec_parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep the cloned files after the wrapped command exits.",
    )
    exec_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after '--'.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command_name == "start":
            payload = _start_harness(
                source_root=REPO_ROOT,
                port=args.port,
                prefix=args.prefix,
                ready_timeout_seconds=args.ready_timeout_seconds,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command_name == "stop":
            payload = _stop_harness(Path(args.manifest), keep_files=args.keep_files)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command_name == "exec":
            return _run_exec_mode(args)

        raise HarnessError(f"unsupported subcommand: {args.command_name}")
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
