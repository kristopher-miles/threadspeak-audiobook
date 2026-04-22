"""Opt-in heavy live full-project flow against a fresh clone and real local backends."""

import errno
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager

import pytest
import requests

from llm import LMStudioModelLoadService

from ._stage_ui_helpers import *  # noqa: F401,F403


REQUIRED_LIVE_SPEAKERS = {"NARRATOR"}
LMSTUDIO_API_KEY = "local"
LMSTUDIO_MODEL_CONTEXT_LENGTH = 16384
LMSTUDIO_MODEL_WAIT_TIMEOUT_SECONDS = 300
LMSTUDIO_MODEL_POLL_INTERVAL_SECONDS = 2
LIVE_FRESH_CLONE_LOCK_PATH = os.path.join(
    tempfile.gettempdir(),
    "threadspeak_fresh_clone_live_full_project_narrated.lock",
)
LIVE_PARTIAL_POINTER_PATH = os.path.join(
    tempfile.gettempdir(),
    "threadspeak_fresh_clone_live_narrated_partial_state.json",
)
LIVE_OUTPUT_DIR = os.path.join(
    tempfile.gettempdir(),
    "threadspeak_fresh_clone_live_narrated_outputs",
)
LIVE_PARTIAL_CHECKPOINT_BASENAME = "fresh_clone_live_narrated_checkpoint.json"
LIVE_STAGE1 = "stage1_script_generation"
LIVE_STAGE2 = "stage2_voice_suggestions"
LIVE_STAGE3 = "stage3_render_and_proofread"
LIVE_STAGE4 = "stage4_export_merged"

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - platform guard
    fcntl = None


def _seconds_per_request(duration_seconds: float, request_count: int) -> float | None:
    if int(request_count or 0) <= 0:
        return None
    return float(duration_seconds) / float(request_count)


def _format_live_benchmark_block(
    *,
    status: str,
    lm_base_url: str,
    selected_model: str,
    selected_preference: str,
    available_tool_models: list[str],
    phases: list[dict],
    final_audio: dict | None,
    failure_reason: str = "",
) -> str:
    lines = [
        "=== Fresh Clone Live Full-Project Benchmark ===",
        f"Status: {status}",
        f"LM Studio URL: {lm_base_url}",
        f"Selected model: {selected_model}",
        f"Selection rule: {selected_preference}",
        f"Available tool models: {', '.join(available_tool_models) if available_tool_models else '(none)'}",
        f"Reached timed phases: {', '.join(item.get('name', '') for item in phases) if phases else '(none)'}",
    ]
    if phases:
        lines.append("Timed phase metrics:")
        for row in phases:
            sec_per_req = row.get("sec_per_request")
            sec_per_req_text = "n/a" if sec_per_req is None else f"{float(sec_per_req):.2f}s"
            lines.append(
                "  - "
                f"{row.get('name')}: "
                f"duration={float(row.get('duration_seconds') or 0.0):.2f}s, "
                f"requests={int(row.get('request_count') or 0)}, "
                f"sec_per_request={sec_per_req_text}, "
                f"model={row.get('model_label') or ''}"
            )
    if final_audio:
        lines.append(
            "Final MP3: "
            f"duration={float(final_audio.get('duration_seconds') or 0.0):.2f}s, "
            f"size={int(final_audio.get('size_bytes') or 0)} bytes, "
            f"path={str(final_audio.get('path') or '')}"
        )
    if failure_reason:
        lines.append(f"Failure: {failure_reason}")
    lines.append("=== End Benchmark ===")
    return "\n".join(lines)


@contextmanager
def _single_live_fresh_clone_guard():
    if fcntl is None:
        with _exclusive_run_lock("fresh_clone_live_full_project_flow_real_local_backends"):
            yield
        return
    handle = open(LIVE_FRESH_CLONE_LOCK_PATH, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            handle.seek(0)
            owner = (handle.read() or "").strip()
            owner_hint = f" lock owner: {owner}" if owner else ""
            raise AssertionError(
                "Refusing to run fresh-clone live full-project test concurrently. "
                "Another instance is already running. Exit the old run first and retry. "
                f"Lock file: {LIVE_FRESH_CLONE_LOCK_PATH}.{owner_hint}"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} started_at={int(time.time())}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _lmstudio_model_name_matches(model_payload: dict, wanted_model: str) -> bool:
    wanted = str(wanted_model or "").strip()
    if not wanted:
        return False
    candidates = {
        str(model_payload.get("key") or "").strip(),
        str(model_payload.get("display_name") or "").strip(),
    }
    for instance in model_payload.get("loaded_instances") or []:
        if isinstance(instance, dict):
            candidates.add(str(instance.get("id") or "").strip())
    return wanted in candidates


def _wait_for_lmstudio_loaded_state(
    *,
    service: LMStudioModelLoadService,
    origin: str,
    model_name: str,
    expect_loaded_instances: int,
) -> None:
    deadline = time.time() + LMSTUDIO_MODEL_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            payload = service.list_models(base_url=origin, api_key=LMSTUDIO_API_KEY)
        except Exception:
            time.sleep(LMSTUDIO_MODEL_POLL_INTERVAL_SECONDS)
            continue
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            time.sleep(LMSTUDIO_MODEL_POLL_INTERVAL_SECONDS)
            continue

        total_loaded_instances = 0
        target_loaded_instances = 0
        for model in models:
            if not isinstance(model, dict):
                continue
            loaded_instances = [
                item
                for item in (model.get("loaded_instances") or [])
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            ]
            total_loaded_instances += len(loaded_instances)
            if _lmstudio_model_name_matches(model, model_name):
                target_loaded_instances += len(loaded_instances)

        if expect_loaded_instances <= 0:
            if total_loaded_instances == 0:
                return
        else:
            if target_loaded_instances >= expect_loaded_instances and total_loaded_instances == target_loaded_instances:
                return

        time.sleep(LMSTUDIO_MODEL_POLL_INTERVAL_SECONDS)

    raise AssertionError(
        "Timed out waiting for LM Studio model state. "
        f"model={model_name} expected_loaded_instances={expect_loaded_instances}"
    )


def _unload_all_lmstudio_models_once_and_verify(*, service: LMStudioModelLoadService, origin: str) -> None:
    result = service.unload_all_models(base_url=origin, api_key=LMSTUDIO_API_KEY)
    if str(result.get("status") or "") != "ok":
        raise AssertionError(f"LM Studio unload-all returned unexpected status: {result}")
    _wait_for_lmstudio_loaded_state(
        service=service,
        origin=origin,
        model_name="",
        expect_loaded_instances=0,
    )


def _load_lmstudio_model_once_and_verify(
    *,
    service: LMStudioModelLoadService,
    origin: str,
    model_name: str,
) -> None:
    payload = service.load_model(
        base_url=origin,
        api_key=LMSTUDIO_API_KEY,
        model_name=model_name,
        context_length=LMSTUDIO_MODEL_CONTEXT_LENGTH,
        echo_load_config=True,
    )
    if str(payload.get("status") or "") != "loaded":
        raise AssertionError(f"LM Studio model load returned unexpected status: {payload}")
    _wait_for_lmstudio_loaded_state(
        service=service,
        origin=origin,
        model_name=model_name,
        expect_loaded_instances=1,
    )


def _count_new_mode_stage_errors(base_url: str) -> dict:
    payload = _fetch_task_status(base_url, "new_mode_workflow")
    logs = [str(item) for item in _extract_logs(payload)]

    def _is_assign_error(line: str) -> bool:
        lower = line.lower()
        return (
            "assign dialogue" in line
            and (
                "api call failed" in lower
                or "no speaker" in lower
            )
        )

    def _is_extract_error(line: str) -> bool:
        lower = line.lower()
        return (
            "extract temperament" in line
            and (
                "api call failed" in lower
                or "no mood" in lower
            )
        )

    assign_errors = [line for line in logs if _is_assign_error(line)]
    extract_errors = [line for line in logs if _is_extract_error(line)]
    return {
        "assign_errors": assign_errors,
        "extract_errors": extract_errors,
    }


def _wait_for_task_success(
    *,
    base_url: str,
    task_name: str,
    success_fragment: str,
    inactivity_seconds: float = 120.0,
    max_total_seconds: float = 1200.0,
) -> dict:
    def probe():
        payload = _fetch_task_status(base_url, task_name)
        logs = _extract_logs(payload)
        errors = _collect_fatal_log_lines(logs)
        return {
            "running": bool(payload.get("running")),
            "last_error": payload.get("last_error"),
            "logs": logs,
            "errors": errors,
            "has_success_log": any(success_fragment in line for line in logs),
        }

    def done(snapshot):
        if snapshot.get("last_error"):
            raise AssertionError(f"{task_name} failed: {snapshot['last_error']}")
        if snapshot.get("errors"):
            raise AssertionError(f"{task_name} emitted fatal errors: {snapshot['errors']}")
        return (not snapshot.get("running")) and bool(snapshot.get("has_success_log"))

    return _wait_for_activity(
        f"Waiting for {task_name} completion",
        probe,
        done,
        inactivity_seconds=float(inactivity_seconds),
        max_total_seconds=float(max_total_seconds),
        poll_seconds=0.8,
    )


def _retry_small_script_stage_failures_once(
    *,
    base_url: str,
    assign_retry_limit: int = 2,
    extract_retry_limit: int = 2,
) -> dict:
    error_snapshot = _count_new_mode_stage_errors(base_url)
    assign_count = len(error_snapshot.get("assign_errors") or [])
    extract_count = len(error_snapshot.get("extract_errors") or [])

    assign_retried = False
    extract_retried = False

    if 0 < assign_count <= max(1, int(assign_retry_limit)):
        response = requests.post(f"{base_url}/api/assign_dialogue", timeout=20)
        _assert_status(response, 200, "retry assign_dialogue")
        _wait_for_task_success(
            base_url=base_url,
            task_name="assign_dialogue",
            success_fragment="Task assign_dialogue completed successfully.",
        )
        assign_retried = True

    if 0 < extract_count <= max(1, int(extract_retry_limit)):
        response = requests.post(f"{base_url}/api/extract_temperament", timeout=20)
        _assert_status(response, 200, "retry extract_temperament")
        _wait_for_task_success(
            base_url=base_url,
            task_name="extract_temperament",
            success_fragment="Task extract_temperament completed successfully.",
        )
        extract_retried = True

    if assign_retried or extract_retried:
        response = requests.post(f"{base_url}/api/create_script", timeout=20)
        _assert_status(response, 200, "rerun create_script after stage retries")
        _wait_for_task_success(
            base_url=base_url,
            task_name="create_script",
            success_fragment="Task create_script completed successfully.",
        )

    return {
        "assign_error_count": assign_count,
        "extract_error_count": extract_count,
        "assign_retried": assign_retried,
        "extract_retried": extract_retried,
    }


def _read_json_if_exists(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _partial_checkpoint_path(clone_root: str) -> str:
    return os.path.join(clone_root, LIVE_PARTIAL_CHECKPOINT_BASENAME)


def _save_partial_pointer(*, clone_root: str, last_status: str, failed_stage: str = "", failure_reason: str = "") -> None:
    _write_json(
        LIVE_PARTIAL_POINTER_PATH,
        {
            "clone_root": str(clone_root),
            "checkpoint_path": _partial_checkpoint_path(str(clone_root)),
            "last_status": str(last_status),
            "failed_stage": str(failed_stage),
            "failure_reason": str(failure_reason),
            "updated_at": int(time.time()),
        },
    )


def _load_partial_checkpoint(clone_root: str) -> dict:
    payload = _read_json_if_exists(_partial_checkpoint_path(clone_root))
    completed = [str(item) for item in (payload.get("completed_stages") or []) if str(item).strip()]
    return {
        "completed_stages": sorted(set(completed)),
        "failed_stage": str(payload.get("failed_stage") or ""),
        "failure_reason": str(payload.get("failure_reason") or ""),
        "final_audio": payload.get("final_audio") if isinstance(payload.get("final_audio"), dict) else None,
    }


def _save_partial_checkpoint(
    *,
    clone_root: str,
    completed_stages: set[str],
    failed_stage: str = "",
    failure_reason: str = "",
    final_audio: dict | None = None,
) -> None:
    payload = {
        "completed_stages": sorted(str(item) for item in completed_stages if str(item).strip()),
        "failed_stage": str(failed_stage or ""),
        "failure_reason": str(failure_reason or ""),
        "updated_at": int(time.time()),
    }
    if isinstance(final_audio, dict):
        payload["final_audio"] = final_audio
    _write_json(_partial_checkpoint_path(clone_root), payload)


def _is_safe_temp_fresh_clone_root(path: str) -> bool:
    candidate = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not candidate:
        return False
    temp_root = os.path.abspath(tempfile.gettempdir())
    base_name = os.path.basename(candidate.rstrip(os.sep))
    return bool(
        base_name.startswith("threadspeak_e2e_fresh_clone_")
        and (candidate == temp_root or candidate.startswith(temp_root + os.sep))
    )


def _slugify(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text or "").strip().lower())
    cleaned = cleaned.strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "model"


def _persist_final_mp3_artifact(*, final_audio: dict, selected_model: str) -> dict:
    source_path = str((final_audio or {}).get("path") or "").strip()
    if not source_path or not os.path.isfile(source_path):
        raise AssertionError(f"Missing final MP3 to preserve before cleanup: {source_path or '(empty path)'}")
    os.makedirs(LIVE_OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = _slugify(selected_model)
    destination_path = os.path.join(
        LIVE_OUTPUT_DIR,
        f"fresh_clone_live_{timestamp}_{model_slug}.mp3",
    )
    if os.path.exists(destination_path):
        destination_path = os.path.join(
            LIVE_OUTPUT_DIR,
            f"fresh_clone_live_{timestamp}_{model_slug}_{os.getpid()}_{time.time_ns()}.mp3",
        )
    shutil.copy2(source_path, destination_path)
    preserved = dict(final_audio or {})
    preserved["source_path"] = source_path
    preserved["path"] = destination_path
    return preserved


def _cleanup_live_run_artifacts_after_success(*, clone_root: str, checkpoint_path: str = "") -> None:
    try:
        os.remove(LIVE_PARTIAL_POINTER_PATH)
    except FileNotFoundError:
        pass

    checkpoint_candidate = os.path.abspath(os.path.expanduser(str(checkpoint_path or "").strip()))
    if checkpoint_candidate and os.path.isfile(checkpoint_candidate):
        os.remove(checkpoint_candidate)

    root_candidate = os.path.abspath(os.path.expanduser(str(clone_root or "").strip()))
    if not root_candidate:
        return
    if _is_safe_temp_fresh_clone_root(root_candidate):
        if os.path.isdir(root_candidate):
            shutil.rmtree(root_candidate, ignore_errors=True)
            if os.path.exists(root_candidate):
                raise AssertionError(
                    "Expected fresh-clone root cleanup after successful live test, "
                    f"but directory still exists: {root_candidate}"
                )
    elif os.path.exists(root_candidate):
        raise AssertionError(
            "Refusing to delete unexpected successful-run residue path outside temp fresh-clone scope: "
            f"{root_candidate}"
        )


def _clear_non_partial_clone_residue(partial_pointer: dict) -> None:
    clone_root = str(partial_pointer.get("clone_root") or "").strip()
    if clone_root:
        candidate = os.path.abspath(os.path.expanduser(clone_root))
        if _is_safe_temp_fresh_clone_root(candidate):
            if os.path.isdir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
                if os.path.exists(candidate):
                    raise AssertionError(
                        "Non-partial run requires deleting prior fresh-clone residue, "
                        f"but deletion did not complete: {candidate}"
                    )
        elif os.path.exists(candidate):
            raise AssertionError(
                "Refusing to delete unexpected non-partial residue path outside temp fresh-clone scope: "
                f"{candidate}"
            )
    try:
        os.remove(LIVE_PARTIAL_POINTER_PATH)
    except FileNotFoundError:
        pass


def _open_app_tab(page, *, base_url: str, tab_selector: str, panel_selector: str, label: str) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    _wait_for_bootstrap_ready(page)
    _wait_for_nav_unlocked(page, tab_selector, label)
    page.locator(tab_selector).click()
    _wait_for_activity(
        f"Waiting for {label}",
        lambda: {"visible": bool(page.locator(panel_selector).is_visible())},
        lambda snapshot: bool(snapshot.get("visible")),
    )


@pytest.mark.fresh_clone_live_narrated_e2e
def test_fresh_clone_live_full_project_flow_real_local_backends_narrated(request):
    partial_mode = bool(request.config.getoption("--fresh-clone-live-partial"))
    partial_pointer = _read_json_if_exists(LIVE_PARTIAL_POINTER_PATH)
    resume_clone_root = ""
    successful_run = False
    successful_clone_root = ""
    successful_checkpoint_path = ""
    retained_mp3_path = ""
    if partial_mode:
        candidate = str(partial_pointer.get("clone_root") or "").strip()
        if candidate and os.path.isdir(candidate):
            resume_clone_root = candidate
    else:
        _clear_non_partial_clone_residue(partial_pointer)

    with _hard_test_timeout(3600, label="fresh-clone live full-project real-backend E2E"):
        with _single_live_fresh_clone_guard():
            discovery = _discover_lmstudio_tool_model(origin=LMSTUDIO_DEFAULT_ORIGIN)
            if str(discovery.get("status") or "") != "ok":
                reason = str(discovery.get("reason") or "LM Studio backend cannot be reached")
                pytest.skip(f"LLM backend can't be reached: {reason}")

            lm_base_url = str(discovery.get("base_url") or LMSTUDIO_DEFAULT_V1_BASE_URL)
            selected_model = str(discovery.get("model_name") or "").strip()
            assert selected_model, f"LM Studio discovery returned an empty model selection: {discovery}"
            available_tool_models = [
                str(item).strip()
                for item in (discovery.get("available_tool_models") or [])
                if str(item).strip()
            ]

            lmstudio_service = LMStudioModelLoadService(timeout_seconds=240)
            _unload_all_lmstudio_models_once_and_verify(
                service=lmstudio_service,
                origin=LMSTUDIO_DEFAULT_ORIGIN,
            )
            _load_lmstudio_model_once_and_verify(
                service=lmstudio_service,
                origin=LMSTUDIO_DEFAULT_ORIGIN,
                model_name=selected_model,
            )

            with _exclusive_run_lock("fresh_clone_live_full_project_flow_real_local_backends"):
                book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")
                assert os.path.exists(book_path), f"Missing book fixture: {book_path}"

                console_errors: list[str] = []
                page_errors: list[str] = []
                warnings: list[str] = []
                http_failures: list[str] = []
                phase_rows: list[dict] = []
                final_audio: dict | None = None

                with _report_directory("threadspeak_fresh_clone_live_report_"):
                    with _FreshCloneServer(
                        source_ref="HEAD",
                        include_worktree_changes=True,
                        reuse_source_env=True,
                        bootstrap_config_values={
                            "asr": {
                                "parallel_workers": 1,
                                "cpu_threads": 1,
                            },
                            "tts": {
                                "local_backend": "auto",
                            }
                        },
                        existing_repo_root=resume_clone_root or None,
                        preserve_clone_root=partial_mode,
                    ) as app_server:
                        successful_clone_root = app_server.repo_root
                        successful_checkpoint_path = _partial_checkpoint_path(app_server.repo_root)
                        if partial_mode:
                            _save_partial_pointer(
                                clone_root=app_server.repo_root,
                                last_status="running",
                                failed_stage="",
                                failure_reason="",
                            )
                        checkpoint = _load_partial_checkpoint(app_server.repo_root) if partial_mode else {}
                        completed_stages = set(str(item) for item in (checkpoint.get("completed_stages") or []))
                        if isinstance(checkpoint.get("final_audio"), dict):
                            final_audio = dict(checkpoint["final_audio"])
                        current_stage = "startup"

                        with sync_playwright() as playwright:
                            browser = playwright.chromium.launch(headless=True)
                            context = browser.new_context()
                            page = context.new_page()

                            def _on_console(message):
                                text = str(message.text or "").strip()
                                kind = str(message.type or "").strip().lower()
                                if kind == "error":
                                    console_errors.append(text)
                                elif kind == "warning":
                                    warnings.append(text)

                            def _on_page_error(err):
                                page_errors.append(str(err))

                            def _on_response(response):
                                try:
                                    status = int(response.status)
                                except Exception:
                                    status = 0
                                if status >= 400:
                                    method = str(getattr(response.request, "method", "") or "")
                                    http_failures.append(f"{status} {method} {response.url}")

                            page.on("console", _on_console)
                            page.on("pageerror", _on_page_error)
                            page.on("response", _on_response)

                            try:
                                if LIVE_STAGE1 in completed_stages:
                                    _open_app_tab(
                                        page,
                                        base_url=app_server.base_url,
                                        tab_selector='.nav-link[data-tab="voices"]',
                                        panel_selector="#voices-tab",
                                        label="Voices tab",
                                    )
                                else:
                                    current_stage = LIVE_STAGE1
                                    phase_start = time.perf_counter()
                                    _run_stage1_to_voices_tab(
                                        page=page,
                                        app_base_url=app_server.base_url,
                                        book_path=book_path,
                                        script_llm_base_url=lm_base_url,
                                        script_llm_model_name=selected_model,
                                        tts_mode="local",
                                        tts_parallel_workers=1,
                                        script_workflow_inactivity_seconds=180.0,
                                        script_workflow_max_total_seconds=1800.0,
                                        fail_on_script_logged_errors=False,
                                        full_cast=False,
                                    )

                                    retry_summary = _retry_small_script_stage_failures_once(
                                        base_url=app_server.base_url,
                                        assign_retry_limit=2,
                                        extract_retry_limit=2,
                                    )

                                    phase_seconds = time.perf_counter() - phase_start
                                    stage1_requests = _count_task_llm_mode_events(app_server.base_url, "new_mode_workflow")
                                    phase_rows.append(
                                        {
                                            "name": LIVE_STAGE1,
                                            "duration_seconds": phase_seconds,
                                            "request_count": int(stage1_requests),
                                            "sec_per_request": _seconds_per_request(phase_seconds, int(stage1_requests)),
                                            "model_label": selected_model,
                                            "retry_summary": retry_summary,
                                        }
                                    )
                                    completed_stages.add(LIVE_STAGE1)
                                    if partial_mode:
                                        _save_partial_checkpoint(
                                            clone_root=app_server.repo_root,
                                            completed_stages=completed_stages,
                                            final_audio=final_audio,
                                        )

                                if LIVE_STAGE2 not in completed_stages:
                                    current_stage = LIVE_STAGE2
                                    phase_start = time.perf_counter()
                                    voice_summary = _run_stage2_voices_flow_relaxed_live(
                                        page=page,
                                        voice_server_base_url=lm_base_url,
                                        voice_model_name=selected_model,
                                        required_speakers=REQUIRED_LIVE_SPEAKERS,
                                        min_total_speakers=1,
                                        retry_partial_failures_once=True,
                                        retry_failure_limit=2,
                                        voice_generation_inactivity_seconds=180.0,
                                        voice_generation_max_total_seconds=1800.0,
                                    )
                                    phase_seconds = time.perf_counter() - phase_start
                                    stage2_requests = int(voice_summary.get("generated_count") or 0)
                                    phase_rows.append(
                                        {
                                            "name": LIVE_STAGE2,
                                            "duration_seconds": phase_seconds,
                                            "request_count": stage2_requests,
                                            "sec_per_request": _seconds_per_request(phase_seconds, stage2_requests),
                                            "model_label": selected_model,
                                        }
                                    )
                                    completed_stages.add(LIVE_STAGE2)
                                    if partial_mode:
                                        _save_partial_checkpoint(
                                            clone_root=app_server.repo_root,
                                            completed_stages=completed_stages,
                                            final_audio=final_audio,
                                        )

                                if LIVE_STAGE3 not in completed_stages:
                                    current_stage = LIVE_STAGE3
                                    phase_start = time.perf_counter()
                                    _run_stage3_to_stage4_flow_from_editor(
                                        page=page,
                                        app_base_url=app_server.base_url,
                                        retry_partial_render_failures_once=True,
                                        retry_failure_limit=2,
                                        render_inactivity_seconds=180.0,
                                        render_max_total_seconds=3000.0,
                                        proofread_inactivity_seconds=180.0,
                                        proofread_max_total_seconds=3000.0,
                                    )
                                    strict_snapshot = _wait_for_stage4_strict_pass(
                                        page=page,
                                        app_base_url=app_server.base_url,
                                        inactivity_seconds=180.0,
                                        max_total_seconds=600.0,
                                        allow_failures=True,
                                    )
                                    ui_summary = dict(strict_snapshot.get("ui") or {})
                                    assert int(ui_summary.get("row_count") or 0) > 0
                                    total_outcomes = (
                                        int(ui_summary.get("passed") or 0)
                                        + int(ui_summary.get("failed") or 0)
                                        + int(ui_summary.get("auto_failed") or 0)
                                    )
                                    assert total_outcomes >= int(ui_summary.get("row_count") or 0)
                                    assert str(ui_summary.get("phase_text") or "").strip().lower() == "complete"

                                    phase_seconds = time.perf_counter() - phase_start
                                    audio_payload = _fetch_task_status(app_server.base_url, "audio")
                                    recent_jobs = list((audio_payload or {}).get("recent_jobs") or [])
                                    latest_job = dict(recent_jobs[0] or {}) if recent_jobs else {}
                                    stage3_requests = int(latest_job.get("processed_clips") or 0)
                                    phase_rows.append(
                                        {
                                            "name": LIVE_STAGE3,
                                            "duration_seconds": phase_seconds,
                                            "request_count": stage3_requests,
                                            "sec_per_request": _seconds_per_request(phase_seconds, stage3_requests),
                                            "model_label": "local-auto-tts",
                                        }
                                    )
                                    completed_stages.add(LIVE_STAGE3)
                                    if partial_mode:
                                        _save_partial_checkpoint(
                                            clone_root=app_server.repo_root,
                                            completed_stages=completed_stages,
                                            final_audio=final_audio,
                                        )

                                if LIVE_STAGE4 not in completed_stages or not isinstance(final_audio, dict):
                                    current_stage = LIVE_STAGE4
                                    assert app_server.layout is not None, "Missing fresh-clone runtime layout."
                                    final_audio = _export_merged_audiobook_via_ui(
                                        page,
                                        app_base_url=app_server.base_url,
                                        layout=app_server.layout,
                                        min_duration_seconds=180.0,
                                    )
                                    completed_stages.add(LIVE_STAGE4)
                                    if partial_mode:
                                        _save_partial_checkpoint(
                                            clone_root=app_server.repo_root,
                                            completed_stages=completed_stages,
                                            final_audio=final_audio,
                                        )

                                final_audio = _persist_final_mp3_artifact(
                                    final_audio=dict(final_audio or {}),
                                    selected_model=selected_model,
                                )
                                retained_mp3_path = str(final_audio.get("path") or "").strip()
                                successful_run = True
                                report = _format_live_benchmark_block(
                                    status="PASS",
                                    lm_base_url=lm_base_url,
                                    selected_model=selected_model,
                                    selected_preference=str(discovery.get("selected_preference") or ""),
                                    available_tool_models=available_tool_models,
                                    phases=phase_rows,
                                    final_audio=final_audio,
                                )
                                print(report, flush=True)

                                assert not console_errors, _report_console(console_errors, page_errors, warnings)
                                assert not page_errors, _report_console(console_errors, page_errors, warnings)
                            except Exception as exc:
                                if partial_mode:
                                    _save_partial_checkpoint(
                                        clone_root=app_server.repo_root,
                                        completed_stages=completed_stages,
                                        failed_stage=current_stage,
                                        failure_reason=str(exc),
                                        final_audio=final_audio,
                                    )
                                    _save_partial_pointer(
                                        clone_root=app_server.repo_root,
                                        last_status="fail",
                                        failed_stage=current_stage,
                                        failure_reason=str(exc),
                                    )

                                script_logs = ""
                                proofread_logs = ""
                                audio_logs = ""
                                try:
                                    script_logs = page.locator("#script-logs").inner_text(timeout=2000)
                                except Exception:
                                    script_logs = ""
                                try:
                                    proofread_logs = page.locator("#proofread-logs").inner_text(timeout=2000)
                                except Exception:
                                    proofread_logs = ""
                                try:
                                    audio_logs = page.locator("#audio-logs").inner_text(timeout=2000)
                                except Exception:
                                    audio_logs = ""

                                report = _format_live_benchmark_block(
                                    status="FAIL",
                                    lm_base_url=lm_base_url,
                                    selected_model=selected_model,
                                    selected_preference=str(discovery.get("selected_preference") or ""),
                                    available_tool_models=available_tool_models,
                                    phases=phase_rows,
                                    final_audio=final_audio,
                                    failure_reason=str(exc),
                                )
                                print(report, flush=True)

                                partial_hint = ""
                                if partial_mode:
                                    partial_hint = (
                                        f"Partial clone root: {app_server.repo_root}\n"
                                        f"Partial checkpoint: {_partial_checkpoint_path(app_server.repo_root)}\n"
                                    )

                                raise AssertionError(
                                    f"Fresh-clone live full-project UI flow failed: {exc}\n"
                                    f"{partial_hint}"
                                    f"Script logs tail:\n{script_logs[-2000:]}\n"
                                    f"Proofread logs tail:\n{proofread_logs[-2000:]}\n"
                                    f"Audio logs tail:\n{audio_logs[-2000:]}\n"
                                    f"HTTP failures:\n{chr(10).join(http_failures[-20:]) or 'none'}\n"
                                    f"Fresh clone server log tail:\n{_tail_file(app_server.log_path, max_chars=5000)}\n"
                                    f"{report}\n"
                                    f"{_report_console(console_errors, page_errors, warnings)}"
                                ) from exc
                            finally:
                                context.close()
                                browser.close()
    if successful_run:
        _cleanup_live_run_artifacts_after_success(
            clone_root=successful_clone_root,
            checkpoint_path=successful_checkpoint_path,
        )
        if retained_mp3_path:
            print(f"[e2e-cleanup] retained final MP3: {retained_mp3_path}", flush=True)
