"""Reset failure repro for submit-then-reset error-toast failures."""

from ._stage_ui_helpers import *  # noqa: F401,F403


def test_reset_failure():
    with _exclusive_run_lock("fresh_clone_stage1_submit_then_immediate_reset"):
        fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
        script_fixture_path = os.path.join(fixtures_dir, "lmstudio_generate_script_test_book.json")
        book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")

        assert os.path.exists(script_fixture_path), f"Missing fixture: {script_fixture_path}"
        assert os.path.exists(book_path), f"Missing book fixture: {book_path}"

        script_payload = _read_json(script_fixture_path)
        script_model = str(((script_payload.get("metadata") or {}).get("model_name") or "").strip())
        assert script_model, "Script fixture metadata.model_name is required."

        console_errors: list[str] = []
        page_errors: list[str] = []
        warnings: list[str] = []
        http_failures: list[str] = []

        with _report_directory("r_") as report_root:
            script_lm_trace = os.path.join(report_root, "lm-script-trace.jsonl")
            env_overrides = {
                "THREADSPEAK_E2E_SIM_ENABLED": "1",
                "THREADSPEAK_E2E_SIM_STRICT": "1",
                MODEL_DOWNLOAD_DISABLE_ENV: "1",
            }

            with LMStudioSimServer(
                script_fixture_path,
                trace_path=script_lm_trace,
                trace_label="fresh-clone-stage1-script-lm",
            ) as script_server:
                with _FreshCloneServer(
                    include_worktree_changes=False,
                    reuse_source_env=True,
                    env_overrides=env_overrides,
                    bootstrap_config_values={
                        "llm": {
                            "base_url": f"{script_server.base_url}/v1",
                            "api_key": "local",
                            "model_name": script_model,
                            "llm_workers": 1,
                        },
                        "generation": {
                            "legacy_mode": False,
                            "chunk_size": 600,
                            "max_tokens": 1024,
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 0.0,
                            "banned_tokens": [],
                            "temperament_words": 150,
                        },
                    },
                ) as app_server:
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
                            _install_error_toast_guard(page)
                            page.goto(app_server.base_url, wait_until="domcontentloaded", timeout=10000)
                            _wait_for_bootstrap_ready(page)
                            _wait_for_script_tab_ready(page)
                            _maybe_reset_project_from_script_tab(page)

                            with page.expect_response(
                                lambda response: (
                                    response.url.endswith("/api/upload")
                                    and response.request.method == "POST"
                                    and response.status == 200
                                ),
                                timeout=10000,
                            ):
                                page.locator("#file-upload").set_input_files(book_path)
                            _wait_for_upload_loaded(page)

                            process_voices_toggle = page.locator("#process-voices-toggle-v2")
                            if process_voices_toggle.is_checked():
                                process_voices_toggle.click()
                            assert not process_voices_toggle.is_checked(), "Process Voices must be unchecked for repro test."

                            with page.expect_response(
                                lambda response: (
                                    response.url.endswith("/api/new_mode_workflow/start")
                                    and response.request.method == "POST"
                                    and response.status == 200
                                ),
                                timeout=10000,
                            ):
                                page.locator("#btn-process-script-v2").click()

                            _wait_for_activity(
                                "Waiting for stage-1 submission to register",
                                lambda: page.evaluate(
                                    """() => {
                                        const btn = document.querySelector('#btn-process-script-v2');
                                        const logs = String(document.querySelector('#script-logs')?.innerText || '');
                                        return {
                                            button_hidden: !!btn && getComputedStyle(btn).display === 'none',
                                            button_disabled: !!btn && !!btn.disabled,
                                            logs_len: logs.length,
                                        };
                                    }"""
                                ),
                                lambda snapshot: bool(
                                    snapshot.get("button_hidden") or snapshot.get("button_disabled") or int(snapshot.get("logs_len") or 0) > 0
                                ),
                                max_total_seconds=20.0,
                            )

                            _reset_project_from_script_tab(page)

                            reset_snapshot = _wait_for_activity(
                                "Waiting for Script tab reset state",
                                lambda: page.evaluate(
                                    """() => {
                                        const statusEl = document.querySelector('#upload-status');
                                        const text = String(statusEl?.innerText || '').trim();
                                        const hasSuccess = !!statusEl?.querySelector('.text-success');
                                        const uploadSection = document.querySelector('#file-upload-section');
                                        const uploadVisible = !!uploadSection && getComputedStyle(uploadSection).display !== 'none';
                                        return {
                                            upload_text: text,
                                            has_success: hasSuccess,
                                            upload_visible: uploadVisible,
                                        };
                                    }"""
                                ),
                                lambda snapshot: (
                                    "Loaded:" not in str(snapshot.get("upload_text") or "")
                                    and not bool(snapshot.get("has_success"))
                                    and bool(snapshot.get("upload_visible"))
                                ),
                            )
                            assert "Loaded:" not in str(reset_snapshot.get("upload_text") or "")
                            _assert_no_error_toasts_for(
                                page,
                                seconds=3.0,
                                context="post-reset settle window",
                            )

                            assert not console_errors, _report_console(console_errors, page_errors, warnings)
                            assert not page_errors, _report_console(console_errors, page_errors, warnings)
                        except Exception as exc:
                            script_logs = ""
                            try:
                                script_logs = page.locator("#script-logs").inner_text(timeout=2000)
                            except Exception:
                                script_logs = ""
                            raise AssertionError(
                                f"Fresh-clone stage1 submit-then-reset flow failed: {exc}\n"
                                f"Script logs tail:\n{script_logs[-2000:]}\n"
                                f"HTTP failures:\n{chr(10).join(http_failures[-20:]) or 'none'}\n"
                                f"LM script trace tail ({script_lm_trace}):\n{_tail_file(script_lm_trace)}\n"
                                f"{_report_console(console_errors, page_errors, warnings)}"
                            ) from exc
                        finally:
                            context.close()
                            browser.close()
