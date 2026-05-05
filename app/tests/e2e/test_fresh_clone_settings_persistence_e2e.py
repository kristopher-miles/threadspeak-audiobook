"""Focused fresh-clone Settings persistence coverage."""

from ._stage_ui_helpers import *  # noqa: F401,F403


@pytest.mark.fresh_clone_e2e
def test_fresh_clone_settings_bootstrap_defaults_and_restart_persistence():
    with _hard_test_timeout(420, label="fresh-clone Settings persistence E2E"):
        with _exclusive_run_lock("fresh_clone_settings_bootstrap_defaults_and_restart_persistence"):
            console_errors: list[str] = []
            page_errors: list[str] = []
            warnings: list[str] = []
            http_failures: list[str] = []

            edited_values = {
                "llm_base_url": "http://127.0.0.1:4321/v1",
                "llm_api_key": "persisted-test-key",
                "llm_model_name": "persisted-test-model",
                "llm_workers": 3,
                "tts_provider": "qwen3",
                "script_max_length": 321,
                "bad_clip_retries_enabled": False,
                "bad_clip_retries_attempts": 5,
                "tts_mode": "external",
                "parallel_workers": 6,
            }

            with _FreshCloneServer(
                source_ref="HEAD",
                include_worktree_changes=True,
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

                    def _filtered_console_errors() -> list[str]:
                        transient_tokens = (
                            "net::ERR_CONNECTION_RESET",
                            "net::ERR_ABORTED",
                        )
                        return [
                            entry
                            for entry in console_errors
                            if not any(token in str(entry) for token in transient_tokens)
                        ]

                    page.on("console", _on_console)
                    page.on("pageerror", _on_page_error)
                    page.on("response", _on_response)

                    try:
                        page.goto(app_server.base_url, wait_until="domcontentloaded", timeout=10000)
                        _wait_for_bootstrap_ready(page)
                        _wait_for_setup_tab_ready(page, expect_active=True)

                        initial = _read_setup_settings_snapshot(page)
                        assert initial == {
                            "llm_base_url": "",
                            "llm_api_key": "",
                            "llm_model_name": "",
                            "llm_workers": 1,
                            "tts_provider": "qwen3",
                            "script_max_length": 250,
                            "bad_clip_retries_enabled": True,
                            "bad_clip_retries_attempts": 3,
                            "tts_mode": "local",
                            "parallel_workers": 4,
                        }, f"Unexpected fresh-clone Settings defaults: {json.dumps(initial, ensure_ascii=False, indent=2)}"

                        _apply_setup_settings_snapshot(page, edited_values)
                        page.wait_for_timeout(3000)

                        console_errors.clear()
                        page_errors.clear()
                        warnings.clear()
                        http_failures.clear()
                        app_server.restart()

                        page.goto(app_server.base_url, wait_until="domcontentloaded", timeout=10000)
                        _wait_for_bootstrap_ready(page)
                        _open_setup_tab(page)

                        persisted = _read_setup_settings_snapshot(page)
                        assert persisted == edited_values, (
                            "Edited Settings did not survive app restart.\n"
                            f"Expected:\n{json.dumps(edited_values, ensure_ascii=False, indent=2)}\n"
                            f"Actual:\n{json.dumps(persisted, ensure_ascii=False, indent=2)}"
                        )

                        final_console_errors = _filtered_console_errors()
                        assert not final_console_errors, _report_console(final_console_errors, page_errors, warnings)
                        assert not page_errors, _report_console(console_errors, page_errors, warnings)
                    except Exception as exc:
                        raise AssertionError(
                            f"Fresh-clone Settings persistence flow failed: {exc}\n"
                            f"HTTP failures:\n{chr(10).join(http_failures[-20:]) or 'none'}\n"
                            f"Fresh clone server log tail:\n{_tail_file(app_server.log_path, max_chars=5000)}\n"
                            f"{_report_console(console_errors, page_errors, warnings)}"
                        ) from exc
                    finally:
                        context.close()
                        browser.close()
