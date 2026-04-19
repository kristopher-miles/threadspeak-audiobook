"""Opt-in full UX E2E coverage against a fresh git clone of origin/main."""

from ._stage_ui_helpers import *  # noqa: F401,F403


@pytest.mark.fresh_clone_e2e
def test_fresh_clone_full_ui_flow_to_merged_audiobook_and_reset_project():
    """
    Top-level rule:
    once browser interactions begin, this flow uses UI navigation/actions only.
    """
    with _exclusive_run_lock("fresh_clone_full_ui_flow_to_merged_audiobook"):
        fixtures_dir = os.path.join(SOURCE_APP_DIR, "test_fixtures", "e2e_sim")
        script_fixture_path = os.path.join(fixtures_dir, "lmstudio_generate_script_test_book.json")
        voice_fixture_path = os.path.join(fixtures_dir, "lmstudio_voice_profiles_test_book.json")
        qwen_fixture_path = os.path.join(fixtures_dir, "qwen_local_full_e2e_test_book.json")
        proofread_fixture_path = os.path.join(fixtures_dir, "proofread_text_test_book.json")
        book_path = os.path.join(SOURCE_APP_DIR, "test_fixtures", "files", "test_book.epub")

        assert os.path.exists(script_fixture_path), f"Missing fixture: {script_fixture_path}"
        assert os.path.exists(voice_fixture_path), f"Missing fixture: {voice_fixture_path}"
        assert os.path.exists(qwen_fixture_path), f"Missing fixture: {qwen_fixture_path}"
        assert os.path.exists(proofread_fixture_path), f"Missing fixture: {proofread_fixture_path}"
        assert os.path.exists(book_path), f"Missing book fixture: {book_path}"

        script_payload = _read_json(script_fixture_path)
        voice_payload = _read_json(voice_fixture_path)
        script_model = str(((script_payload.get("metadata") or {}).get("model_name") or "").strip())
        voice_model = str(((voice_payload.get("metadata") or {}).get("model_name") or "").strip())
        assert script_model, "Script fixture metadata.model_name is required."
        assert voice_model, "Voice fixture metadata.model_name is required."

        console_errors: list[str] = []
        page_errors: list[str] = []
        warnings: list[str] = []
        http_failures: list[str] = []
        expected_speakers = {
            str(item).strip()
            for item in (((voice_payload.get("metadata") or {}).get("speakers")) or [])
            if str(item).strip()
        }
        assert expected_speakers, "Voice fixture metadata.speakers must include at least one speaker."

        with _report_directory("threadspeak_fresh_clone_export_report_") as report_root:
            script_lm_trace = os.path.join(report_root, "lm-script-trace.jsonl")
            voice_lm_trace = os.path.join(report_root, "lm-voice-trace.jsonl")
            qwen_report = os.path.join(report_root, "qwen-report.json")
            qwen_trace = os.path.join(report_root, "qwen-trace.jsonl")
            proofread_report = os.path.join(report_root, "proofread-report.json")
            proofread_trace = os.path.join(report_root, "proofread-trace.jsonl")
            env_overrides = {
                "THREADSPEAK_E2E_SIM_ENABLED": "1",
                "THREADSPEAK_E2E_QWEN_FIXTURE": os.path.abspath(qwen_fixture_path),
                "THREADSPEAK_E2E_QWEN_REPORT_PATH": qwen_report,
                "THREADSPEAK_E2E_QWEN_TRACE_PATH": qwen_trace,
                "THREADSPEAK_E2E_PROOFREAD_FIXTURE": os.path.abspath(proofread_fixture_path),
                "THREADSPEAK_E2E_PROOFREAD_REPORT_PATH": proofread_report,
                    "THREADSPEAK_E2E_PROOFREAD_TRACE_PATH": proofread_trace,
                    "THREADSPEAK_E2E_PROOFREAD_FALLBACK": "chunk_text",
                    "THREADSPEAK_E2E_SIM_STRICT": "1",
                    MODEL_DOWNLOAD_DISABLE_ENV: "1",
                }

            with LMStudioSimServer(
                script_fixture_path,
                trace_path=script_lm_trace,
                trace_label="fresh-clone-script-lm",
            ) as script_server:
                with LMStudioSimServer(
                    voice_fixture_path,
                    trace_path=voice_lm_trace,
                    trace_label="fresh-clone-voice-lm",
                ) as voice_server:
                    with _FreshCloneServer(
                        env_overrides=env_overrides,
                        bootstrap_config_values={
                            "tts": {
                                "local_backend": "qwen",
                            }
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
                                _run_stage1_to_voices_tab(
                                    page=page,
                                    app_base_url=app_server.base_url,
                                    book_path=book_path,
                                    script_llm_base_url=f"{script_server.base_url}/v1",
                                    script_llm_model_name=script_model,
                                    tts_mode="local",
                                    tts_parallel_workers=1,
                                )
                                _run_stage2_to_stage4_proofread_flow(
                                    page=page,
                                    app_base_url=app_server.base_url,
                                    voice_server_base_url=f"{voice_server.base_url}/v1",
                                    voice_model_name=voice_model,
                                    expected_speakers=expected_speakers,
                                )

                                _wait_for_nav_unlocked(page, '.nav-link[data-tab="audio"]', "Export tab")
                                page.locator('.nav-link[data-tab="audio"]').click()
                                _wait_for_activity(
                                    "Waiting for Export tab",
                                    lambda: page.evaluate(
                                        """() => ({
                                            visible: !!document.querySelector('#audio-tab') && getComputedStyle(document.querySelector('#audio-tab')).display !== 'none',
                                            has_merge_btn: !!document.querySelector('#btn-merge'),
                                            has_logs: !!document.querySelector('#audio-logs'),
                                            has_chapter_select: !!document.querySelector('#export-chapter-select')
                                        })"""
                                    ),
                                    lambda snapshot: bool(
                                        snapshot.get("visible")
                                        and snapshot.get("has_merge_btn")
                                        and snapshot.get("has_logs")
                                        and snapshot.get("has_chapter_select")
                                    ),
                                )

                                page.locator("#export-chapter-select").select_option("")
                                _wait_for_activity(
                                    "Waiting for full-project export scope selection",
                                    lambda: {
                                        "chapter_value": page.evaluate(
                                            "() => String(document.querySelector('#export-chapter-select')?.value || '')"
                                        )
                                    },
                                    lambda snapshot: str(snapshot.get("chapter_value") or "") == "",
                                )

                                with page.expect_response(
                                    lambda response: (
                                        response.url.endswith("/api/merge")
                                        and response.request.method == "POST"
                                        and response.status == 200
                                    ),
                                    timeout=10000,
                                ):
                                    page.locator("#btn-merge").click()
                                    _confirm_modal_if_present(page, timeout_ms=4000)

                                _wait_for_audio_merge_completion(app_server.base_url)

                                _wait_for_activity(
                                    "Waiting for merged export UI readiness",
                                    lambda: page.evaluate(
                                        """() => ({
                                            player_visible: !!document.querySelector('#audio-player-container')
                                                && getComputedStyle(document.querySelector('#audio-player-container')).display !== 'none',
                                            audio_src: String(document.querySelector('#main-audio')?.getAttribute('src') || ''),
                                            download_href: String(document.querySelector('#download-link')?.getAttribute('href') || ''),
                                        })"""
                                    ),
                                    lambda snapshot: bool(
                                        snapshot.get("player_visible")
                                        and ("/api/audiobook" in str(snapshot.get("audio_src") or "") or "/api/audiobook" in str(snapshot.get("download_href") or ""))
                                    ),
                                )

                                mp3_response = requests.get(f"{app_server.base_url}/api/audiobook", timeout=30)
                                _assert_status(mp3_response, 200, "download merged audiobook")
                                content_type = str(mp3_response.headers.get("content-type") or "").lower()
                                assert "audio/mpeg" in content_type, f"Expected audio/mpeg content-type, got: {content_type}"
                                assert len(mp3_response.content) > 1024, "Merged audiobook download is unexpectedly small."

                                assert app_server.layout is not None, "Missing fresh clone runtime layout."
                                fresh_clone_mp3 = app_server.layout.audiobook_path
                                assert os.path.exists(fresh_clone_mp3), f"Merged output not found: {fresh_clone_mp3}"
                                fresh_clone_size = os.path.getsize(fresh_clone_mp3)
                                assert fresh_clone_size > 1024, f"Merged output is too small: {fresh_clone_size} bytes"
                                assert _looks_like_mp3(fresh_clone_mp3), f"Merged output is not recognized as MP3: {fresh_clone_mp3}"
                                duration_seconds = _audio_duration_seconds(fresh_clone_mp3)
                                assert duration_seconds > 240.0, (
                                    f"Merged audiobook duration must be > 4 minutes, got {duration_seconds:.2f}s"
                                )

                                page.locator('.nav-link[data-tab="script"]').click()
                                _wait_for_script_tab_ready(page)
                                complete_states = _wait_for_script_step_states(
                                    page,
                                    {
                                        "process_paragraphs": "complete",
                                        "assign_dialogue": "complete",
                                        "extract_temperament": "complete",
                                        "create_script": "complete",
                                    },
                                )
                                assert all(value == "complete" for value in complete_states.values()), (
                                    f"Expected complete script step states before reset, got: {complete_states}"
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

                                _wait_for_nav_locked(page, '.nav-link[data-tab="voices"]', "Voices tab")
                                _wait_for_nav_locked(page, '.nav-link[data-tab="editor"]', "Editor tab")
                                _wait_for_nav_locked(page, '.nav-link[data-tab="proofread"]', "Proofread tab")
                                _wait_for_nav_locked(page, '.nav-link[data-tab="audio"]', "Export tab")

                                post_reset_states = _wait_for_script_step_states(
                                    page,
                                    {
                                        "process_paragraphs": "not_started",
                                        "assign_dialogue": "not_started",
                                        "extract_temperament": "not_started",
                                        "create_script": "not_started",
                                    },
                                )
                                assert all(value != "complete" for value in post_reset_states.values()), (
                                    f"Expected reset script step states to drop completion, got: {post_reset_states}"
                                )

                                _assert_runtime_audio_artifacts_removed(app_server.layout)

                                assert not console_errors, _report_console(console_errors, page_errors, warnings)
                                assert not page_errors, _report_console(console_errors, page_errors, warnings)
                            except Exception as exc:
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
                                raise AssertionError(
                                    f"Fresh-clone full UI flow failed: {exc}\n"
                                    f"Script logs tail:\n{script_logs[-2000:]}\n"
                                    f"Proofread logs tail:\n{proofread_logs[-2000:]}\n"
                                    f"Audio logs tail:\n{audio_logs[-2000:]}\n"
                                    f"HTTP failures:\n{chr(10).join(http_failures[-20:]) or 'none'}\n"
                                    f"LM script trace tail ({script_lm_trace}):\n{_tail_file(script_lm_trace)}\n"
                                    f"LM voice trace tail ({voice_lm_trace}):\n{_tail_file(voice_lm_trace)}\n"
                                    f"Qwen trace tail ({qwen_trace}):\n{_tail_file(qwen_trace)}\n"
                                    f"Qwen report tail ({qwen_report}):\n{_tail_file(qwen_report)}\n"
                                    f"Proofread trace tail ({proofread_trace}):\n{_tail_file(proofread_trace)}\n"
                                    f"Proofread report tail ({proofread_report}):\n{_tail_file(proofread_report)}\n"
                                    f"{_report_console(console_errors, page_errors, warnings)}"
                                ) from exc
                            finally:
                                context.close()
                                browser.close()

                script_server.assert_all_consumed()
                voice_server.assert_all_consumed()

            if os.path.exists(qwen_report):
                qwen_summary = _read_json(qwen_report)
                qwen_pending = dict(qwen_summary.get("pending") or {})
                allowed_pending = {"generate_voice_clone", "generate_voice_design"}
                disallowed_pending = {
                    key: value
                    for key, value in qwen_pending.items()
                    if str(key) not in allowed_pending
                }
                assert not disallowed_pending, f"Unexpected pending Qwen interactions: {disallowed_pending}"

            if os.path.exists(proofread_report):
                proofread_summary = _read_json(proofread_report)
                proofread_pending = list(proofread_summary.get("pending") or [])
                assert not proofread_pending, f"Proofread simulator still has pending entries: {proofread_pending}"
