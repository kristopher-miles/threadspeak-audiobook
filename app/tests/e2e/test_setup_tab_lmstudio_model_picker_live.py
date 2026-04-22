"""Live Playwright coverage for LM Studio model picker behavior on Setup tab."""

import pytest

from ._stage_ui_helpers import *  # noqa: F401,F403

pytestmark = pytest.mark.lmstudio_live_e2e


LMSTUDIO_MODELS_ENDPOINT = f"{LMSTUDIO_DEFAULT_ORIGIN}/api/v1/models"


def _probe_lmstudio_or_skip() -> None:
    try:
        response = requests.get(LMSTUDIO_MODELS_ENDPOINT, timeout=4)
    except Exception as exc:
        pytest.skip(f"LM Studio is not reachable at {LMSTUDIO_MODELS_ENDPOINT}: {exc}")

    if int(response.status_code) != 200:
        pytest.skip(
            f"LM Studio probe failed at {LMSTUDIO_MODELS_ENDPOINT} with status {int(response.status_code)}"
        )


def test_setup_tab_lmstudio_model_picker_popup_live_roundtrip():
    with _exclusive_run_lock("setup_tab_lmstudio_model_picker_popup_live_roundtrip"):
        _probe_lmstudio_or_skip()

        config_patch = {
            "llm": {
                "base_url": "",
                "api_key": "local",
                "model_name": "",
                "llm_workers": 1,
            },
            "generation": {
                "legacy_mode": False,
            },
        }

        console_errors: list[str] = []
        page_errors: list[str] = []
        warnings: list[str] = []

        with _IsolatedServer(config_patch=config_patch) as app_server:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                _install_error_toast_guard(page)

                def _on_console(message):
                    text = str(message.text or "").strip()
                    kind = str(message.type or "").strip().lower()
                    if kind == "error":
                        console_errors.append(text)
                    elif kind == "warning":
                        warnings.append(text)

                def _on_page_error(err):
                    page_errors.append(str(err))

                page.on("console", _on_console)
                page.on("pageerror", _on_page_error)

                try:
                    page.goto(app_server.base_url, wait_until="domcontentloaded", timeout=10000)
                    _wait_for_bootstrap_ready(page)
                    _open_setup_tab(page)

                    llm_url = page.locator("#llm-url")
                    llm_key = page.locator("#llm-key")
                    llm_model = page.locator("#llm-model")

                    llm_url.fill(LMSTUDIO_DEFAULT_V1_BASE_URL)
                    llm_url.blur()
                    llm_key.fill("local")
                    llm_key.blur()
                    llm_model.fill("")

                    with page.expect_response(
                        lambda response: (
                            response.url.endswith("/api/config/lmstudio/list_models")
                            and response.request.method == "POST"
                        ),
                        timeout=20000,
                    ):
                        llm_model.click()

                    popup_snapshot = _wait_for_activity(
                        "Waiting for LM Studio model picker popup",
                        lambda: page.evaluate(
                            """() => {
                                const popup = document.querySelector('#llm-model-suggestion-popup');
                                const options = popup
                                    ? Array.from(popup.querySelectorAll('button[data-model-key]'))
                                    : [];
                                return {
                                    visible: !!popup && getComputedStyle(popup).display !== 'none',
                                    option_count: options.length,
                                    first_key: String(options[0]?.getAttribute('data-model-key') || ''),
                                };
                            }"""
                        ),
                        lambda snapshot: bool(snapshot.get("visible") and int(snapshot.get("option_count") or 0) > 0),
                        inactivity_seconds=20.0,
                        max_total_seconds=90.0,
                    )

                    first_key = str(popup_snapshot.get("first_key") or "").strip()
                    assert first_key, f"LM Studio popup did not expose a first model key: {popup_snapshot}"

                    page.locator("#llm-model-suggestion-popup button").first.click()
                    _wait_for_activity(
                        "Waiting for model field update after selecting LM Studio option",
                        lambda: {
                            "value": llm_model.input_value(),
                            "popup_visible": page.evaluate(
                                """() => {
                                    const popup = document.querySelector('#llm-model-suggestion-popup');
                                    return !!popup && getComputedStyle(popup).display !== 'none';
                                }"""
                            ),
                        },
                        lambda snapshot: str(snapshot.get("value") or "").strip() == first_key,
                        inactivity_seconds=10.0,
                        max_total_seconds=30.0,
                    )

                    selected_value = llm_model.input_value().strip()
                    assert selected_value == first_key, (
                        f"Expected selected model '{first_key}', got '{selected_value}'"
                    )

                    llm_model.click()
                    second_popup_snapshot = _wait_for_activity(
                        "Waiting for LM Studio popup to reopen on second click",
                        lambda: page.evaluate(
                            """() => {
                                const popup = document.querySelector('#llm-model-suggestion-popup');
                                const options = popup
                                    ? Array.from(popup.querySelectorAll('button[data-model-key]'))
                                    : [];
                                return {
                                    visible: !!popup && getComputedStyle(popup).display !== 'none',
                                    option_count: options.length,
                                };
                            }"""
                        ),
                        lambda snapshot: bool(snapshot.get("visible") and int(snapshot.get("option_count") or 0) > 0),
                        inactivity_seconds=20.0,
                        max_total_seconds=60.0,
                    )
                    assert int(second_popup_snapshot.get("option_count") or 0) > 0

                    assert not console_errors, _report_console(console_errors, page_errors, warnings)
                    assert not page_errors, _report_console(console_errors, page_errors, warnings)
                finally:
                    context.close()
                    browser.close()
