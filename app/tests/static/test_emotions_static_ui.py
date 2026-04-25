from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_emotions_tab_is_wired_below_projects():
    index_html = (ROOT / "app/static/index.html").read_text(encoding="utf-8")
    projects_marker = 'data-tab="saved-scripts">Projects'
    emotions_marker = 'data-tab="emotions">Emotions'

    assert emotions_marker in index_html
    assert index_html.index(projects_marker) < index_html.index(emotions_marker)


def test_emotions_fragment_and_script_are_bootstrapped():
    main_js = (ROOT / "app/static/js/main.js").read_text(encoding="utf-8")

    assert "/static/fragments/emotions.html" in main_js
    assert "/static/js/legacy/17_emotions_tab.js" in main_js


def test_emotions_fragment_contains_expected_controls():
    fragment = (ROOT / "app/static/fragments/emotions.html").read_text(encoding="utf-8")

    assert 'id="emotions-tab"' in fragment
    assert 'id="emotions-text"' in fragment
    assert 'id="emotions-voice-card"' in fragment
    assert 'id="emotions-table-body"' in fragment
    assert "emotionsPlaySequence()" in fragment
    assert "emotionsRender(false)" in fragment
    assert "emotionsRender(true)" in fragment


def test_setup_fragment_exposes_voxcpm2_provider_option():
    fragment = (ROOT / "app/static/fragments/setup.html").read_text(encoding="utf-8")

    assert '<option value="qwen3">QWEN3</option>' in fragment
    assert '<option value="voxcpm2">VoxCPM2</option>' in fragment


def test_setup_fragment_exposes_voxcpm2_backend_controls():
    fragment = (ROOT / "app/static/fragments/setup.html").read_text(encoding="utf-8")

    assert 'id="voxcpm2-options"' in fragment
    assert 'id="voxcpm-optimize-group"' in fragment
    for field_id in (
        "voxcpm-model-id",
        "voxcpm-cfg-value",
        "voxcpm-inference-timesteps",
        "voxcpm-normalize",
        "voxcpm-denoise",
        "voxcpm-load-denoiser",
        "voxcpm-denoise-reference",
        "voxcpm-optimize",
    ):
        assert f'id="{field_id}"' in fragment
    assert 'id="voxcpm-cfg-value" value="1.6" min="1" max="3"' in fragment
    assert 'id="voxcpm-inference-timesteps" value="10" min="4" max="30"' in fragment


def test_setup_script_loads_saves_and_toggles_voxcpm2_controls():
    script = (ROOT / "app/static/js/legacy/03_setup_tab.js").read_text(encoding="utf-8")

    assert "function toggleVoxCPM2Options()" in script
    assert "function getTTSScriptMaxLengthDefault" in script
    assert "function clampNumber" in script
    assert "voxcpm2-options" in script
    assert "voxcpm-optimize-group" in script
    assert "isMacHostUI()" in script
    for config_key, field_id in (
        ("voxcpm_model_id", "voxcpm-model-id"),
        ("voxcpm_cfg_value", "voxcpm-cfg-value"),
        ("voxcpm_inference_timesteps", "voxcpm-inference-timesteps"),
        ("voxcpm_normalize", "voxcpm-normalize"),
        ("voxcpm_denoise", "voxcpm-denoise"),
        ("voxcpm_load_denoiser", "voxcpm-load-denoiser"),
        ("voxcpm_denoise_reference", "voxcpm-denoise-reference"),
        ("voxcpm_optimize", "voxcpm-optimize"),
    ):
        assert config_key in script
        assert field_id in script


def test_emotions_script_loads_and_renders_standalone_rows():
    script = (ROOT / "app/static/js/legacy/17_emotions_tab.js").read_text(encoding="utf-8")

    assert "/api/emotions" in script
    assert "window.loadEmotions" in script
    assert "function buildEmotionsRowHtml" in script
    assert "EMOTIONS_TEST_VOICE" in script
