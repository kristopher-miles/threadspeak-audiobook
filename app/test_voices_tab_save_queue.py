import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOICES_TAB_JS = ROOT / "app" / "static" / "js" / "legacy" / "09_voices_tab.js"


class VoicesTabSaveQueueTests(unittest.TestCase):
    maxDiff = None

    def _run_node_test(self, body: str):
        script = textwrap.dedent(
            f"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const source = fs.readFileSync({str(VOICES_TAB_JS)!r}, 'utf8')
                + '\\nthis.__voicesTabTestHooks = {{ saveVoicesNow, suggestVoiceDescriptionsBulk, collectVoiceConfig, flushPendingVoiceSavesOnUnload }};';

            function createControl(initial = '', classNames = []) {{
                return {{
                    value: initial,
                    checked: false,
                    disabled: false,
                    textContent: '',
                    className: '',
                    classList: createClassList(classNames),
                    style: {{ display: 'block' }},
                    title: '',
                    addEventListener() {{}},
                    removeEventListener() {{}},
                    click() {{}},
                    pause() {{}},
                    load() {{}},
                    play() {{ return Promise.resolve(); }},
                    setAttribute() {{}},
                    removeAttribute() {{}},
                    remove() {{}},
                }};
            }}

            function createClassList(initial = []) {{
                const values = new Set(initial);
                return {{
                    contains(name) {{ return values.has(name); }},
                    add(name) {{ values.add(name); }},
                    remove(name) {{ values.delete(name); }},
                    toggle(name, force) {{
                        if (force === undefined) {{
                            if (values.has(name)) {{
                                values.delete(name);
                                return false;
                            }}
                            values.add(name);
                            return true;
                        }}
                        if (force) {{
                            values.add(name);
                            return true;
                        }}
                        values.delete(name);
                        return false;
                    }},
                }};
            }}

            function createVoiceCard(name, options = {{}}) {{
                const state = {{
                    type: options.type || 'design',
                    classes: options.classes || [],
                }};

                const controls = {{
                    aliasInput: createControl(options.alias || '', ['voice-alias-input']),
                    narrates: createControl(''),
                    customVoice: createControl('Aiden'),
                    customStyle: createControl(''),
                    builtinLoraSelect: createControl(''),
                    builtinLoraStyle: createControl(''),
                    loraAdapterSelect: createControl(''),
                    loraStyle: createControl(''),
                    cloneRefText: createControl(''),
                    cloneRefAudio: createControl(''),
                    designDescription: createControl(options.description || ''),
                    designSampleText: createControl(options.sampleText || ''),
                    designRefAudio: createControl(options.refAudio || ''),
                    designGeneratedRefText: createControl(options.generatedRefText || ''),
                    generateButton: createControl('Generate', ['design-generate-btn']),
                    suggestButton: createControl('Suggest', ['design-suggest-btn']),
                    playButton: createControl('Play', ['design-play-btn']),
                    downloadButton: createControl('Download', ['design-download-btn']),
                    customOpts: {{ style: {{ display: state.type === 'custom' ? 'block' : 'none' }}, classList: createClassList() }},
                    builtinLoraOpts: {{ style: {{ display: state.type === 'builtin_lora' ? 'block' : 'none' }}, classList: createClassList() }},
                    cloneOpts: {{ style: {{ display: state.type === 'clone' ? 'block' : 'none' }}, classList: createClassList() }},
                    loraOpts: {{ style: {{ display: state.type === 'lora' ? 'block' : 'none' }}, classList: createClassList() }},
                    designOpts: {{ style: {{ display: state.type === 'design' ? 'block' : 'none' }}, classList: createClassList() }},
                }};

                controls.narrates.checked = Boolean(options.narrates);

                const radios = {{
                    custom: createControl(''),
                    builtin_lora: createControl(''),
                    clone: createControl(''),
                    lora: createControl(''),
                    design: createControl(''),
                }};
                Object.entries(radios).forEach(([key, radio]) => {{
                    radio.value = key;
                    let checkedValue = key === state.type;
                    Object.defineProperty(radio, 'checked', {{
                        get() {{
                            return checkedValue;
                        }},
                        set(nextValue) {{
                            checkedValue = Boolean(nextValue);
                            if (checkedValue) {{
                                Object.entries(radios).forEach(([otherKey, otherRadio]) => {{
                                    if (otherKey !== key) {{
                                        otherRadio.checked = false;
                                    }}
                                }});
                                state.type = key;
                            }}
                        }},
                    }});
                }});

                const allControls = [
                    controls.aliasInput,
                    controls.narrates,
                    controls.customVoice,
                    controls.customStyle,
                    controls.builtinLoraSelect,
                    controls.builtinLoraStyle,
                    controls.loraAdapterSelect,
                    controls.loraStyle,
                    controls.cloneRefText,
                    controls.cloneRefAudio,
                    controls.designDescription,
                    controls.designSampleText,
                    controls.designRefAudio,
                    controls.designGeneratedRefText,
                    controls.generateButton,
                    controls.suggestButton,
                    controls.playButton,
                    controls.downloadButton,
                    controls.customOpts,
                    controls.builtinLoraOpts,
                    controls.cloneOpts,
                    controls.loraOpts,
                    controls.designOpts,
                    ...Object.values(radios),
                ];

                let card;
                const body = {{
                    querySelector(selector) {{
                        switch (selector) {{
                            case '.design-description': return controls.designDescription;
                            case '.design-sample-text': return controls.designSampleText;
                            case '.design-ref-audio': return controls.designRefAudio;
                            case '.design-generated-ref-text': return controls.designGeneratedRefText;
                            case '.design-generate-btn': return controls.generateButton;
                            case '.design-suggest-btn': return controls.suggestButton;
                            case '.design-play-btn': return controls.playButton;
                            case '.design-download-btn': return controls.downloadButton;
                            case '.voice-type:checked':
                                return Object.values(radios).find(radio => radio.checked) || null;
                            case '.voice-type[value="custom"]': return radios.custom;
                            case '.voice-type[value="builtin_lora"]': return radios.builtin_lora;
                            case '.voice-type[value="clone"]': return radios.clone;
                            case '.voice-type[value="lora"]': return radios.lora;
                            case '.voice-type[value="design"]': return radios.design;
                            case '.voice-select': return controls.customVoice;
                            case '.character-style': return controls.customStyle;
                            case '.builtin-lora-select': return controls.builtinLoraSelect;
                            case '.builtin-lora-style': return controls.builtinLoraStyle;
                            case '.lora-adapter-select': return controls.loraAdapterSelect;
                            case '.lora-character-style': return controls.loraStyle;
                            case '.ref-text': return controls.cloneRefText;
                            case '.ref-audio': return controls.cloneRefAudio;
                            case '.custom-opts': return controls.customOpts;
                            case '.builtin-lora-opts': return controls.builtinLoraOpts;
                            case '.clone-opts': return controls.cloneOpts;
                            case '.lora-opts': return controls.loraOpts;
                            case '.design-opts': return controls.designOpts;
                            default: return null;
                        }}
                    }},
                    querySelectorAll(selector) {{
                        if (selector === 'input, select, textarea, button') {{
                            return allControls;
                        }}
                        return [];
                    }},
                    closest(selector) {{
                        return selector === '.voice-card' ? card : null;
                    }},
                }};

                Object.values(radios).forEach((radio) => {{
                    radio.closest = (selector) => selector === '.card-body' ? body : card;
                }});
                controls.generateButton.closest = (selector) => selector === '.card-body' ? body : card;
                controls.suggestButton.closest = (selector) => selector === '.card-body' ? body : card;
                Object.values(controls).forEach((control) => {{
                    if (control && typeof control === 'object' && !control.closest) {{
                        control.closest = (selector) => {{
                            if (selector === '.voice-card') return card;
                            if (selector === '.card-body') return body;
                            return null;
                        }};
                    }}
                }});

                card = {{
                    dataset: {{
                        voice: name,
                        suggestedSample: options.suggestedSample || '',
                        lineCount: String(options.lineCount || 10),
                        paragraphCount: String(options.paragraphCount || 10),
                    }},
                    classList: createClassList(state.classes),
                    querySelector(selector) {{
                        if (selector === '.card-body') return body;
                        if (selector === '.voice-alias-input') return controls.aliasInput;
                        if (selector === '.voice-narrates') return controls.narrates;
                        return body.querySelector(selector);
                    }},
                    querySelectorAll(selector) {{
                        return body.querySelectorAll(selector);
                    }},
                }};

                return {{ card, body, controls, radios }};
            }}

            function createContext(cards, apiPost) {{
                const bulkButton = createControl('Generate outstanding');
                const saveStatus = {{ innerHTML: '' }};
                const listeners = {{}};
                const voicesList = {{
                    addEventListener(type, handler) {{
                        listeners[type] = handler;
                    }},
                    style: {{}},
                    getBoundingClientRect() {{ return {{ top: 0 }}; }},
                }};
                const narratorThresholdInput = createControl('10');
                narratorThresholdInput.addEventListener = () => {{}};
                const voicesTab = {{ style: {{ display: 'block' }} }};
                const clearButton = createControl('Clear');
                const toasts = [];
                const spinnerCalls = [];
                const fetchCalls = [];
                const context = {{
                    console,
                    Promise,
                    setTimeout: (fn, _ms) => global.setTimeout(fn, 0),
                    clearTimeout: (id) => global.clearTimeout(id),
                    requestAnimationFrame: (fn) => global.setTimeout(fn, 0),
                    window: null,
                    document: {{
                        hidden: false,
                        body: {{ appendChild() {{}}, removeChild() {{}} }},
                        querySelectorAll(selector) {{
                            if (selector === '.voice-card') return cards.map(item => item.card);
                            return [];
                        }},
                        getElementById(id) {{
                            if (id === 'voices-list') return voicesList;
                            if (id === 'voice-save-status') return saveStatus;
                            if (id === 'generate-outstanding-voices-btn') return bulkButton;
                            if (id === 'clear-outstanding-voices-btn') return clearButton;
                            if (id === 'narrator-threshold-input') return narratorThresholdInput;
                            if (id === 'voices-tab') return voicesTab;
                            return null;
                        }},
                        createElement() {{
                            return {{
                                href: '',
                                download: '',
                                click() {{}},
                                remove() {{}},
                            }};
                        }},
                    }},
                    API: {{
                        post: apiPost,
                        get: async () => [],
                    }},
                    fetch(url, options) {{
                        fetchCalls.push({{ url, options }});
                        return Promise.resolve({{ ok: true }});
                    }},
                    showToast(message, level) {{
                        toasts.push({{ message, level }});
                    }},
                    showConfirm: async () => true,
                    escapeHtml: (value) => String(value ?? ''),
                    playSharedPreviewAudio: async () => {{}},
                    isPreviewAbortError: () => false,
                    onDesignedVoiceSelect() {{}},
                    handleCloneVoiceUpload() {{}},
                    deleteCloneVoice() {{}},
                    syncEditorChunksOnNavigation: async () => null,
                    lockGenerationMode: async () => null,
                    Audio: function Audio() {{
                        return {{
                            preload: 'auto',
                            pause() {{}},
                            removeAttribute() {{}},
                            load() {{}},
                            play() {{ return Promise.resolve(); }},
                        }};
                    }},
                    AVAILABLE_VOICES: [],
                    _cloneVoicesCache: [],
                    _designedVoicesCache: [],
                    _loraModelsCache: [],
                    _narratingVoicesCache: null,
                }};

                context.window = context;
                context.window.addEventListener = () => {{}};
                context.window.setNavTaskSpinner = (name) => spinnerCalls.push(['set', name]);
                context.window.releaseNavTaskSpinner = (name) => spinnerCalls.push(['release', name]);
                context.__bulkButton = bulkButton;
                context.__saveStatus = saveStatus;
                context.__toasts = toasts;
                context.__spinnerCalls = spinnerCalls;
                context.__voicesListListeners = listeners;
                context.__fetchCalls = fetchCalls;
                return context;
            }}

            async function tick() {{
                await new Promise((resolve) => setTimeout(resolve, 0));
            }}

            async function waitFor(predicate, attempts = 20) {{
                for (let index = 0; index < attempts; index += 1) {{
                    if (predicate()) return;
                    await tick();
                }}
                throw new Error('Timed out waiting for condition');
            }}

            {textwrap.dedent(body)}
            """
        )

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script)
            script_path = handle.name
        try:
            result = subprocess.run(
                ["node", script_path],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)
        if result.returncode != 0:
            raise AssertionError(
                "Node harness failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def test_save_queues_second_request_while_first_is_in_flight(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', { type: 'design', description: '', sampleText: 'line' });
                let releaseFirstSave;
                const savePayloads = [];
                const context = createContext([voice], async (url, payload) => {
                    if (url !== '/api/voices/batch') {
                        return { status: 'ok' };
                    }
                    savePayloads.push(JSON.parse(JSON.stringify(payload)));
                    if (savePayloads.length === 1) {
                        await new Promise((resolve) => { releaseFirstSave = resolve; });
                    }
                    return { status: 'saved' };
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                const firstSave = context.__voicesTabTestHooks.saveVoicesNow();
                await tick();
                voice.controls.designDescription.value = 'Warm, composed contralto';
                const flushed = context.__voicesTabTestHooks.saveVoicesNow({
                    promptConfirmation: false,
                    retryOnNetworkFailure: true,
                });
                releaseFirstSave();
                await Promise.all([firstSave, flushed]);

                assert.strictEqual(savePayloads.length, 2, 'expected a follow-up save request');
                assert.strictEqual(savePayloads[0].config.Aerial.description, '');
                assert.strictEqual(savePayloads[1].config.Aerial.description, 'Warm, composed contralto');
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_generate_outstanding_voices_flushes_bulk_suggestions_before_generation(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', {
                    type: 'custom',
                    description: '',
                    sampleText: '',
                    suggestedSample: 'The clouds parted over the ridge.',
                });
                let releaseFirstSave;
                const savePayloads = [];
                const generatedSpeakers = [];
                const context = createContext([voice], async (url, payload) => {
                    if (url === '/api/voices/batch') {
                        savePayloads.push(JSON.parse(JSON.stringify(payload)));
                        if (savePayloads.length === 1) {
                            await new Promise((resolve) => { releaseFirstSave = resolve; });
                        }
                        return { status: 'saved' };
                    }
                    if (url === '/api/voices/suggest_descriptions_bulk') {
                        await waitFor(() => typeof releaseFirstSave === 'function');
                        return {
                            results: [{ speaker: 'Aerial', voice: 'Wind-bright mezzo with steady resolve' }],
                            failures: [],
                        };
                    }
                    if (url === '/api/voices/unload_bulk_generation') {
                        return { status: 'unloaded', unloaded: true };
                    }
                    throw new Error(`Unexpected API call: ${url}`);
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                context.window.generateVoiceDesignClone = async (btn) => {
                    const card = btn.closest('.card-body');
                    generatedSpeakers.push(card.closest('.voice-card').dataset.voice);
                    card.querySelector('.design-ref-audio').value = 'clone_voices/aerial.wav';
                };

                const runPromise = context.window.generateOutstandingVoices();
                await waitFor(() => typeof releaseFirstSave === 'function');
                releaseFirstSave();
                await runPromise;

                assert.ok(savePayloads.length >= 1, 'expected at least one save request');
                assert.strictEqual(savePayloads[savePayloads.length - 1].config.Aerial.type, 'design');
                assert.strictEqual(
                    savePayloads[savePayloads.length - 1].config.Aerial.description,
                    'Wind-bright mezzo with steady resolve'
                );
                assert.deepStrictEqual(generatedSpeakers, ['Aerial']);
                assert.ok(
                    context.__toasts.every((entry) => !entry.message.includes('missing description')),
                    'generation should not fail on missing descriptions'
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_generate_outstanding_voices_reloads_live_card_state_after_suggestions(self):
        self._run_node_test(
            """
            (async () => {
                const cards = [
                    createVoiceCard('Aerial', {
                        type: 'custom',
                        description: '',
                        sampleText: '',
                        suggestedSample: 'The clouds parted over the ridge.',
                    }),
                ];
                let saveCount = 0;
                const generatedDescriptions = [];
                const context = createContext(cards, async (url, payload) => {
                    if (url === '/api/voices/batch') {
                        saveCount += 1;
                        if (saveCount === 1) {
                            const replacement = createVoiceCard('Aerial', {
                                type: 'design',
                                description: 'Wind-bright mezzo with steady resolve',
                                sampleText: 'The clouds parted over the ridge.',
                                suggestedSample: 'The clouds parted over the ridge.',
                            });
                            cards[0] = replacement;
                        }
                        return { status: 'saved' };
                    }
                    if (url === '/api/voices/suggest_descriptions_bulk') {
                        return {
                            results: [{ speaker: 'Aerial', voice: 'Wind-bright mezzo with steady resolve' }],
                            failures: [],
                        };
                    }
                    if (url === '/api/voices/unload_bulk_generation') {
                        return { status: 'unloaded', unloaded: true };
                    }
                    throw new Error(`Unexpected API call: ${url}`);
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                context.window.generateVoiceDesignClone = async (btn) => {
                    const card = btn.closest('.card-body');
                    generatedDescriptions.push(card.querySelector('.design-description').value);
                    card.querySelector('.design-ref-audio').value = 'clone_voices/aerial.wav';
                };

                await context.window.generateOutstandingVoices();

                assert.deepStrictEqual(
                    generatedDescriptions,
                    ['Wind-bright mezzo with steady resolve'],
                    'expected bulk generation to use refreshed description state'
                );
                assert.ok(
                    context.__toasts.every((entry) => !entry.message.includes('missing description')),
                    'generation should not fail on missing descriptions after a live card refresh'
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_background_confirmation_required_keeps_unsaved_status(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', {
                    type: 'design',
                    description: 'Warm, composed contralto',
                    sampleText: 'line',
                    alias: '',
                });
                const context = createContext([voice], async (url, payload) => {
                    if (url !== '/api/voices/batch') {
                        throw new Error(`Unexpected API call: ${url}`);
                    }
                    return {
                        status: 'confirmation_required',
                        invalidated_clips: 3,
                    };
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                await context.__voicesTabTestHooks.saveVoicesNow({
                    promptConfirmation: false,
                    retryOnNetworkFailure: true,
                }).catch(() => {});

                assert.ok(
                    context.__saveStatus.innerHTML.includes('unsaved'),
                    `expected unsaved status, got: ${context.__saveStatus.innerHTML}`
                );
                assert.ok(
                    !context.__saveStatus.innerHTML.includes('save cancelled'),
                    `did not expect cancelled status, got: ${context.__saveStatus.innerHTML}`
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_generate_voice_prompts_for_invalidation_and_continues(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', {
                    type: 'design',
                    description: 'Warm, composed contralto',
                    sampleText: 'line',
                    alias: '',
                });
                const savePayloads = [];
                const context = createContext([voice], async (url, payload) => {
                    if (url === '/api/voices/batch') {
                        savePayloads.push(JSON.parse(JSON.stringify(payload)));
                        if (savePayloads.length === 1) {
                            return { status: 'confirmation_required', invalidated_clips: 2 };
                        }
                        return { status: 'saved', invalidated_clips: 2 };
                    }
                    if (url === '/api/voices/design_generate') {
                        return {
                            ref_audio: 'clone_voices/aerial.wav',
                            generated_ref_text: 'line',
                            ref_text: 'line',
                        };
                    }
                    if (url === '/api/clone_voices/list') {
                        return [];
                    }
                    throw new Error(`Unexpected API call: ${url}`);
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                const button = voice.controls.generateButton;
                await context.window.generateVoiceDesignClone(button);

                assert.ok(savePayloads.length >= 2, 'expected confirmable save flow');
                assert.strictEqual(savePayloads[0].confirm_invalidation, false);
                assert.strictEqual(savePayloads[1].confirm_invalidation, true);
                assert.strictEqual(voice.controls.designRefAudio.value, 'clone_voices/aerial.wav');
                assert.ok(
                    context.__toasts.some((entry) => entry.message.includes('Generated voice for Aerial.')),
                    'expected successful generation toast'
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_alias_change_uses_confirmable_save_flow(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', {
                    type: 'design',
                    description: 'Warm, composed contralto',
                    sampleText: 'line',
                    alias: '',
                });
                const savePayloads = [];
                const context = createContext([voice], async (url, payload) => {
                    if (url !== '/api/voices/batch') {
                        throw new Error(`Unexpected API call: ${url}`);
                    }
                    savePayloads.push(JSON.parse(JSON.stringify(payload)));
                    if (savePayloads.length === 1) {
                        return { status: 'confirmation_required', invalidated_clips: 4 };
                    }
                    return { status: 'saved', invalidated_clips: 4 };
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                voice.controls.aliasInput.value = 'Skylark';
                await context.__voicesListListeners.change({ target: voice.controls.aliasInput });
                await tick();
                await tick();

                assert.ok(savePayloads.length >= 2, 'expected confirmable alias save flow');
                assert.strictEqual(savePayloads[0].confirm_invalidation, false);
                assert.strictEqual(savePayloads[1].confirm_invalidation, true);
                assert.strictEqual(savePayloads[0].config.Aerial.alias, 'Skylark');
                assert.ok(
                    !context.__saveStatus.innerHTML.includes('unsaved'),
                    `did not expect unsaved status, got: ${context.__saveStatus.innerHTML}`
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_alias_input_flushes_confirmable_payload_on_unload(self):
        self._run_node_test(
            """
            (async () => {
                const voice = createVoiceCard('Aerial', {
                    type: 'design',
                    description: 'Warm, composed contralto',
                    sampleText: 'line',
                    alias: '',
                });
                const context = createContext([voice], async () => {
                    throw new Error('did not expect API.post during unload flush test');
                });

                vm.createContext(context);
                vm.runInContext(source, context);

                voice.controls.aliasInput.value = 'Skylark';
                await context.__voicesListListeners.input({ target: voice.controls.aliasInput });
                context.__voicesTabTestHooks.flushPendingVoiceSavesOnUnload();

                assert.strictEqual(context.__fetchCalls.length, 1, 'expected keepalive unload flush');
                const request = context.__fetchCalls[0];
                const payload = JSON.parse(request.options.body);
                assert.strictEqual(request.url, '/api/voices/batch');
                assert.strictEqual(payload.confirm_invalidation, true);
                assert.strictEqual(payload.config.Aerial.alias, 'Skylark');
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )


if __name__ == "__main__":
    unittest.main()
