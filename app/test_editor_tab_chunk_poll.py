import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDITOR_TAB_JS = ROOT / "app" / "static" / "js" / "legacy" / "11_editor_tab.js"


class EditorTabChunkPollTests(unittest.TestCase):
    maxDiff = None

    def _run_node_test(self, body: str):
        script = textwrap.dedent(
            f"""
            const assert = require('assert');
            const fs = require('fs');
            const vm = require('vm');

            const source = fs.readFileSync({str(EDITOR_TAB_JS)!r}, 'utf8') + `
            this.__editorTabTestHooks = {{
                getCachedChunks: () => cachedChunks,
                setCachedChunks: (value) => {{ cachedChunks = value; }},
                getTrackedChunkPollCount: () => activeChunkStatusPolls.size,
                setLoadChunks: (fn) => {{ loadChunks = fn; }},
                setUpdateProofreadRow: (fn) => {{ updateProofreadRow = fn; }},
                setSelectedEditorChapter: (value) => {{ selectedEditorChapter = value; }},
                getNarratorSelections: () => getNarratorSelections(),
            }};
            `;

            function createClassList() {{
                const values = new Set();
                return {{
                    contains(name) {{ return values.has(name); }},
                    add(...names) {{
                        names.forEach((name) => values.add(name));
                    }},
                    remove(...names) {{
                        names.forEach((name) => values.delete(name));
                    }},
                    toggle(name, force) {{
                        if (force === true) {{
                            values.add(name);
                            return true;
                        }}
                        if (force === false) {{
                            values.delete(name);
                            return false;
                        }}
                        if (values.has(name)) {{
                            values.delete(name);
                            return false;
                        }}
                        values.add(name);
                        return true;
                    }},
                }};
            }}

            function createGenericElement() {{
                return {{
                    value: '',
                    checked: true,
                    disabled: false,
                    innerHTML: '',
                    innerText: '',
                    textContent: '',
                    style: {{}},
                    dataset: {{}},
                    children: [],
                    className: '',
                    classList: createClassList(),
                    addEventListener() {{}},
                    removeEventListener() {{}},
                    appendChild(child) {{
                        this.children.push(child);
                        return child;
                    }},
                    removeChild(child) {{
                        this.children = this.children.filter(item => item !== child);
                    }},
                    replaceChildren(...children) {{
                        this.children = children;
                    }},
                    querySelector() {{ return null; }},
                    querySelectorAll() {{ return []; }},
                    closest() {{ return null; }},
                    click() {{}},
                    focus() {{}},
                    pause() {{}},
                    load() {{}},
                    play() {{ return Promise.resolve(); }},
                    scrollIntoView() {{}},
                    getBoundingClientRect() {{ return {{ top: 0 }}; }},
                    setAttribute(name, value) {{ this[name] = value; }},
                    getAttribute(name) {{ return this[name] || null; }},
                    remove() {{}},
                }};
            }}

            function createAudioNode(html) {{
                const srcMatch = html.match(/src="([^"]+)"/);
                const pathMatch = html.match(/data-audio-path="([^"]+)"/);
                const fingerprintMatch = html.match(/data-audio-fingerprint="([^"]+)"/);
                return {{
                    dataset: {{
                        audioPath: pathMatch ? pathMatch[1] : '',
                        audioFingerprint: fingerprintMatch ? fingerprintMatch[1] : '',
                        audioRetryCount: '0',
                    }},
                    src: srcMatch ? srcMatch[1] : '',
                    loadCalls: 0,
                    paused: true,
                    load() {{
                        this.loadCalls += 1;
                    }},
                    pause() {{
                        this.paused = true;
                    }},
                    getAttribute(name) {{
                        if (name === 'src') return this.src;
                        return null;
                    }},
                    setAttribute(name, value) {{
                        if (name === 'src') this.src = value;
                    }},
                    remove() {{
                        if (this.__container) this.__container.audio = null;
                    }},
                }};
            }}

            function createGenerateSlot(chunkRef) {{
                const slot = createGenericElement();
                const button = createGenericElement();
                button.className = 'btn btn-primary btn-sm chunk-generate-btn';
                button.innerHTML = '<i class="fas fa-play"></i> Gen';
                button.onclick = null;
                slot.button = button;
                slot.progress = null;
                slot.querySelector = (selector) => {{
                    if (selector === 'button') return slot.button;
                    if (selector === '.progress') return slot.progress;
                    return null;
                }};
                Object.defineProperty(slot, 'innerHTML', {{
                    get() {{
                        return '';
                    }},
                    set(value) {{
                        if (value.includes('chunk-generate-progress') || value.includes('progress-bar')) {{
                            slot.button = null;
                            slot.progress = createGenericElement();
                            slot.progress.className = 'progress chunk-generate-progress';
                        }} else if (value.includes('chunk-generate-btn') || value.includes('Gen')) {{
                            slot.progress = null;
                            const nextButton = createGenericElement();
                            nextButton.className = 'btn btn-primary btn-sm chunk-generate-btn';
                            nextButton.innerHTML = '<i class="fas fa-play"></i> Gen';
                            nextButton.onclick = null;
                            slot.button = nextButton;
                        }} else {{
                            slot.button = null;
                            slot.progress = null;
                        }}
                    }},
                }});
                return slot;
            }}

            function createAudioContainer(chunkRef) {{
                const container = createGenericElement();
                container.audio = null;
                container.querySelector = (selector) => {{
                    if (selector === 'audio') return container.audio;
                    return null;
                }};
                container.insertAdjacentHTML = (_position, html) => {{
                    if (html.includes('<audio')) {{
                        container.audio = createAudioNode(html);
                        container.audio.__container = container;
                    }}
                }};
                return container;
            }}

            function createChunkRow(chunk) {{
                const chunkRef = String(chunk.id);
                const row = createGenericElement();
                if (chunk.status === 'done') {{
                    row.classList.add('status-done');
                }} else if (chunk.status === 'generating') {{
                    row.classList.add('status-generating');
                }}

                const textArea = createGenericElement();
                textArea.value = chunk.text || '';
                textArea.dataset.editorField = 'text';
                textArea.className = 'chunk-text';

                const speakerInput = createGenericElement();
                speakerInput.value = chunk.speaker || '';
                speakerInput.dataset.editorField = 'speaker';

                const instructInput = createGenericElement();
                instructInput.value = chunk.instruct || '';
                instructInput.dataset.editorField = 'instruct';

                const generateSlot = createGenerateSlot(chunkRef);
                const audioContainer = createAudioContainer(chunkRef);

                row.dataset = {{ id: chunkRef }};
                row.querySelector = (selector) => {{
                    if (selector === '.chunk-text') return textArea;
                    if (selector === '.chunk-generate-slot') return generateSlot;
                    if (selector === '.chunk-audio-slot') return audioContainer;
                    return null;
                }};
                row.querySelectorAll = (selector) => {{
                    if (selector === '[data-editor-field]') {{
                        return [speakerInput, textArea, instructInput];
                    }}
                    return [];
                }};
                row.__generateSlot = generateSlot;
                row.__audioContainer = audioContainer;
                row.__textArea = textArea;
                row.__speakerInput = speakerInput;
                row.__instructInput = instructInput;
                return row;
            }}

            function createContext() {{
                const genericElements = new Map();
                const rows = new Map();
                const toasts = [];
                const localStorageStore = new Map();
                const context = {{
                    console,
                    Promise,
                    setTimeout: (fn, _ms) => global.setTimeout(fn, 0),
                    clearTimeout: (id) => global.clearTimeout(id),
                    setInterval: (fn, _ms) => global.setInterval(fn, 0),
                    clearInterval: (id) => global.clearInterval(id),
                    requestAnimationFrame: (fn) => global.setTimeout(fn, 0),
                    window: null,
                    API: {{
                        get: async () => [],
                        post: async () => ({{ status: 'ok' }}),
                    }},
                    showToast(message, level) {{
                        toasts.push({{ message, level }});
                    }},
                    showConfirm: async () => true,
                    pollLogs: async () => null,
                    renderAudioMergeProgress() {{}},
                    renderProofreadTaskStatus() {{}},
                    renderEditorProgressBar() {{}},
                    scheduleDictionaryCountsRefresh() {{}},
                    refreshDictionaryCounts() {{}},
                    stopOthers() {{}},
                    loadPipelineStepIcons: async () => null,
                    updateNewModeWorkflowButtons() {{}},
                    markTaskActionRequested() {{}},
                    formatBytes: (value) => String(value),
                    formatDuration: (value) => String(value),
                    formatNumber: (value) => String(value),
                    Audio: function Audio() {{
                        return createGenericElement();
                    }},
                    Option: function Option(label, value) {{
                        return {{ label, value }};
                    }},
                    localStorage: {{
                        getItem(key) {{
                            return localStorageStore.has(key) ? localStorageStore.get(key) : null;
                        }},
                        setItem(key, value) {{
                            localStorageStore.set(key, String(value));
                        }},
                        removeItem(key) {{
                            localStorageStore.delete(key);
                        }},
                    }},
                    document: {{
                        body: createGenericElement(),
                        querySelector(selector) {{
                            const rowMatch = selector.match(/^tr\\[data-id="(.+)"\\]$/);
                            if (rowMatch) return rows.get(rowMatch[1]) || null;
                            if (selector === '[data-tab="audio"]') return createGenericElement();
                            if (selector === '[data-tab="editor"]') return createGenericElement();
                            return createGenericElement();
                        }},
                        querySelectorAll(selector) {{
                            if (selector === 'audio') return [];
                            if (selector === '#proofread-table-body tr[data-proofread-id]') return [];
                            return [];
                        }},
                        getElementById(id) {{
                            if (!genericElements.has(id)) {{
                                const element = createGenericElement();
                                if (id === 'editor-chapter-only') element.checked = true;
                                if (id === 'legacy-mode-toggle') element.checked = true;
                                if (id === 'chunks-table-body') {{
                                    element.children = [];
                                    element.querySelectorAll = () => Array.from(rows.values());
                                }}
                                if (id === 'editor-tab' || id === 'proofread-tab') {{
                                    element.style.display = 'block';
                                }}
                                if (id === 'proofread-table-body') {{
                                    element.querySelectorAll = () => [];
                                }}
                                genericElements.set(id, element);
                            }}
                            return genericElements.get(id);
                        }},
                        createElement(tagName) {{
                            const element = createGenericElement();
                            element.tagName = String(tagName || '').toUpperCase();
                            return element;
                        }},
                    }},
                }};
                context.window = context;
                context.window.addEventListener = () => {{}};
                context.__rows = rows;
                context.__toasts = toasts;
                context.__createChunkRow = createChunkRow;
                return context;
            }}

            async function flushTicks(count = 6) {{
                for (let index = 0; index < count; index += 1) {{
                    await new Promise((resolve) => setTimeout(resolve, 0));
                }}
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
                "Node harness failed\\n"
                f"stdout:\\n{result.stdout}\\n"
                f"stderr:\\n{result.stderr}"
            )

    def test_generate_chunk_polls_until_done_without_broad_reload(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 1,
                    speaker: 'Narrator',
                    text: 'A finished clip should update in place.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('1', row);

                let loadChunksCalls = 0;
                const savePayloads = [];
                const chunkStates = [
                    { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/clip.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/1') {
                        savePayloads.push(payload);
                        return { id: 1, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/1/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/1') {
                        return chunkStates.shift() || { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/clip.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {{ loadChunksCalls += 1; }});

                await context.window.generateChunk('1');
                await flushTicks();

                assert.strictEqual(savePayloads.length, 1);
                assert.ok(row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
                assert.strictEqual(loadChunksCalls, 0, 'single-clip tracking should not trigger broad chunk reloads');
                assert.ok(row.__generateSlot.button, 'generate button should be restored');
                assert.strictEqual(row.__generateSlot.progress, null, 'progress bar should be removed');
                assert.ok(row.__audioContainer.audio, 'audio player should be inserted after completion');
                assert.strictEqual(row.__audioContainer.audio !== null, true);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_generate_chunk_prefers_targeted_poll_results_over_stale_broad_state(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 6,
                    speaker: 'Narrator',
                    text: 'The focused chunk poll should win even if broad state is stale.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('6', row);

                let loadChunksCalls = 0;
                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/6') {
                        return { id: 6, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/6/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/6') {
                        return { id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/overlay.mp3', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {
                    loadChunksCalls += 1;
                    context.__editorTabTestHooks.setCachedChunks([{ id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                });

                await context.window.generateChunk('6');
                await flushTicks();

                assert.strictEqual(loadChunksCalls, 0, 'focused tracking should not fall back to broad reloads');
                assert.ok(row.classList.contains('status-done'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].audio_path, 'voicelines/overlay.mp3');
                assert.ok(row.__audioContainer.audio, 'audio player should be rendered from the targeted poll result');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_generate_chunk_polls_until_error_and_restores_button(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 2,
                    speaker: 'Narrator',
                    text: 'A failed clip should recover the button.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('2', row);

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/2') {
                        return { id: 2, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/2/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/2') {
                        return { id: 2, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'error', audio_path: null, audio_validation: { error: 'tts failed' } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 2, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);

                await context.window.generateChunk('2');
                await flushTicks();

                assert.ok(!row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.ok(row.__generateSlot.button, 'button should return after error');
                assert.strictEqual(row.__generateSlot.progress, null);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'error');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_generate_chunk_keeps_generating_state_during_initial_pending_race(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 4,
                    speaker: 'Narrator',
                    text: 'Pending should not wipe the optimistic generating state.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('4', row);

                let getCalls = 0;
                const chunkStates = [
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null },
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/4') {
                        return { id: 4, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/4/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/4') {
                        getCalls += 1;
                        return chunkStates.shift() || { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);

                const generationPromise = context.window.generateChunk('4');
                await flushTicks(1);
                assert.ok(row.classList.contains('status-generating'), 'optimistic generating state should survive the initial pending poll response');
                assert.ok(row.__generateSlot.progress, 'progress bar should remain visible while backend catches up');

                await generationPromise;
                await flushTicks();

                assert.ok(getCalls >= 3, 'poller should continue past the stale pending response');
                assert.ok(row.classList.contains('status-done'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_regenerate_proofread_chunk_uses_targeted_tracker(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const proofreadUpdates = [];
                let loadChunksCalls = 0;
                const chunkStates = [
                    { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'done', audio_path: 'voicelines/proofread.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url) => {
                    if (url === '/api/chunks/3/regenerate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/3') {
                        return chunkStates.shift() || { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'done', audio_path: 'voicelines/proofread.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {{ loadChunksCalls += 1; }});
                context.__editorTabTestHooks.setUpdateProofreadRow((chunk) => {{
                    proofreadUpdates.push(chunk.status);
                    return true;
                }});

                await context.window.regenerateProofreadChunk('3');
                await flushTicks();

                assert.deepStrictEqual(proofreadUpdates.slice(-2), ['generating', 'done']);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
                assert.strictEqual(loadChunksCalls, 0, 'proofread regeneration should not trigger broad chunk reloads');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_regenerate_proofread_chunk_keeps_generating_state_during_initial_pending_race(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const proofreadUpdates = [];
                const chunkStates = [
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'pending', audio_path: null, audio_validation: null },
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'done', audio_path: 'voicelines/proofread-race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url) => {
                    if (url === '/api/chunks/5/regenerate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/5') {
                        return chunkStates.shift() || { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'done', audio_path: 'voicelines/proofread-race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setUpdateProofreadRow((chunk) => {
                    proofreadUpdates.push(chunk.status);
                    return true;
                });

                const regenerationPromise = context.window.regenerateProofreadChunk('5');
                await flushTicks(1);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'generating');

                await regenerationPromise;
                await flushTicks();

                assert.deepStrictEqual(proofreadUpdates.slice(-2), ['generating', 'done']);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_update_chunk_row_removes_stale_audio_after_invalidation(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 7,
                    speaker: 'Narrator',
                    text: 'Invalidated narrator audio should disappear.',
                    instruct: '',
                    status: 'done',
                    audio_path: 'voicelines/stale.mp3',
                    audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                });
                row.__audioContainer.audio = createAudioNode('<audio src="/voicelines/stale.mp3?t=old" data-audio-path="voicelines/stale.mp3" data-audio-fingerprint="old"></audio>');
                row.__audioContainer.audio.__container = row.__audioContainer;
                context.__rows.set('7', row);

                vm.createContext(context);
                vm.runInContext(source, context);

                const changed = context.updateChunkRow({
                    id: 7,
                    speaker: 'Narrator',
                    text: 'Invalidated narrator audio should disappear.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });

                assert.strictEqual(changed, true);
                assert.ok(!row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.strictEqual(row.__audioContainer.audio, null);
                assert.ok(row.__generateSlot.button, 'generate button should be visible again');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_handle_chunk_audio_error_retries_once_with_fresh_src(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const audio = createAudioNode('<audio src="/voicelines/clip.mp3?t=old" data-audio-path="voicelines/clip.mp3" data-audio-fingerprint="fp"></audio>');
                const originalSrc = audio.src;

                context.handleChunkAudioError(audio);
                const retriedSrc = audio.src;
                context.handleChunkAudioError(audio);

                assert.notStrictEqual(retriedSrc, originalSrc, 'retry should replace the src with a fresh cache token');
                assert.strictEqual(audio.loadCalls, 1, 'audio reload should only happen once');
                assert.strictEqual(audio.dataset.audioRetryCount, '1');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_audio_queue_poll_preserves_existing_audio_element_while_generation_runs(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const initialChunk = {
                    id: 8,
                    speaker: 'Narrator',
                    text: 'Existing audio should remain usable during active generation.',
                    instruct: '',
                    status: 'done',
                    audio_path: 'voicelines/existing.mp3',
                    audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                };
                const row = context.__createChunkRow(initialChunk);
                row.__audioContainer.audio = createAudioNode('<audio src="/voicelines/existing.mp3?t=stable" data-audio-path="voicelines/existing.mp3" data-audio-fingerprint="voicelines/existing.mp3|10|1|done"></audio>');
                row.__audioContainer.audio.__container = row.__audioContainer;
                const preservedAudio = row.__audioContainer.audio;
                context.__rows.set('8', row);

                context.API.get = async (url) => {
                    if (url === '/api/status/audio') {
                        return {
                            running: true,
                            queue: [],
                            current_job: { total_chunks: 2 },
                            metrics: { processed_clips: 1 },
                        };
                    }
                    if (url === '/api/chunks/view') {
                        return [initialChunk];
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([initialChunk]);

                await context.pollAudioQueueOnce();
                await flushTicks();

                assert.strictEqual(row.__audioContainer.audio, preservedAudio, 'running generation should not replace an unchanged audio element');
                assert.strictEqual(row.__audioContainer.audio.src, '/voicelines/existing.mp3?t=stable');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_narrator_selection_is_saved_before_reload_and_not_reverted(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.__editorTabTestHooks.setSelectedEditorChapter('Chapter 1');
                context.localStorage.setItem('threadspeak-narrator-selection', JSON.stringify({ 'Chapter 1': 'Old Voice' }));

                const select = context.document.getElementById('editor-narrator-select');
                select.value = 'New Voice';

                context.__editorTabTestHooks.setCachedChunks([
                    { id: 1, speaker: 'NARRATOR', text: 'Narrator line', chapter: 'Chapter 1', audio_path: 'voicelines/existing.mp3' }
                ]);

                let selectionSeenDuringReload = null;
                context.__editorTabTestHooks.setLoadChunks(async () => {
                    selectionSeenDuringReload = context.__editorTabTestHooks.getNarratorSelections()['Chapter 1'] || null;
                });

                context.API.post = async (url, payload) => {
                    assert.strictEqual(url, '/api/narrator_overrides');
                    assert.strictEqual(payload.chapter, 'Chapter 1');
                    assert.strictEqual(payload.voice, 'New Voice');
                    assert.strictEqual(payload.invalidate_audio, true);
                    return { status: 'saved' };
                };

                await context.window.onNarratorSelectorChange();

                const finalSelections = context.__editorTabTestHooks.getNarratorSelections();
                assert.strictEqual(selectionSeenDuringReload, 'New Voice', 'loadChunks should observe the new selection, not the stale one');
                assert.strictEqual(finalSelections['Chapter 1'], 'New Voice');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_sync_narrator_selections_from_backend_replaces_stale_local_entries(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.localStorage.setItem('threadspeak-narrator-selection', JSON.stringify({
                    'Chapter 1': 'Old Voice',
                    'Chapter 2': 'Another Voice',
                }));

                context.API.get = async (url) => {
                    assert.strictEqual(url, '/api/narrator_overrides');
                    return { 'Chapter 1': 'Fresh Voice' };
                };

                await context.syncNarratorSelectionsFromBackend();

                const finalSelections = context.__editorTabTestHooks.getNarratorSelections();
                assert.strictEqual(JSON.stringify(finalSelections), JSON.stringify({ 'Chapter 1': 'Fresh Voice' }));
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )


if __name__ == "__main__":
    unittest.main()
