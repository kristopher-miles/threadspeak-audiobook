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
            }};
            `;

            function createClassList() {{
                return {{
                    contains() {{ return false; }},
                    add() {{}},
                    remove() {{}},
                    toggle() {{ return false; }},
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
                    }},
                    src: srcMatch ? srcMatch[1] : '',
                    loadCalls: 0,
                    load() {{
                        this.loadCalls += 1;
                    }},
                    getAttribute(name) {{
                        if (name === 'src') return this.src;
                        return null;
                    }},
                }};
            }}

            function createNoAudioNode(container) {{
                return {{
                    className: 'text-muted small',
                    set outerHTML(html) {{
                        container.noAudio = null;
                        container.audio = createAudioNode(html);
                    }},
                }};
            }}

            function createActionContainer(chunkRef) {{
                const container = createGenericElement();
                const button = createGenericElement();
                button.className = 'btn btn-sm btn-primary';
                button.innerHTML = '<i class="fas fa-play"></i> Gen';
                button.onclick = null;
                container.button = button;
                container.progress = null;
                container.audio = null;
                container.noAudio = createNoAudioNode(container);
                container.querySelector = (selector) => {{
                    if (selector === 'button') return container.button;
                    if (selector === '.progress') return container.progress;
                    if (selector === 'audio') return container.audio;
                    if (selector === '.text-muted.small') return container.noAudio;
                    return null;
                }};
                container.replaceChild = (nextNode, previousNode) => {{
                    if (previousNode === container.button) {{
                        container.button = null;
                        container.progress = nextNode;
                        return previousNode;
                    }}
                    if (previousNode === container.progress) {{
                        container.progress = null;
                        container.button = nextNode;
                        return previousNode;
                    }}
                    return previousNode;
                }};
                container.insertAdjacentHTML = (_position, html) => {{
                    if (html.includes('No audio')) {{
                        container.noAudio = createNoAudioNode(container);
                    }}
                }};
                return container;
            }}

            function createChunkRow(chunk) {{
                const chunkRef = String(chunk.id);
                const row = createGenericElement();
                const statusBadge = createGenericElement();
                statusBadge.className = 'badge bg-secondary';
                statusBadge.innerText = chunk.status || 'pending';

                const statusCell = createGenericElement();
                statusCell.querySelector = (selector) => selector === '.badge' ? statusBadge : null;

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

                const actionContainer = createActionContainer(chunkRef);

                row.dataset = {{ id: chunkRef }};
                row.querySelector = (selector) => {{
                    if (selector === '.chunk-text') return textArea;
                    if (selector === '.badge') return statusBadge;
                    if (selector === '.chunk-status-cell') return statusCell;
                    if (selector === '.d-flex') return actionContainer;
                    return null;
                }};
                row.querySelectorAll = (selector) => {{
                    if (selector === '[data-editor-field]') {{
                        return [speakerInput, textArea, instructInput];
                    }}
                    return [];
                }};
                row.__statusBadge = statusBadge;
                row.__statusCell = statusCell;
                row.__actionContainer = actionContainer;
                row.__textArea = textArea;
                row.__speakerInput = speakerInput;
                row.__instructInput = instructInput;
                return row;
            }}

            function createContext() {{
                const genericElements = new Map();
                const rows = new Map();
                const toasts = [];
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
                assert.strictEqual(row.__statusBadge.innerText, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
                assert.strictEqual(loadChunksCalls, 0, 'single-clip tracking should not trigger broad chunk reloads');
                assert.ok(row.__actionContainer.button, 'generate button should be restored');
                assert.strictEqual(row.__actionContainer.progress, null, 'progress bar should be removed');
                assert.ok(row.__actionContainer.audio, 'audio player should be inserted after completion');
                assert.strictEqual(row.__actionContainer.noAudio, null, '"No audio" placeholder should be replaced');
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

                assert.strictEqual(row.__statusBadge.innerText, 'error');
                assert.ok(row.__actionContainer.button, 'button should return after error');
                assert.strictEqual(row.__actionContainer.progress, null);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'error');
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


if __name__ == "__main__":
    unittest.main()
