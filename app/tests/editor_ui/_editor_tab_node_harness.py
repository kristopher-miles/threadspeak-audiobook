import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
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
                enqueueTrackedChunkUpdate: (chunk) => enqueueTrackedChunkUpdate(chunk),
                flushQueuedTrackedChunkUpdates: () => flushQueuedTrackedChunkUpdates(),
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
                    replace(oldName, newName) {{
                        const hadOld = values.delete(oldName);
                        if (hadOld) values.add(newName);
                        return hadOld;
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
                    replaceWith(node) {{
                        if (!this.__container) return;
                        this.__container.audio = node;
                        if (node) node.__container = this.__container;
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
                    if (selector === '.chunk-audio-slot audio') return audioContainer.audio;
                    if (selector === 'audio') return audioContainer.audio;
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
                class FakeURLSearchParams {{
                    constructor() {{
                        this._pairs = [];
                    }}
                    set(key, value) {{
                        this._pairs = this._pairs.filter(([existing]) => existing !== key);
                        this._pairs.push([key, value]);
                    }}
                    toString() {{
                        return this._pairs.map(([key, value]) => `${{encodeURIComponent(key)}}=${{encodeURIComponent(value)}}`).join('&');
                    }}
                }}
                const context = {{
                    console,
                    Promise,
                    setTimeout: (fn, _ms) => global.setTimeout(fn, 0),
                    clearTimeout: (id) => global.clearTimeout(id),
                    setInterval: (fn, _ms) => global.setInterval(fn, 0),
                    clearInterval: (id) => global.clearInterval(id),
                    requestAnimationFrame: (fn) => global.setTimeout(fn, 0),
                    window: null,
                    URLSearchParams: FakeURLSearchParams,
                    EventSource: function EventSource(url) {{
                        this.url = url;
                        this.addEventListener = () => {{}};
                        this.close = () => {{}};
                        this.onerror = null;
                    }},
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
                                    element.querySelectorAll = (selector) => {{
                                        if (selector === 'tr[data-id]') return Array.from(rows.values());
                                        return [];
                                    }};
                                    Object.defineProperty(element, 'innerHTML', {{
                                        get() {{
                                            return element.__html || '';
                                        }},
                                        set(value) {{
                                            element.__html = String(value || '');
                                            rows.clear();
                                            const html = element.__html;
                                            const rowPattern = /<tr\\b([^>]*)data-id="([^"]+)"([^>]*)>([\\s\\S]*?)<\\/tr>/g;
                                            let match = null;
                                            while ((match = rowPattern.exec(html)) !== null) {{
                                                const attrs = `${{match[1] || ''}} ${{match[3] || ''}}`;
                                                const classMatch = attrs.match(/class="([^"]+)"/);
                                                const classNames = classMatch ? classMatch[1] : '';
                                                const row = createChunkRow({{
                                                    id: match[2],
                                                    status: classNames.includes('status-done')
                                                        ? 'done'
                                                        : (classNames.includes('status-generating') ? 'generating' : 'pending'),
                                                }});
                                                const audioMatch = match[4].match(/<audio\\b[^>]*>/);
                                                if (audioMatch) {{
                                                    row.__audioContainer.audio = createAudioNode(audioMatch[0]);
                                                    row.__audioContainer.audio.__container = row.__audioContainer;
                                                }}
                                                rows.set(String(match[2]), row);
                                            }}
                                            element.children = Array.from(rows.values());
                                        }},
                                    }});
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
