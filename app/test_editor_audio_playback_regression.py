import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
EDITOR_TAB_JS = APP_DIR / "static" / "js" / "legacy" / "11_editor_tab.js"


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class EditorAudioPlaybackRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("node") is None:
            raise unittest.SkipTest("node is required for editor playback regression tests")

        cls.port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        env = os.environ.copy()
        env["PINOKIO_SHARE_LOCAL"] = "false"
        env["PINOKIO_SHARE_LOCAL_PORT"] = str(cls.port)
        env["PYTHONUNBUFFERED"] = "1"

        cls.server = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=str(APP_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        deadline = time.time() + 45.0
        while time.time() < deadline:
            if cls.server.poll() is not None:
                output = ""
                if cls.server.stdout:
                    try:
                        output = cls.server.stdout.read() or ""
                    except Exception:
                        output = ""
                raise RuntimeError(
                    f"editor playback regression server exited early with code {cls.server.returncode}\n"
                    f"{output[-4000:]}"
                )
            try:
                response = requests.get(f"{cls.base_url}/", timeout=1.5)
                if response.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(0.3)

        raise RuntimeError(f"timed out waiting for editor playback regression server at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        server = getattr(cls, "server", None)
        if server is None:
            return
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass

    def _run_playback_probe(self):
        script = f"""
const fs = require('fs');
const vm = require('vm');

const baseUrl = process.argv[2];
const sourcePath = process.argv[3];
const source = fs.readFileSync(sourcePath, 'utf8');

function makeClassList() {{
  const values = new Set();
  return {{
    add(...names) {{ names.forEach((name) => values.add(name)); }},
    remove(...names) {{ names.forEach((name) => values.delete(name)); }},
    contains(name) {{ return values.has(name); }},
    toggle(name, force) {{
      if (force === true) {{ values.add(name); return true; }}
      if (force === false) {{ values.delete(name); return false; }}
      if (values.has(name)) {{ values.delete(name); return false; }}
      values.add(name);
      return true;
    }},
  }};
}}

function createElement() {{
  const listeners = new Map();
  return {{
    dataset: {{}},
    style: {{}},
    classList: makeClassList(),
    innerHTML: '',
    title: '',
    paused: true,
    ended: false,
    currentTime: 0,
    error: null,
    addEventListener(type, cb) {{
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(cb);
    }},
    dispatch(type) {{
      for (const cb of listeners.get(type) || []) cb.call(this);
    }},
    setAttribute(name, value) {{ this[name] = value; }},
    getAttribute(name) {{ return this[name] || null; }},
    load() {{}},
    remove() {{}},
    replaceChildren() {{}},
    appendChild(child) {{ return child; }},
    querySelector() {{ return null; }},
    querySelectorAll() {{ return []; }},
    closest() {{ return null; }},
  }};
}}

function createPreviewAudio() {{
  const audio = createElement();
  audio.preload = 'auto';
  audio.pause = function() {{
    this.paused = true;
    this.dispatch('pause');
  }};
  audio.play = async function() {{
    const resolvedSrc = new URL(this.src, baseUrl).toString();
    const response = await fetch(resolvedSrc, {{ redirect: 'follow' }});
    const contentType = String(response.headers.get('content-type') || '');
    if (!response.ok || !contentType.startsWith('audio/')) {{
      const err = new Error('NotSupportedError: Failed to load because no supported source was found.');
      err.name = 'NotSupportedError';
      this.error = {{ code: 4, message: err.message }};
      this.dispatch('error');
      throw err;
    }}
    await response.arrayBuffer();
    this.error = null;
    this.paused = false;
    this.ended = false;
    this.currentTime = 0.25;
    this.dispatch('play');
  }};
  return audio;
}}

(async () => {{
  const context = {{
    console,
    Promise,
    setTimeout: global.setTimeout,
    clearTimeout: global.clearTimeout,
    setInterval: global.setInterval,
    clearInterval: global.clearInterval,
    requestAnimationFrame: (fn) => global.setTimeout(fn, 0),
    URLSearchParams,
    fetch,
    window: null,
    EventSource: function() {{ this.addEventListener = () => {{}}; this.close = () => {{}}; }},
    API: {{
      get: async (url) => {{
        const response = await fetch(`${{baseUrl}}${{url}}`);
        if (!response.ok) throw new Error(`GET ${{url}} -> ${{response.status}}`);
        return await response.json();
      }},
      post: async () => ({{ status: 'ok' }}),
    }},
    showToast() {{}},
    showConfirm: async () => true,
    pollLogs: async () => null,
    renderAudioMergeProgress() {{}},
    renderProofreadTaskStatus() {{}},
    renderEditorProgressBar() {{}},
    scheduleDictionaryCountsRefresh() {{}},
    refreshDictionaryCounts() {{}},
    loadPipelineStepIcons: async () => null,
    updateNewModeWorkflowButtons() {{}},
    markTaskActionRequested() {{}},
    formatBytes: (value) => String(value),
    formatDuration: (value) => String(value),
    formatNumber: (value) => String(value),
    Audio: function Audio() {{ return createPreviewAudio(); }},
    Option: function Option(label, value) {{ return {{ label, value }}; }},
    localStorage: {{ getItem() {{ return null; }}, setItem() {{}}, removeItem() {{}} }},
    document: {{
      body: createElement(),
      querySelector() {{ return null; }},
      querySelectorAll(selector) {{
        if (selector === 'audio') return context.__slot ? [context.__slot.audio] : [];
        return [];
      }},
      getElementById() {{ return createElement(); }},
      createElement() {{ return createElement(); }},
    }},
  }};
  context.window = context;
  context.window.addEventListener = () => {{}};

  vm.createContext(context);
  vm.runInContext(source, context);

  const chunks = await context.API.get('/api/chunks/view');
  const chunk = chunks.find((entry) => entry && entry.audio_path);
  if (!chunk) {{
    throw new Error('No editor chunk with audio_path found in the live project.');
  }}
  const chunkRef = context.getChunkRef(chunk);

  const slot = {{
    button: createElement(),
    audio: createElement(),
    querySelector(selector) {{
      if (selector === '.chunk-audio-toggle') return this.button;
      if (selector === 'audio.chunk-audio' || selector === 'audio') return this.audio;
      return null;
    }},
    replaceChildren() {{
      this.audio = null;
      this.button = null;
    }},
  }};
  slot.button.classList = makeClassList();
  slot.button.closest = () => slot;
  slot.audio.closest = () => slot;
  slot.audio.dataset = {{
    id: String(chunkRef),
    audioPath: String(chunk.audio_path || ''),
    audioFingerprint: String(context.getChunkAudioFingerprint(chunk) || ''),
    audioRetryCount: '0',
  }};
  slot.audio.src = context.buildChunkAudioSrc(chunk, Date.now().toString());
  slot.audio.currentSrc = slot.audio.src;
  slot.audio.load = function() {{}};
  context.__slot = slot;

  await context.toggleChunkAudio(slot.button);
  const preview = context.window._editorPreviewAudio;
  console.log(JSON.stringify({{
    chunkRef,
    audioPath: chunk.audio_path,
    hiddenSrc: slot.audio ? slot.audio.src : null,
    previewSrc: preview ? (preview.currentSrc || preview.src) : null,
    previewPaused: preview ? preview.paused : null,
    previewCurrentTime: preview ? preview.currentTime : null,
    previewError: preview && preview.error ? preview.error : null,
    buttonTitle: slot.button ? slot.button.title : null,
    buttonActive: slot.button ? slot.button.classList.contains('active') : false,
  }}));
}})().catch((error) => {{
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}});
"""

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script)
            script_path = handle.name

        try:
            result = subprocess.run(
                ["node", script_path, self.base_url, str(EDITOR_TAB_JS)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

        if result.returncode != 0:
            raise AssertionError(
                "editor playback probe failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, f"probe produced no output\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return json.loads(lines[-1])

    def test_editor_compact_player_starts_playback_for_live_project_audio(self):
        probe = self._run_playback_probe()

        self.assertIsNone(
            probe["previewError"],
            f"editor preview audio reported an error for {probe['audioPath']}: {probe['previewError']}",
        )
        self.assertFalse(
            probe["previewPaused"],
            f"editor preview audio stayed paused for {probe['audioPath']} (src={probe['previewSrc']})",
        )
        self.assertGreater(
            float(probe["previewCurrentTime"] or 0.0),
            0.0,
            f"editor preview audio never advanced for {probe['audioPath']} (src={probe['previewSrc']})",
        )
        self.assertTrue(
            probe["buttonActive"],
            f"compact audio button never entered the playing state for {probe['audioPath']}",
        )


if __name__ == "__main__":
    unittest.main()
