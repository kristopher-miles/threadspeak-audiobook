import { loadFragments, loadClassicScript } from './core/fragment-loader.js';

const TAB_FRAGMENTS = [
  '/static/fragments/setup.html',
  '/static/fragments/script.html',
  '/static/fragments/voices.html',
  '/static/fragments/dictionary.html',
  '/static/fragments/saved-scripts.html',
  '/static/fragments/designer.html',
  '/static/fragments/training.html',
  '/static/fragments/dataset-builder.html',
  '/static/fragments/editor.html',
  '/static/fragments/proofread.html',
  '/static/fragments/audio.html',
];

const LEGACY_SCRIPTS = [
  '/static/js/legacy/00_utils.js',
  '/static/js/legacy/01_navigation.js',
  '/static/js/legacy/02_api_helpers.js',
  '/static/js/legacy/03_setup_tab.js',
  '/static/js/legacy/04_script_tab.js',
  '/static/js/legacy/05_advanced_controls.js',
  '/static/js/legacy/07_legacy_mode_toggle.js',
  '/static/js/legacy/08_new_mode.js',
  '/static/js/legacy/09_voices_tab.js',
  '/static/js/legacy/10_dictionary_tab.js',
  '/static/js/legacy/11_editor_tab.js',
  '/static/js/legacy/12_audacity_export.js',
  '/static/js/legacy/13_polling_logic.js',
  '/static/js/legacy/14_voice_designer.js',
  '/static/js/legacy/15_clone_voice_handlers.js',
  '/static/js/legacy/16_lora_training.js',
];

async function bootstrap() {
  window.__THREADSPEAK_BOOTSTRAP_DONE = false;
  window.__THREADSPEAK_BOOTSTRAP_ERROR = null;
  window.__THREADSPEAK_BOOTSTRAP_STEP = 'init';
  window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY = Date.now();
  const root = document.getElementById('tab-fragments-root');
  window.__THREADSPEAK_BOOTSTRAP_STEP = 'fragments';
  await loadFragments({ root, fragments: TAB_FRAGMENTS });
  window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY = Date.now();
  // Ensure updated legacy scripts are fetched after backend/UI patches.
  const scriptVersion = Date.now().toString();
  window.__THREADSPEAK_BOOTSTRAP_STEP = 'legacy-scripts';
  for (const script of LEGACY_SCRIPTS) {
    await loadClassicScript(`${script}?v=${scriptVersion}`);
    window.__THREADSPEAK_BOOTSTRAP_STEP = `loaded:${script}`;
    window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY = Date.now();
  }
  window.__THREADSPEAK_BOOTSTRAP_DONE = true;
  window.__THREADSPEAK_BOOTSTRAP_STEP = 'done';
  window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY = Date.now();
}

bootstrap().catch((error) => {
  window.__THREADSPEAK_BOOTSTRAP_DONE = false;
  window.__THREADSPEAK_BOOTSTRAP_ERROR = String(error?.stack || error?.message || error || 'unknown bootstrap error');
  window.__THREADSPEAK_BOOTSTRAP_STEP = 'error';
  window.__THREADSPEAK_BOOTSTRAP_LAST_ACTIVITY = Date.now();
  console.error('UI bootstrap failed', error);
  const root = document.getElementById('tab-fragments-root');
  if (root) {
    root.innerHTML = '<div class="alert alert-danger">Failed to initialize UI modules. Check browser console for details.</div>';
  }
});
