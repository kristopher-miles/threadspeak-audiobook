import { loadFragments, loadClassicScript } from './core/fragment-loader.js';

const TAB_FRAGMENTS = [
  '/static/fragments/setup.html',
  '/static/fragments/script.html',
  '/static/fragments/voices.html',
  '/static/fragments/dictionary.html',
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
  '/static/js/legacy/06_auto_proofread.js',
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
  const root = document.getElementById('tab-fragments-root');
  await loadFragments({ root, fragments: TAB_FRAGMENTS });
  // Ensure updated legacy scripts are fetched after backend/UI patches.
  const scriptVersion = Date.now().toString();
  for (const script of LEGACY_SCRIPTS) {
    await loadClassicScript(`${script}?v=${scriptVersion}`);
  }
}

bootstrap().catch((error) => {
  console.error('UI bootstrap failed', error);
  const root = document.getElementById('tab-fragments-root');
  if (root) {
    root.innerHTML = '<div class="alert alert-danger">Failed to initialize UI modules. Check browser console for details.</div>';
  }
});
