        // --- Legacy Mode Toggle ---
        (function() {
            const toggle = document.getElementById('legacy-mode-toggle');
            const wrapper = document.getElementById('legacy-mode-toggle-wrapper');
            const legacyControls = document.getElementById('legacy-mode-controls');
            const newControls = document.getElementById('new-mode-controls');
            const legacyAdvanced = document.getElementById('legacy-mode-advanced');
            const newAdvanced = document.getElementById('new-mode-advanced');
            const setupLegacyPromptFields = document.getElementById('setup-legacy-prompt-fields');
            const setupNonLegacyPromptFields = document.getElementById('setup-nonlegacy-prompt-fields');
            function applyLegacyMode(isLegacy) {
                legacyControls.style.display = isLegacy ? '' : 'none';
                newControls.style.display = isLegacy ? 'none' : '';
                if (legacyAdvanced) legacyAdvanced.style.display = isLegacy ? '' : 'none';
                if (newAdvanced) newAdvanced.style.display = isLegacy ? 'none' : '';
                if (setupLegacyPromptFields) {
                    setupLegacyPromptFields.style.display = isLegacy ? '' : 'none';
                }
                if (setupNonLegacyPromptFields) {
                    setupNonLegacyPromptFields.style.display = isLegacy ? 'none' : '';
                }
                const legacyButtons = document.getElementById('legacy-only-buttons');
                if (legacyButtons) legacyButtons.style.display = isLegacy ? 'flex' : 'none';
                document.querySelectorAll('.nav-legacy-only').forEach(el => {
                    el.style.display = isLegacy ? '' : 'none';
                });
                requestAnimationFrame(() => refreshPromptTextareaHeights());
                if (window.updatePipelineTabLocks && typeof loadPipelineStepIcons === 'function') {
                    loadPipelineStepIcons().catch(() => {});
                }
            }
            toggle.addEventListener('change', () => {
                if (generationModeLocked) {
                    showToast('Reset Project to change generation modes.', 'warning');
                    return;
                }
                applyLegacyMode(toggle.checked);
                if (toggle.dataset.suspendPersist === '1') return;
                persistNavbarPreferences();
            });
            if (wrapper) {
                wrapper.addEventListener('click', (e) => {
                    if (!generationModeLocked) return;
                    showToast('Reset Project to change generation modes.', 'warning');
                });
            }
            applyLegacyMode(toggle.checked);
        })();

