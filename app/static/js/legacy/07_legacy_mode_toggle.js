        // --- Legacy Mode Toggle ---
        (function() {
            const toggle = document.getElementById('legacy-mode-toggle');
            const wrapper = document.getElementById('legacy-mode-toggle-wrapper');
            const legacyControls = document.getElementById('legacy-mode-controls');
            const newControls = document.getElementById('new-mode-controls');
            const setupLegacyPromptFields = document.getElementById('setup-legacy-prompt-fields');
            const setupNonLegacyPromptFields = document.getElementById('setup-nonlegacy-prompt-fields');
            function applyLegacyMode(isLegacy) {
                legacyControls.style.display = isLegacy ? '' : 'none';
                newControls.style.display = isLegacy ? 'none' : '';
                if (setupLegacyPromptFields) {
                    setupLegacyPromptFields.style.display = isLegacy ? '' : 'none';
                }
                if (setupNonLegacyPromptFields) {
                    setupNonLegacyPromptFields.style.display = isLegacy ? 'none' : '';
                }
                const legacyButtons = document.getElementById('legacy-only-buttons');
                if (legacyButtons) legacyButtons.style.display = isLegacy ? 'flex' : 'none';
                requestAnimationFrame(() => refreshPromptTextareaHeights());
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

