        // --- API Helpers ---
        const API = {
            _formatErrorDetail: (detail, fallback) => {
                if (typeof detail === 'string' && detail.trim()) return detail;
                if (Array.isArray(detail)) {
                    const messages = detail.map(item => {
                        if (typeof item === 'string') return item;
                        if (!item || typeof item !== 'object') return '';
                        const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
                        const msg = item.msg || item.message || JSON.stringify(item);
                        return loc ? `${loc}: ${msg}` : msg;
                    }).filter(Boolean);
                    if (messages.length) return messages.join('; ');
                }
                if (detail && typeof detail === 'object') {
                    return detail.message || JSON.stringify(detail);
                }
                return fallback;
            },
            _handleError: async (res) => {
                if (res.ok) return;
                try {
                    const body = await res.json();
                    throw new Error(API._formatErrorDetail(body.detail, res.statusText));
                } catch (e) {
                    if (e.message) throw e;
                    throw new Error(res.statusText);
                }
            },
            get: async (url) => {
                const res = await fetch(url);
                await API._handleError(res);
                return res.json();
            },
            post: async (url, data) => {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                await API._handleError(res);
                return res.json();
            },
            upload: async (file) => {
                const formData = new FormData();
                formData.append('file', file);
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                await API._handleError(res);
                return res.json();
            }
        };

        async function syncEditorChunksOnNavigation() {
            try {
                const result = await API.post('/api/chunks/sync_from_script_if_stale', {});
                if (result?.synced) {
                    showToast(`Editor refreshed from the latest script (${result.chunk_count || 0} chunks).`, 'info', 3500);
                }
                return result;
            } catch (e) {
                console.error('Failed to sync editor chunks from script:', e);
                return null;
            }
        }

        const THEME_STORAGE_KEY = 'alexandria-theme';
        let generationModeLocked = false;

        function applyTheme(theme) {
            const resolvedTheme = theme === 'light' ? 'light' : 'dark';
            document.body.dataset.theme = resolvedTheme;
            const toggle = document.getElementById('dark-mode-toggle');
            if (toggle) {
                toggle.checked = resolvedTheme === 'dark';
            }
        }

        function getStoredTheme() {
            const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
            return storedTheme === 'light' ? 'light' : 'dark';
        }

        window.toggleDarkMode = (enabled) => {
            const theme = enabled ? 'dark' : 'light';
            localStorage.setItem(THEME_STORAGE_KEY, theme);
            applyTheme(theme);
            persistNavbarPreferences();
        };

        function autoResizeTextarea(textarea) {
            if (!textarea) return;
            if (textarea.offsetParent === null) return;
            textarea.style.height = 'auto';
            const minHeight = 112; // ~7rem baseline so prompts never collapse to a single line
            textarea.style.height = `${Math.max(textarea.scrollHeight, minHeight)}px`;
        }

        function initPromptTextareaAutosize() {
            document.querySelectorAll('textarea.prompt-textarea').forEach((textarea) => {
                if (textarea.dataset.autosizeBound === '1') return;
                textarea.dataset.autosizeBound = '1';
                textarea.addEventListener('input', () => autoResizeTextarea(textarea));
            });
            requestAnimationFrame(() => refreshPromptTextareaHeights());
        }

        function refreshPromptTextareaHeights() {
            document.querySelectorAll('textarea.prompt-textarea').forEach(autoResizeTextarea);
        }

        function bindAudioProcessingDependency(processVoicesToggleId, generateAudioToggleId) {
            const processVoicesToggle = document.getElementById(processVoicesToggleId);
            const generateAudioToggle = document.getElementById(generateAudioToggleId);
            if (!processVoicesToggle || !generateAudioToggle) return;

            const enforce = (source) => {
                if (source === 'generate' && generateAudioToggle.checked) {
                    processVoicesToggle.checked = true;
                }
                if (source === 'voices' && !processVoicesToggle.checked && generateAudioToggle.checked) {
                    generateAudioToggle.checked = false;
                }
            };

            generateAudioToggle.addEventListener('change', () => enforce('generate'));
            processVoicesToggle.addEventListener('change', () => enforce('voices'));
            enforce('init');
        }

        async function persistNavbarPreferences() {
            const legacyToggle = document.getElementById('legacy-mode-toggle');
            const darkToggle = document.getElementById('dark-mode-toggle');
            if (!legacyToggle || !darkToggle) return;
            try {
                await API.post('/api/config/preferences', {
                    legacy_mode: !!legacyToggle.checked,
                    dark_mode: !!darkToggle.checked,
                });
            } catch (e) {
                console.warn('Failed to persist navbar preferences', e);
            }
        }

        function applyGenerationModeLock(locked) {
            generationModeLocked = !!locked;
            const toggle = document.getElementById('legacy-mode-toggle');
            const wrapper = document.getElementById('legacy-mode-toggle-wrapper');
            if (!toggle) return;
            toggle.disabled = generationModeLocked;
            if (wrapper) {
                wrapper.classList.toggle('opacity-50', generationModeLocked);
                wrapper.title = generationModeLocked
                    ? 'Reset Project to change generation modes.'
                    : '';
            }
        }

        async function lockGenerationMode(trigger) {
            if (generationModeLocked) return;
            try {
                const result = await API.post('/api/generation_mode_lock', {
                    locked: true,
                    trigger: trigger || null,
                });
                applyGenerationModeLock(!!result.locked);
            } catch (e) {
                console.warn('Failed to lock generation mode', e);
            }
        }
