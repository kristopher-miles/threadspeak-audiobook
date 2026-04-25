        // --- Setup Tab ---

        const TTS_PROVIDER_SCRIPT_MAX_LENGTH_DEFAULTS = {
            qwen3: 250,
            voxcpm2: 240
        };
        const VOXCPM2_CFG_VALUE_DEFAULT = 1.6;
        const VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT = 10;
        let _lastTTSProvider = 'qwen3';

        window.resetProject = async () => {
            const confirmed = await showConfirm('Reset the current project? This will delete the loaded source, generated script, detected characters, chunk audio, final audiobook files, and sanity/repair state. Global settings and prompt configuration will be kept.');
            if (!confirmed) return;
            try {
                await API.post('/api/reset_project', {});
                window.resetPipelineTabLocks?.();
                showToast('Project reset. Reloading...', 'success', 1500);
                setTimeout(() => window.location.reload(), 500);
            } catch (e) {
                showToast('Failed to reset project: ' + e.message, 'error');
            }
        };

        function toggleTTSMode() {
            const mode = document.getElementById('tts-mode').value;
            document.getElementById('tts-url-group').style.display = mode === 'external' ? '' : 'none';
            document.getElementById('tts-local-options').style.display = mode === 'local' ? '' : 'none';
            toggleVoxCPM2Options();
            applyBatchSettingsVisibility();
        }

        function toggleVoxCPM2Options() {
            const provider = document.getElementById('tts-provider')?.value || 'qwen3';
            const options = document.getElementById('voxcpm2-options');
            if (options) {
                options.style.display = provider === 'voxcpm2' ? '' : 'none';
            }
            const optimizeGroup = document.getElementById('voxcpm-optimize-group');
            const optimizeToggle = document.getElementById('voxcpm-optimize');
            if (optimizeGroup) {
                const hideOptimize = provider === 'voxcpm2' && isMacHostUI();
                optimizeGroup.style.display = hideOptimize ? 'none' : '';
                if (hideOptimize && optimizeToggle) {
                    optimizeToggle.checked = false;
                }
            }
        }

        function isMacHostUI() {
            const uaPlatform = navigator.userAgentData?.platform || '';
            const legacyPlatform = navigator.platform || '';
            const userAgent = navigator.userAgent || '';
            return /mac/i.test(uaPlatform) || /mac/i.test(legacyPlatform) || /mac os x/i.test(userAgent);
        }

        function applyBatchSettingsVisibility() {
            const hideBatchControls = isMacHostUI();
            const batchOnlyGroupIds = [
                'batch-seed-group',
                'batch-group-by-type-group',
                'sub-batch-enabled-group',
                'sub-batch-min-group',
                'sub-batch-ratio-group',
                'sub-batch-max-chars-group',
                'sub-batch-max-items-group',
            ];
            batchOnlyGroupIds.forEach((id) => {
                const el = document.getElementById(id);
                if (!el) return;
                el.style.display = hideBatchControls ? 'none' : '';
            });
        }

        function coerceConfigBool(value, fallback = false) {
            if (typeof value === 'boolean') return value;
            if (value == null) return !!fallback;
            if (typeof value === 'number') return value !== 0;
            if (typeof value === 'string') {
                const text = value.trim().toLowerCase();
                if (['1', 'true', 'yes', 'on'].includes(text)) return true;
                if (['0', 'false', 'no', 'off', ''].includes(text)) return false;
            }
            return !!value;
        }

        function parseIntOrDefault(inputId, fallback) {
            const value = parseInt(document.getElementById(inputId).value, 10);
            return Number.isNaN(value) ? fallback : value;
        }

        function parseFloatOrDefault(inputId, fallback) {
            const value = parseFloat(document.getElementById(inputId).value);
            return Number.isNaN(value) ? fallback : value;
        }

        function clampNumber(value, min, max, fallback) {
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) return fallback;
            return Math.min(max, Math.max(min, parsed));
        }

        function getTTSScriptMaxLengthDefault(provider) {
            const normalized = String(provider || 'qwen3').trim().toLowerCase();
            return TTS_PROVIDER_SCRIPT_MAX_LENGTH_DEFAULTS[normalized] || TTS_PROVIDER_SCRIPT_MAX_LENGTH_DEFAULTS.qwen3;
        }

        function applyProviderScriptMaxLengthDefault(provider) {
            const input = document.getElementById('script-max-length');
            if (!input) return;
            const nextProvider = String(provider || 'qwen3').trim().toLowerCase() || 'qwen3';
            const previousDefault = getTTSScriptMaxLengthDefault(_lastTTSProvider);
            const nextDefault = getTTSScriptMaxLengthDefault(nextProvider);
            const current = parseInt(input.value, 10);
            if (!Number.isInteger(current) || current === previousDefault) {
                input.value = nextDefault;
            }
            _lastTTSProvider = nextProvider;
        }

        function handleTTSProviderChange() {
            const provider = document.getElementById('tts-provider')?.value || 'qwen3';
            applyProviderScriptMaxLengthDefault(provider);
            toggleVoxCPM2Options();
        }

        let _llmToolCapabilityCache = null;
        let _llmToolCapabilityTimer = null;
        let _llmToolCapabilityRequestId = 0;
        let _lmStudioModelSuggestionsCache = null;
        let _lmStudioModelSuggestionsInFlight = null;
        let _lmStudioModelSuggestionKeys = new Set();

        function _getLLMToolCapabilityElements() {
            return {
                status: document.getElementById('llm-tool-capability-status'),
                model: document.getElementById('llm-model'),
                url: document.getElementById('llm-url'),
                key: document.getElementById('llm-key'),
                legacy: document.getElementById('legacy-mode-toggle')
            };
        }

        function _renderLLMToolCapabilityStatus(state, message = '') {
            const { status } = _getLLMToolCapabilityElements();
            if (!status) return;
            status.title = message || '';
            status.setAttribute('aria-label', message || state || 'Tool calling verification');
            status.className = 'llm-tool-capability-status';
            if (!state || state === 'hidden') {
                status.style.display = 'none';
                status.innerHTML = '';
                return;
            }
            status.style.display = '';
            if (state === 'checking') {
                status.classList.add('text-muted');
                status.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            } else if (state === 'supported') {
                status.classList.add('text-success');
                status.innerHTML = '<i class="fas fa-check-circle"></i>';
            } else if (state === 'unsupported') {
                status.classList.add('text-danger');
                status.innerHTML = '<i class="fas fa-times-circle"></i>';
            } else {
                status.classList.add('text-warning');
                status.innerHTML = '<i class="fas fa-question-circle"></i>';
            }
        }

        function _clearLLMToolCapabilityCache() {
            _llmToolCapabilityCache = null;
            _llmToolCapabilityRequestId += 1;
            if (_llmToolCapabilityTimer) {
                clearTimeout(_llmToolCapabilityTimer);
                _llmToolCapabilityTimer = null;
            }
        }

        async function _verifyLLMToolCapabilityNow() {
            const { model, url, key, legacy } = _getLLMToolCapabilityElements();
            if (!model || !url || !key) return;
            if (legacy?.checked) {
                _renderLLMToolCapabilityStatus('hidden');
                return;
            }
            const modelName = model.value.trim();
            if (!modelName) {
                _renderLLMToolCapabilityStatus('hidden');
                return;
            }
            if (_llmToolCapabilityCache) {
                _renderLLMToolCapabilityStatus(_llmToolCapabilityCache.status, _llmToolCapabilityCache.message);
                return;
            }
            const requestId = ++_llmToolCapabilityRequestId;
            _renderLLMToolCapabilityStatus('checking', 'Checking tool calling support...');
            let result;
            try {
                result = await API.post('/api/config/verify_tool_capability', {
                    base_url: url.value,
                    api_key: key.value,
                    model_name: modelName
                });
            } catch (e) {
                result = {
                    status: 'unknown',
                    message: `Could not verify tool calling support: ${e?.message || e}`
                };
            }
            if (requestId !== _llmToolCapabilityRequestId) return;
            const status = ['supported', 'unsupported', 'unknown'].includes(result?.status)
                ? result.status
                : 'unknown';
            _llmToolCapabilityCache = {
                status,
                message: result?.message || 'Tool calling support could not be verified.',
                warned: false
            };
            _renderLLMToolCapabilityStatus(_llmToolCapabilityCache.status, _llmToolCapabilityCache.message);
            if (_llmToolCapabilityCache.status === 'unsupported' && !_llmToolCapabilityCache.warned) {
                _llmToolCapabilityCache.warned = true;
                showToast('Tool calling ability is required for models to process text. Switch to a tool-capable model.', 'warning', 7000);
            }
        }

        function scheduleLLMToolCapabilityVerification(delayMs = 450) {
            const { legacy, model } = _getLLMToolCapabilityElements();
            if (legacy?.checked || !model?.value.trim()) {
                _renderLLMToolCapabilityStatus('hidden');
                return;
            }
            if (_llmToolCapabilityCache) {
                _renderLLMToolCapabilityStatus(_llmToolCapabilityCache.status, _llmToolCapabilityCache.message);
                return;
            }
            if (_llmToolCapabilityTimer) clearTimeout(_llmToolCapabilityTimer);
            _llmToolCapabilityTimer = setTimeout(() => {
                _llmToolCapabilityTimer = null;
                _verifyLLMToolCapabilityNow();
            }, delayMs);
        }

        function _getLLMModelSuggestionElements() {
            const llmElements = _getLLMToolCapabilityElements();
            return {
                ...llmElements,
                popup: document.getElementById('llm-model-suggestion-popup'),
            };
        }

        function _llmSuggestionsCacheKey(baseUrl, apiKey) {
            return `${String(baseUrl || '').trim()}::${String(apiKey || '')}`;
        }

        function _hideLLMModelSuggestionPopup() {
            const { popup } = _getLLMModelSuggestionElements();
            if (!popup) return;
            popup.style.display = 'none';
        }

        function _renderLLMModelSuggestions(models, { showPopup = false } = {}) {
            const { popup, model: modelInput } = _getLLMModelSuggestionElements();
            if (!popup || !modelInput) return;

            popup.innerHTML = '';
            _lmStudioModelSuggestionKeys = new Set();
            for (const modelEntry of Array.isArray(models) ? models : []) {
                const key = String(modelEntry?.key || '').trim();
                if (!key) continue;
                const displayName = String(modelEntry?.display_name || '').trim();
                const option = document.createElement('button');
                option.type = 'button';
                option.className = 'llm-model-suggestion-option';
                option.dataset.modelKey = key;
                if (displayName && displayName !== key) {
                    option.textContent = `${displayName} (${key})`;
                } else {
                    option.textContent = key;
                }
                option.addEventListener('click', () => {
                    modelInput.value = key;
                    _hideLLMModelSuggestionPopup();
                    _clearLLMToolCapabilityCache();
                    _renderLLMToolCapabilityStatus('hidden');
                    modelInput.dispatchEvent(new Event('input', { bubbles: true }));
                    modelInput.dispatchEvent(new Event('change', { bubbles: true }));
                });
                popup.appendChild(option);
                _lmStudioModelSuggestionKeys.add(key);
            }
            if (showPopup && popup.childElementCount > 0) {
                popup.style.display = '';
            } else if (popup.childElementCount === 0) {
                popup.style.display = 'none';
            }
        }

        function _clearLLMModelSuggestionsCache() {
            _lmStudioModelSuggestionsCache = null;
            _lmStudioModelSuggestionsInFlight = null;
            _renderLLMModelSuggestions([], { showPopup: false });
        }

        function _isKnownLMStudioModelSuggestion(value) {
            return _lmStudioModelSuggestionKeys.has(String(value || '').trim());
        }

        async function _fetchLMStudioModelSuggestions({ showPopup = false } = {}) {
            const { url, key, popup } = _getLLMModelSuggestionElements();
            if (!url || !popup) return [];
            const baseUrl = String(url.value || '').trim();
            const apiKey = String(key?.value || '');
            if (!baseUrl) {
                _renderLLMModelSuggestions([], { showPopup: false });
                return [];
            }

            const cacheKey = _llmSuggestionsCacheKey(baseUrl, apiKey);
            if (_lmStudioModelSuggestionsCache?.cacheKey === cacheKey) {
                _renderLLMModelSuggestions(_lmStudioModelSuggestionsCache.models, { showPopup });
                return _lmStudioModelSuggestionsCache.models;
            }

            if (_lmStudioModelSuggestionsInFlight?.cacheKey === cacheKey) {
                return _lmStudioModelSuggestionsInFlight.promise;
            }

            const request = (async () => {
                try {
                    const response = await API.post('/api/config/lmstudio/list_models', {
                        base_url: baseUrl,
                        api_key: apiKey,
                    });
                    const models = Array.isArray(response?.models) ? response.models : [];
                    _lmStudioModelSuggestionsCache = { cacheKey, models };
                    _renderLLMModelSuggestions(models, { showPopup });
                    return models;
                } catch (_error) {
                    _lmStudioModelSuggestionsCache = { cacheKey, models: [] };
                    _renderLLMModelSuggestions([], { showPopup: false });
                    return [];
                } finally {
                    if (_lmStudioModelSuggestionsInFlight?.cacheKey === cacheKey) {
                        _lmStudioModelSuggestionsInFlight = null;
                    }
                }
            })();

            _lmStudioModelSuggestionsInFlight = { cacheKey, promise: request };
            return request;
        }

        function collectExportConfigFromUI() {
            return {
                silence_between_speakers_ms: parseIntOrDefault('silence-between-speakers', 500),
                silence_same_speaker_ms: parseIntOrDefault('silence-same-speaker', 250),
                silence_end_of_chapter_ms: parseIntOrDefault('silence-end-of-chapter', 3000),
                silence_paragraph_ms: parseIntOrDefault('silence-paragraph', 750),
                trim_clip_silence_enabled: document.getElementById('trim-clip-silence-enabled').checked,
                trim_silence_threshold_dbfs: parseFloatOrDefault('trim-silence-threshold-dbfs', -50),
                trim_min_silence_len_ms: parseIntOrDefault('trim-min-silence-len-ms', 150),
                trim_keep_padding_ms: parseIntOrDefault('trim-keep-padding-ms', 40),
                normalize_enabled: document.getElementById('normalize-enabled').checked,
                normalize_target_lufs_mono: parseFloatOrDefault('normalize-target-lufs-mono', -18),
                normalize_target_lufs_stereo: parseFloatOrDefault('normalize-target-lufs-stereo', -16),
                normalize_true_peak_dbtp: parseFloatOrDefault('normalize-true-peak-dbtp', -1),
                normalize_lra: parseFloatOrDefault('normalize-lra', 11)
            };
        }

        window.collectExportConfigFromUI = collectExportConfigFromUI;
        window.persistExportConfigFromUI = async () => {
            const exportConfig = collectExportConfigFromUI();
            await API.post('/api/config/export', exportConfig);
            return exportConfig;
        };

        function wireExportConfigAutoSave() {
            const ids = [
                'silence-between-speakers',
                'silence-same-speaker',
                'silence-end-of-chapter',
                'silence-paragraph',
                'trim-clip-silence-enabled',
                'trim-silence-threshold-dbfs',
                'trim-min-silence-len-ms',
                'trim-keep-padding-ms',
                'normalize-enabled',
                'normalize-target-lufs-mono',
                'normalize-target-lufs-stereo',
                'normalize-true-peak-dbtp',
                'normalize-lra'
            ];
            const debounceMap = new Map();
            const scheduleSave = (id, delayMs = 250) => {
                const prior = debounceMap.get(id);
                if (prior) clearTimeout(prior);
                const token = setTimeout(async () => {
                    debounceMap.delete(id);
                    try {
                        await window.persistExportConfigFromUI();
                    } catch (e) {
                        showToast('Failed to save export settings: ' + (e?.message || e), 'error');
                    }
                }, delayMs);
                debounceMap.set(id, token);
            };
            for (const id of ids) {
                const el = document.getElementById(id);
                if (!el || el.dataset.exportAutosaveBound === '1') continue;
                el.dataset.exportAutosaveBound = '1';
                // checkboxes emit reliable change events; numeric fields are safer on input+blur.
                if (el.type === 'checkbox') {
                    el.addEventListener('change', () => scheduleSave(id, 0));
                } else {
                    el.addEventListener('input', () => scheduleSave(id, 300));
                    el.addEventListener('change', () => scheduleSave(id, 0));
                    el.addEventListener('blur', () => scheduleSave(id, 0));
                }
            }
        }
        wireExportConfigAutoSave();

        async function loadConfig() {
            _setupAutoSaveEnabled = false;
            document.getElementById('chunk-size').value = 3000;
            document.getElementById('max-tokens').value = 4096;

            try {
                const config = await API.get('/api/config');
                document.getElementById('llm-url').value = config.llm.base_url;
                document.getElementById('llm-key').value = config.llm.api_key ?? '';
                document.getElementById('script-error-retry-attempts').value = config.generation && config.generation.script_error_retry_attempts != null
                    ? config.generation.script_error_retry_attempts
                    : 3;
                document.getElementById('llm-model').value = config.llm.model_name;
                document.getElementById('llm-workers').value = config.llm.llm_workers ?? 1;
                const ttsProvider = config.tts.provider || 'qwen3';
                document.getElementById('tts-provider').value = ttsProvider;
                _lastTTSProvider = ttsProvider;
                document.getElementById('tts-mode').value = config.tts.mode || 'external';
                document.getElementById('tts-url').value = config.tts.url || 'http://127.0.0.1:7860';
                document.getElementById('tts-language').value = config.tts.language || 'English';
                document.getElementById('parallel-workers').value = config.tts.parallel_workers || 4;
                if (config.tts.batch_seed != null) {
                    document.getElementById('batch-seed').value = config.tts.batch_seed;
                }
                document.getElementById('compile-codec').checked = !!config.tts.compile_codec;
                document.getElementById('batch-group-by-type').checked = !!config.tts.batch_group_by_type;
                document.getElementById('sub-batch-enabled').checked = config.tts.sub_batch_enabled !== false;
                document.getElementById('auto-regenerate-bad-clips').checked = !!config.tts.auto_regenerate_bad_clips;
                document.getElementById('auto-regenerate-bad-clip-attempts').value = config.tts.auto_regenerate_bad_clip_attempts ?? 3;
                if (config.tts.sub_batch_min_size != null) {
                    document.getElementById('sub-batch-min-size').value = config.tts.sub_batch_min_size;
                }
                if (config.tts.sub_batch_ratio != null) {
                    document.getElementById('sub-batch-ratio').value = config.tts.sub_batch_ratio;
                }
                if (config.tts.sub_batch_max_chars != null) {
                    document.getElementById('sub-batch-max-chars').value = config.tts.sub_batch_max_chars;
                }
                if (config.tts.sub_batch_max_items != null) {
                    document.getElementById('sub-batch-max-items').value = config.tts.sub_batch_max_items;
                }
                document.getElementById('script-max-length').value = config.tts.script_max_length ?? getTTSScriptMaxLengthDefault(ttsProvider);
                document.getElementById('voxcpm-model-id').value = config.tts.voxcpm_model_id || 'openbmb/VoxCPM2';
                document.getElementById('voxcpm-cfg-value').value = clampNumber(
                    config.tts.voxcpm_cfg_value,
                    1,
                    3,
                    VOXCPM2_CFG_VALUE_DEFAULT
                );
                document.getElementById('voxcpm-inference-timesteps').value = clampNumber(
                    config.tts.voxcpm_inference_timesteps,
                    4,
                    30,
                    VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT
                );
                document.getElementById('voxcpm-normalize').checked = coerceConfigBool(config.tts.voxcpm_normalize, false);
                document.getElementById('voxcpm-denoise').checked = coerceConfigBool(config.tts.voxcpm_denoise, false);
                document.getElementById('voxcpm-load-denoiser').checked = coerceConfigBool(config.tts.voxcpm_load_denoiser, false);
                document.getElementById('voxcpm-denoise-reference').checked = coerceConfigBool(config.tts.voxcpm_denoise_reference, false);
                document.getElementById('voxcpm-optimize').checked = coerceConfigBool(config.tts.voxcpm_optimize, false);
                toggleTTSMode();
                applyBatchSettingsVisibility();

                // Load custom prompts if they exist and are non-empty
                if (config.prompts) {
                    if (config.prompts.system_prompt) {
                        document.getElementById('system-prompt').value = config.prompts.system_prompt;
                    }
                    if (config.prompts.user_prompt) {
                        document.getElementById('user-prompt').value = config.prompts.user_prompt;
                    }
                    if (config.prompts.review_system_prompt) {
                        document.getElementById('review-system-prompt').value = config.prompts.review_system_prompt;
                    }
                    if (config.prompts.review_user_prompt) {
                        document.getElementById('review-user-prompt').value = config.prompts.review_user_prompt;
                    }
                    if (config.prompts.attribution_system_prompt) {
                        document.getElementById('attribution-system-prompt').value = config.prompts.attribution_system_prompt;
                    }
                    if (config.prompts.attribution_user_prompt) {
                        document.getElementById('attribution-user-prompt').value = config.prompts.attribution_user_prompt;
                    }
                    if (config.prompts.voice_prompt) {
                        document.getElementById('voice-prompt').value = config.prompts.voice_prompt;
                    }
                    if (config.prompts.dialogue_identification_system_prompt) {
                        document.getElementById('dialogue-identification-system-prompt').value = config.prompts.dialogue_identification_system_prompt;
                    }
                    if (config.prompts.temperament_extraction_system_prompt) {
                        document.getElementById('temperament-extraction-system-prompt').value = config.prompts.temperament_extraction_system_prompt;
                    }
                }

                // If review/attribution prompts are still empty, fetch defaults
                if (
                    !document.getElementById('review-system-prompt').value ||
                    !document.getElementById('review-user-prompt').value ||
                    !document.getElementById('attribution-system-prompt').value ||
                    !document.getElementById('attribution-user-prompt').value ||
                    !document.getElementById('voice-prompt').value ||
                    !document.getElementById('dialogue-identification-system-prompt').value
                ) {
                    try {
                        const defaults = await API.get('/api/default_prompts');
                        if (!document.getElementById('review-system-prompt').value && defaults.review_system_prompt) {
                            document.getElementById('review-system-prompt').value = defaults.review_system_prompt;
                        }
                        if (!document.getElementById('review-user-prompt').value && defaults.review_user_prompt) {
                            document.getElementById('review-user-prompt').value = defaults.review_user_prompt;
                        }
                        if (!document.getElementById('attribution-system-prompt').value && defaults.attribution_system_prompt) {
                            document.getElementById('attribution-system-prompt').value = defaults.attribution_system_prompt;
                        }
                        if (!document.getElementById('attribution-user-prompt').value && defaults.attribution_user_prompt) {
                            document.getElementById('attribution-user-prompt').value = defaults.attribution_user_prompt;
                        }
                        if (!document.getElementById('voice-prompt').value && defaults.voice_prompt) {
                            document.getElementById('voice-prompt').value = defaults.voice_prompt;
                        }
                        if (!document.getElementById('dialogue-identification-system-prompt').value && defaults.dialogue_identification_system_prompt) {
                            document.getElementById('dialogue-identification-system-prompt').value = defaults.dialogue_identification_system_prompt;
                        }
                        if (!document.getElementById('temperament-extraction-system-prompt').value && defaults.temperament_extraction_system_prompt) {
                            document.getElementById('temperament-extraction-system-prompt').value = defaults.temperament_extraction_system_prompt;
                        }
                    } catch (e) {
                        console.warn("Could not fetch default review/attribution prompts", e);
                    }
                }

                // Load generation settings
                if (config.generation) {
                    if (config.generation.chunk_size) {
                        document.getElementById('chunk-size').value = config.generation.chunk_size;
                    }
                    if (config.generation.temperament_words != null) {
                        document.getElementById('temperament-words').value = config.generation.temperament_words;
                    }
                    if (config.generation.script_error_retry_attempts != null) {
                        document.getElementById('script-error-retry-attempts').value = config.generation.script_error_retry_attempts;
                    }
                    if (config.generation.max_tokens) {
                        document.getElementById('max-tokens').value = config.generation.max_tokens;
                    }
                    if (config.generation.temperature != null) {
                        document.getElementById('temperature').value = config.generation.temperature;
                    }
                    if (config.generation.top_p != null) {
                        document.getElementById('top-p').value = config.generation.top_p;
                    }
                    if (config.generation.top_k != null) {
                        document.getElementById('top-k').value = config.generation.top_k;
                    }
                    if (config.generation.min_p != null) {
                        document.getElementById('min-p').value = config.generation.min_p;
                    }
                    if (config.generation.presence_penalty != null) {
                        document.getElementById('presence-penalty').value = config.generation.presence_penalty;
                    }
                    if (config.generation.banned_tokens && config.generation.banned_tokens.length > 0) {
                        document.getElementById('banned-tokens').value = config.generation.banned_tokens.join(', ');
                    }
                    document.getElementById('merge-narrators').checked = !!config.generation.merge_narrators;
                    document.getElementById('orphaned-text-to-narrator-on-repair').checked =
                        config.generation.orphaned_text_to_narrator_on_repair !== false;
                }
                {
                    const toggle = document.getElementById('legacy-mode-toggle');
                    if (toggle) {
                        toggle.dataset.suspendPersist = '1';
                        toggle.checked = !!(config.generation && config.generation.legacy_mode);
                        toggle.dispatchEvent(new Event('change'));
                        delete toggle.dataset.suspendPersist;
                    }
                }
                if (config.ui && typeof config.ui.dark_mode === 'boolean') {
                    const nextTheme = config.ui.dark_mode ? 'dark' : 'light';
                    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
                    applyTheme(nextTheme);
                }
                applyGenerationModeLock(!!config.generation_mode_locked);
                if (config.proofread && config.proofread.certainty_threshold != null) {
                    document.getElementById('proofread-threshold').value = config.proofread.certainty_threshold;
                }
                if (config.export) {
                    if (config.export.silence_between_speakers_ms != null) {
                        document.getElementById('silence-between-speakers').value = config.export.silence_between_speakers_ms;
                    }
                    if (config.export.silence_same_speaker_ms != null) {
                        document.getElementById('silence-same-speaker').value = config.export.silence_same_speaker_ms;
                    }
                    if (config.export.silence_end_of_chapter_ms != null) {
                        document.getElementById('silence-end-of-chapter').value = config.export.silence_end_of_chapter_ms;
                    }
                    if (config.export.silence_paragraph_ms != null) {
                        document.getElementById('silence-paragraph').value = config.export.silence_paragraph_ms;
                    }
                    if (config.export.trim_clip_silence_enabled != null) {
                        document.getElementById('trim-clip-silence-enabled').checked = coerceConfigBool(config.export.trim_clip_silence_enabled, true);
                    }
                    if (config.export.trim_silence_threshold_dbfs != null) {
                        document.getElementById('trim-silence-threshold-dbfs').value = config.export.trim_silence_threshold_dbfs;
                    }
                    if (config.export.trim_min_silence_len_ms != null) {
                        document.getElementById('trim-min-silence-len-ms').value = config.export.trim_min_silence_len_ms;
                    }
                    if (config.export.trim_keep_padding_ms != null) {
                        document.getElementById('trim-keep-padding-ms').value = config.export.trim_keep_padding_ms;
                    }
                    if (config.export.normalize_enabled != null) {
                        document.getElementById('normalize-enabled').checked = coerceConfigBool(config.export.normalize_enabled, true);
                    }
                    if (config.export.normalize_target_lufs_mono != null) {
                        document.getElementById('normalize-target-lufs-mono').value = config.export.normalize_target_lufs_mono;
                    }
                    if (config.export.normalize_target_lufs_stereo != null) {
                        document.getElementById('normalize-target-lufs-stereo').value = config.export.normalize_target_lufs_stereo;
                    }
                    if (config.export.normalize_true_peak_dbtp != null) {
                        document.getElementById('normalize-true-peak-dbtp').value = config.export.normalize_true_peak_dbtp;
                    }
                    if (config.export.normalize_lra != null) {
                        document.getElementById('normalize-lra').value = config.export.normalize_lra;
                    }
                }

                // Show previously loaded file
                if (config.current_file) {
                    document.getElementById('upload-status').innerHTML =
                        `<span class="text-success"><i class="fas fa-check me-1"></i>Loaded: ${config.current_file}</span>`;
                    document.getElementById('file-upload-section').style.display = 'none';
                    document.getElementById('script-help-text').style.display = '';
                }
                renderPrepComplete = !!config.render_prep_complete;
                refreshPromptTextareaHeights();

                // Default to Settings tab if LLM is not configured
                if (!config.llm?.base_url || !config.llm?.model_name) {
                    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                    document.getElementById('setup-tab').style.display = '';
                    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                    document.querySelector('.nav-link[data-tab="setup"]').classList.add('active');
                }
            } catch (e) {
                console.error("Failed to load config", e);
            }
            _setupAutoSaveEnabled = true;
            scheduleLLMToolCapabilityVerification(100);
        }

        // Reset prompts and generation settings to factory defaults
        window.resetPrompts = async () => {
            _setupAutoSaveEnabled = false;
            try {
                const defaults = await API.get('/api/factory_default_prompts');
                document.getElementById('system-prompt').value = defaults.system_prompt;
                document.getElementById('user-prompt').value = defaults.user_prompt;
                if (defaults.review_system_prompt) {
                    document.getElementById('review-system-prompt').value = defaults.review_system_prompt;
                }
                if (defaults.review_user_prompt) {
                    document.getElementById('review-user-prompt').value = defaults.review_user_prompt;
                }
                if (defaults.attribution_system_prompt) {
                    document.getElementById('attribution-system-prompt').value = defaults.attribution_system_prompt;
                }
                if (defaults.attribution_user_prompt) {
                    document.getElementById('attribution-user-prompt').value = defaults.attribution_user_prompt;
                }
                if (defaults.voice_prompt) {
                    document.getElementById('voice-prompt').value = defaults.voice_prompt;
                }
                if (defaults.dialogue_identification_system_prompt) {
                    document.getElementById('dialogue-identification-system-prompt').value = defaults.dialogue_identification_system_prompt;
                }
                if (defaults.temperament_extraction_system_prompt) {
                    document.getElementById('temperament-extraction-system-prompt').value = defaults.temperament_extraction_system_prompt;
                }
            } catch (e) {
                console.error("Failed to fetch default prompts", e);
                showToast("Failed to load default prompts from server.", 'error');
            }
            document.getElementById('chunk-size').value = 3000;
            document.getElementById('max-tokens').value = 4096;
            document.getElementById('temperature').value = 0.6;
            document.getElementById('top-p').value = 0.8;
            document.getElementById('top-k').value = 20;
            document.getElementById('min-p').value = 0;
            document.getElementById('presence-penalty').value = 0;
            document.getElementById('banned-tokens').value = '';
            document.getElementById('merge-narrators').checked = false;
            document.getElementById('orphaned-text-to-narrator-on-repair').checked = true;
            document.getElementById('script-error-retry-attempts').value = 3;
            document.getElementById('auto-regenerate-bad-clip-attempts').value = 3;
            document.getElementById('proofread-threshold').value = 0.75;
            refreshPromptTextareaHeights();
            _setupAutoSaveEnabled = true;
            _setupDirtySections.add('prompts');
            _setupDirtySections.add('generation');
            _setupDirtySections.add('proofread');
            _flushSetupConfig();
        };

        // Toggle chevron on collapse
        document.getElementById('promptSettings')?.addEventListener('show.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-right', 'fa-chevron-down');
            requestAnimationFrame(() => refreshPromptTextareaHeights());
        });
        document.getElementById('promptSettings')?.addEventListener('hide.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-down', 'fa-chevron-right');
        });

        // --- Setup Config Auto-Save ---

        function collectLLMConfig() {
            return {
                base_url: document.getElementById('llm-url').value,
                api_key: document.getElementById('llm-key').value,
                model_name: document.getElementById('llm-model').value,
                llm_workers: parseInt(document.getElementById('llm-workers').value) || 1
            };
        }

        function collectTTSConfig() {
            let parallelWorkers = parseInt(document.getElementById('parallel-workers').value) || 2;
            parallelWorkers = Math.max(1, parallelWorkers);
            const rawRetryAttempts = parseInt(document.getElementById('auto-regenerate-bad-clip-attempts').value, 10);
            const retryAttempts = Number.isInteger(rawRetryAttempts) && rawRetryAttempts > 0 ? rawRetryAttempts : 0;
            const provider = document.getElementById('tts-provider').value || 'qwen3';
            const scriptMaxLength = parseInt(document.getElementById('script-max-length').value, 10);
            return {
                provider,
                mode: document.getElementById('tts-mode').value,
                local_backend: 'auto',
                url: document.getElementById('tts-url').value,
                device: 'auto',
                language: document.getElementById('tts-language').value,
                parallel_workers: parallelWorkers,
                batch_seed: document.getElementById('batch-seed').value ? parseInt(document.getElementById('batch-seed').value) : null,
                compile_codec: document.getElementById('compile-codec').checked,
                batch_group_by_type: document.getElementById('batch-group-by-type').checked,
                sub_batch_enabled: document.getElementById('sub-batch-enabled').checked,
                auto_regenerate_bad_clips: document.getElementById('auto-regenerate-bad-clips').checked,
                auto_regenerate_bad_clip_attempts: retryAttempts,
                sub_batch_min_size: parseInt(document.getElementById('sub-batch-min-size').value) || 4,
                sub_batch_ratio: parseFloat(document.getElementById('sub-batch-ratio').value) || 5,
                sub_batch_max_chars: parseInt(document.getElementById('sub-batch-max-chars').value) || 3000,
                sub_batch_max_items: parseInt(document.getElementById('sub-batch-max-items').value) || 0,
                script_max_length: Number.isInteger(scriptMaxLength) ? scriptMaxLength : getTTSScriptMaxLengthDefault(provider),
                voxcpm_model_id: document.getElementById('voxcpm-model-id').value || 'openbmb/VoxCPM2',
                voxcpm_cfg_value: clampNumber(
                    parseFloatOrDefault('voxcpm-cfg-value', VOXCPM2_CFG_VALUE_DEFAULT),
                    1,
                    3,
                    VOXCPM2_CFG_VALUE_DEFAULT
                ),
                voxcpm_inference_timesteps: clampNumber(
                    parseIntOrDefault('voxcpm-inference-timesteps', VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT),
                    4,
                    30,
                    VOXCPM2_INFERENCE_TIMESTEPS_DEFAULT
                ),
                voxcpm_normalize: document.getElementById('voxcpm-normalize').checked,
                voxcpm_denoise: document.getElementById('voxcpm-denoise').checked,
                voxcpm_load_denoiser: document.getElementById('voxcpm-load-denoiser').checked,
                voxcpm_denoise_reference: document.getElementById('voxcpm-denoise-reference').checked,
                voxcpm_optimize: !isMacHostUI() && document.getElementById('voxcpm-optimize').checked
            };
        }

        function collectPromptsConfig() {
            return {
                system_prompt: document.getElementById('system-prompt').value,
                user_prompt: document.getElementById('user-prompt').value,
                review_system_prompt: document.getElementById('review-system-prompt').value,
                review_user_prompt: document.getElementById('review-user-prompt').value,
                attribution_system_prompt: document.getElementById('attribution-system-prompt').value,
                attribution_user_prompt: document.getElementById('attribution-user-prompt').value,
                voice_prompt: document.getElementById('voice-prompt').value,
                dialogue_identification_system_prompt: document.getElementById('dialogue-identification-system-prompt').value,
                temperament_extraction_system_prompt: document.getElementById('temperament-extraction-system-prompt').value
            };
        }

        function collectGenerationConfig() {
            const rawScriptErrorRetryAttempts = parseInt(document.getElementById('script-error-retry-attempts').value, 10);
            const scriptErrorRetryAttempts = Number.isInteger(rawScriptErrorRetryAttempts) && rawScriptErrorRetryAttempts > 0
                ? rawScriptErrorRetryAttempts
                : 0;
            return {
                chunk_size: parseInt(document.getElementById('chunk-size').value) || 3000,
                temperament_words: parseInt(document.getElementById('temperament-words').value) || 150,
                script_error_retry_attempts: scriptErrorRetryAttempts,
                max_tokens: parseInt(document.getElementById('max-tokens').value) || 4096,
                temperature: parseFloat(document.getElementById('temperature').value),
                top_p: parseFloat(document.getElementById('top-p').value),
                top_k: parseInt(document.getElementById('top-k').value),
                min_p: parseFloat(document.getElementById('min-p').value),
                presence_penalty: parseFloat(document.getElementById('presence-penalty').value),
                banned_tokens: document.getElementById('banned-tokens').value
                    ? document.getElementById('banned-tokens').value.split(',').map(t => t.trim()).filter(t => t)
                    : [],
                merge_narrators: document.getElementById('merge-narrators').checked,
                orphaned_text_to_narrator_on_repair: document.getElementById('orphaned-text-to-narrator-on-repair').checked,
                legacy_mode: document.getElementById('legacy-mode-toggle').checked
            };
        }

        function collectProofreadConfig() {
            return {
                certainty_threshold: parseFloat(document.getElementById('proofread-threshold').value) || 0.75
            };
        }

        const _setupSectionCollectors = {
            llm: collectLLMConfig,
            tts: collectTTSConfig,
            prompts: collectPromptsConfig,
            generation: collectGenerationConfig,
            proofread: collectProofreadConfig
        };

        let _setupAutoSaveEnabled = false;
        let _setupSaveTimer = null;
        const _setupDirtySections = new Set();
        let _setupSaveIndicatorTimer = null;

        function _showSetupSaveIndicator() {
            const el = document.getElementById('setup-save-indicator');
            if (!el) return;
            el.style.display = '';
            if (_setupSaveIndicatorTimer) clearTimeout(_setupSaveIndicatorTimer);
            _setupSaveIndicatorTimer = setTimeout(() => { el.style.display = 'none'; }, 2000);
        }

        async function _flushSetupConfig() {
            if (_setupSaveTimer) { clearTimeout(_setupSaveTimer); _setupSaveTimer = null; }
            if (_setupDirtySections.size === 0) return;
            const payload = {};
            for (const section of _setupDirtySections) {
                payload[section] = _setupSectionCollectors[section]();
            }
            _setupDirtySections.clear();
            try {
                await API.post('/api/config/setup', payload);
                _showSetupSaveIndicator();
            } catch (e) {
                showToast('Failed to save settings: ' + (e?.message || e), 'error');
            }
        }
        window.flushSetupConfig = _flushSetupConfig;

        function _scheduleSetupSave(section, delayMs) {
            if (!_setupAutoSaveEnabled) return;
            _setupDirtySections.add(section);
            if (_setupSaveTimer) clearTimeout(_setupSaveTimer);
            _setupSaveTimer = setTimeout(() => { _setupSaveTimer = null; _flushSetupConfig(); }, delayMs);
        }

        function wireSetupConfigAutoSave() {
            const fieldSectionMap = {
                // LLM
                'llm-url': 'llm', 'llm-key': 'llm', 'llm-model': 'llm', 'llm-workers': 'llm',
                // TTS
                'tts-provider': 'tts', 'tts-mode': 'tts', 'tts-url': 'tts', 'tts-language': 'tts',
                'parallel-workers': 'tts', 'batch-seed': 'tts',
                'compile-codec': 'tts', 'batch-group-by-type': 'tts',
                'sub-batch-enabled': 'tts', 'sub-batch-min-size': 'tts',
                'sub-batch-ratio': 'tts', 'sub-batch-max-chars': 'tts',
                'sub-batch-max-items': 'tts', 'auto-regenerate-bad-clips': 'tts',
                'auto-regenerate-bad-clip-attempts': 'tts', 'script-max-length': 'tts',
                'voxcpm-model-id': 'tts', 'voxcpm-cfg-value': 'tts',
                'voxcpm-inference-timesteps': 'tts', 'voxcpm-normalize': 'tts',
                'voxcpm-denoise': 'tts', 'voxcpm-load-denoiser': 'tts', 'voxcpm-denoise-reference': 'tts',
                'voxcpm-optimize': 'tts',
                // Generation
                'script-error-retry-attempts': 'generation', 'chunk-size': 'generation', 'max-tokens': 'generation',
                'temperature': 'generation', 'top-p': 'generation', 'top-k': 'generation',
                'min-p': 'generation', 'presence-penalty': 'generation',
                'banned-tokens': 'generation', 'merge-narrators': 'generation',
                'orphaned-text-to-narrator-on-repair': 'generation',
                // Prompts
                'system-prompt': 'prompts', 'user-prompt': 'prompts',
                'review-system-prompt': 'prompts', 'review-user-prompt': 'prompts',
                'attribution-system-prompt': 'prompts', 'attribution-user-prompt': 'prompts',
                'voice-prompt': 'prompts',
                'dialogue-identification-system-prompt': 'prompts',
                'temperament-extraction-system-prompt': 'prompts',
                // Proofread
                'proofread-threshold': 'proofread'
            };

            for (const [id, section] of Object.entries(fieldSectionMap)) {
                const el = document.getElementById(id);
                if (!el || el.dataset.setupAutosaveBound === '1') continue;
                el.dataset.setupAutosaveBound = '1';
                const isTextarea = el.tagName === 'TEXTAREA';
                const isCheckbox = el.type === 'checkbox';
                const isSelect = el.tagName === 'SELECT';
                if (id === 'tts-provider') {
                    el.addEventListener('change', handleTTSProviderChange);
                }
                if (isCheckbox || isSelect) {
                    el.addEventListener('change', () => _scheduleSetupSave(section, 0));
                } else if (isTextarea) {
                    el.addEventListener('input', () => _scheduleSetupSave(section, 1500));
                    el.addEventListener('blur', () => _scheduleSetupSave(section, 0));
                } else {
                    el.addEventListener('input', () => _scheduleSetupSave(section, 500));
                    el.addEventListener('change', () => _scheduleSetupSave(section, 0));
                    el.addEventListener('blur', () => _scheduleSetupSave(section, 0));
                }
            }
        }
        wireSetupConfigAutoSave();

        function wireLLMToolCapabilityVerification() {
            const { model, url, key, legacy, popup } = _getLLMModelSuggestionElements();
            if (model && model.dataset.toolCapabilityBound !== '1') {
                model.dataset.toolCapabilityBound = '1';
                model.addEventListener('input', () => {
                    _clearLLMToolCapabilityCache();
                    _renderLLMToolCapabilityStatus('hidden');
                    if (_isKnownLMStudioModelSuggestion(model.value)) {
                        _verifyLLMToolCapabilityNow();
                        return;
                    }
                    scheduleLLMToolCapabilityVerification(650);
                });
                model.addEventListener('change', () => {
                    _clearLLMToolCapabilityCache();
                    _renderLLMToolCapabilityStatus('hidden');
                    if (_isKnownLMStudioModelSuggestion(model.value)) {
                        _verifyLLMToolCapabilityNow();
                        return;
                    }
                    scheduleLLMToolCapabilityVerification(0);
                });
                model.addEventListener('blur', () => {
                    setTimeout(() => _hideLLMModelSuggestionPopup(), 120);
                    scheduleLLMToolCapabilityVerification(0);
                });
                model.addEventListener('focus', () => {
                    _fetchLMStudioModelSuggestions({ showPopup: true });
                });
                model.addEventListener('click', () => {
                    _fetchLMStudioModelSuggestions({ showPopup: true });
                });
            }
            if (legacy && legacy.dataset.toolCapabilityBound !== '1') {
                legacy.dataset.toolCapabilityBound = '1';
                legacy.addEventListener('change', () => {
                    if (legacy.checked) {
                        _renderLLMToolCapabilityStatus('hidden');
                    } else {
                        scheduleLLMToolCapabilityVerification(100);
                    }
                });
            }
            for (const sourceField of [url, key]) {
                if (!sourceField || sourceField.dataset.modelSuggestionsBound === '1') continue;
                sourceField.dataset.modelSuggestionsBound = '1';
                sourceField.addEventListener('input', () => {
                    _clearLLMToolCapabilityCache();
                    _renderLLMToolCapabilityStatus('hidden');
                    _clearLLMModelSuggestionsCache();
                });
                sourceField.addEventListener('change', () => {
                    _clearLLMToolCapabilityCache();
                    _renderLLMToolCapabilityStatus('hidden');
                    _clearLLMModelSuggestionsCache();
                });
            }
            if (popup && popup.dataset.modelSuggestionsPopupBound !== '1') {
                popup.dataset.modelSuggestionsPopupBound = '1';
                popup.addEventListener('mousedown', (event) => {
                    event.preventDefault();
                });
            }
            if (model && popup && document.body && document.body.dataset.modelSuggestionsDocBound !== '1') {
                document.body.dataset.modelSuggestionsDocBound = '1';
                document.addEventListener('click', (event) => {
                    const target = event?.target;
                    if (target === model) return;
                    if (popup.contains(target)) return;
                    _hideLLMModelSuggestionPopup();
                });
                document.addEventListener('keydown', (event) => {
                    if (event?.key === 'Escape') {
                        _hideLLMModelSuggestionPopup();
                    }
                });
            }
        }
        wireLLMToolCapabilityVerification();

        // Prevent form submission (Enter key in text fields)
        document.getElementById('config-form').addEventListener('submit', (e) => {
            e.preventDefault();
        });

        // Best-effort save on page unload
        window.addEventListener('beforeunload', () => {
            if (_setupDirtySections.size === 0) return;
            const payload = {};
            for (const section of _setupDirtySections) {
                payload[section] = _setupSectionCollectors[section]();
            }
            _setupDirtySections.clear();
            const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
            navigator.sendBeacon('/api/config/setup', blob);
        });

        const resetProjectButton = document.getElementById('btn-reset-project');
        if (resetProjectButton) {
            resetProjectButton.addEventListener('click', window.resetProject);
        }
