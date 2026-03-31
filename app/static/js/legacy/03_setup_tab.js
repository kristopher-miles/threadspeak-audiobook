        // --- Setup Tab ---

        function toggleTTSMode() {
            const mode = document.getElementById('tts-mode').value;
            document.getElementById('tts-url-group').style.display = mode === 'external' ? '' : 'none';
            document.getElementById('tts-device-group').style.display = mode === 'local' ? '' : 'none';
            document.getElementById('tts-local-options').style.display = mode === 'local' ? '' : 'none';
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
            document.getElementById('chunk-size').value = 3000;
            document.getElementById('max-tokens').value = 4096;

            try {
                const config = await API.get('/api/config');
                document.getElementById('llm-url').value = config.llm.base_url;
                document.getElementById('llm-key').value = config.llm.api_key;
                document.getElementById('llm-model').value = config.llm.model_name;
                document.getElementById('tts-mode').value = config.tts.mode || 'external';
                document.getElementById('tts-url').value = config.tts.url || 'http://127.0.0.1:7860';
                document.getElementById('tts-device').value = config.tts.device || 'auto';
                document.getElementById('tts-language').value = config.tts.language || 'English';
                document.getElementById('parallel-workers').value = config.tts.parallel_workers || 2;
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
                toggleTTSMode();

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
                }
                renderPrepComplete = !!config.render_prep_complete;
                refreshPromptTextareaHeights();
            } catch (e) {
                console.error("Failed to load config", e);
            }
        }

        // Reset prompts and generation settings to factory defaults
        window.resetPrompts = async () => {
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
            document.getElementById('auto-regenerate-bad-clip-attempts').value = 3;
            document.getElementById('proofread-threshold').value = 0.7;
            refreshPromptTextareaHeights();
        };

        // Toggle chevron on collapse
        document.getElementById('promptSettings')?.addEventListener('show.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-right', 'fa-chevron-down');
            requestAnimationFrame(() => refreshPromptTextareaHeights());
        });
        document.getElementById('promptSettings')?.addEventListener('hide.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-down', 'fa-chevron-right');
        });

        document.getElementById('config-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            let chunkSize = parseInt(document.getElementById('chunk-size').value) || 3000;

            // Validate parallel workers
            let parallelWorkers = parseInt(document.getElementById('parallel-workers').value) || 2;
            parallelWorkers = Math.max(1, parallelWorkers);
            document.getElementById('parallel-workers').value = parallelWorkers;

            const rawRetryAttempts = parseInt(document.getElementById('auto-regenerate-bad-clip-attempts').value, 10);
            const retryAttempts = Number.isInteger(rawRetryAttempts) && rawRetryAttempts > 0 ? rawRetryAttempts : 0;
            document.getElementById('auto-regenerate-bad-clip-attempts').value = retryAttempts > 0 ? retryAttempts : 0;

            const config = {
                llm: {
                    base_url: document.getElementById('llm-url').value,
                    api_key: document.getElementById('llm-key').value,
                    model_name: document.getElementById('llm-model').value
                },
                tts: {
                    mode: document.getElementById('tts-mode').value,
                    url: document.getElementById('tts-url').value,
                    device: document.getElementById('tts-device').value,
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
                    sub_batch_max_items: parseInt(document.getElementById('sub-batch-max-items').value) || 0
                },
                prompts: {
                    system_prompt: document.getElementById('system-prompt').value,
                    user_prompt: document.getElementById('user-prompt').value,
                    review_system_prompt: document.getElementById('review-system-prompt').value,
                    review_user_prompt: document.getElementById('review-user-prompt').value,
                    attribution_system_prompt: document.getElementById('attribution-system-prompt').value,
                    attribution_user_prompt: document.getElementById('attribution-user-prompt').value,
                    voice_prompt: document.getElementById('voice-prompt').value,
                    dialogue_identification_system_prompt: document.getElementById('dialogue-identification-system-prompt').value,
                    temperament_extraction_system_prompt: document.getElementById('temperament-extraction-system-prompt').value
                },
                generation: {
                    chunk_size: chunkSize,
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
                },
                proofread: {
                    certainty_threshold: parseFloat(document.getElementById('proofread-threshold').value) || 0.7
                },
                export: {
                    ...collectExportConfigFromUI()
                },
                ui: {
                    dark_mode: document.getElementById('dark-mode-toggle').checked
                }
            };
            try {
                await API.post('/api/config', config);
                showToast('Configuration Saved!', 'success');
            } catch (e) {
                showToast('Error saving config: ' + e.message, 'error');
            }
        });

        document.getElementById('btn-reset-project').addEventListener('click', async () => {
            const confirmed = await showConfirm('Reset the current project? This will delete the loaded source, generated script, detected characters, chunk audio, final audiobook files, and sanity/repair state. Global settings and prompt configuration will be kept.');
            if (!confirmed) return;

            try {
                await API.post('/api/reset_project', {});
                showToast('Project reset. Reloading...', 'success', 1500);
                setTimeout(() => window.location.reload(), 500);
            } catch (e) {
                showToast('Failed to reset project: ' + e.message, 'error');
            }
        });
