        // --- Voices Tab ---
        const AVAILABLE_VOICES = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"];

        function normalizeVoiceName(value) {
            return (value || '').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
        }

        let _voiceSettingsSaveTimer = null;
        let _voicesListResizeBound = false;
        let _bulkVoiceGenerationActive = false;

        function layoutVoicesListContainer() {
            const list = document.getElementById('voices-list');
            const tab = document.getElementById('voices-tab');
            if (!list || !tab || tab.style.display === 'none') return;

            // Reset before measuring so we compute available viewport space accurately.
            list.style.maxHeight = 'none';
            list.style.height = 'auto';

            const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
            const rect = list.getBoundingClientRect();
            const bottomPadding = 24;
            const available = Math.floor(viewportHeight - rect.top - bottomPadding);
            const minScrollableHeight = 220;

            if (available >= minScrollableHeight) {
                list.style.maxHeight = `${available}px`;
                list.style.overflowY = 'auto';
            } else {
                list.style.maxHeight = 'none';
                list.style.overflowY = 'visible';
            }
        }

        function getNarratorThresholdValue() {
            const input = document.getElementById('narrator-threshold-input');
            if (!input) return 10;
            const parsed = parseInt(input.value, 10);
            return Number.isFinite(parsed) && parsed >= 0 ? parsed : 10;
        }

        function getNarratorName(cards) {
            const narratorCard = cards.find(card => normalizeVoiceName(card.dataset.voice) === normalizeVoiceName('NARRATOR'));
            return narratorCard ? narratorCard.dataset.voice : '';
        }

        async function loadVoiceSettings() {
            const input = document.getElementById('narrator-threshold-input');
            if (!input) return;
            try {
                const result = await API.get('/api/voices/settings');
                const value = Number(result?.narrator_threshold);
                input.value = Number.isFinite(value) && value >= 0 ? String(value) : '10';
            } catch (_e) {
                input.value = input.value || '10';
            }
        }

        async function saveVoiceSettingsNow() {
            const input = document.getElementById('narrator-threshold-input');
            if (!input) return;
            const value = getNarratorThresholdValue();
            input.value = String(value);
            await API.post('/api/voices/settings', { value });
        }

        function debouncedSaveVoiceSettings() {
            clearTimeout(_voiceSettingsSaveTimer);
            _voiceSettingsSaveTimer = setTimeout(() => {
                saveVoiceSettingsNow().then(() => {
                    updateVoiceAliasStates();
                }).catch((e) => {
                    showToast(`Failed to save narrator threshold: ${e.message}`, 'error');
                });
            }, 500);
        }

        // Suggest aliases for names where one is a substring of another.
        // The name with fewer paragraphs gets aliased to the one with more.
        // Only fills empty AKA inputs — never overwrites a user-set alias.
        function suggestVoiceAliases(voices) {
            // paragraph_count is 0 for legacy projects (no paragraphs.json) — skip silently
            const hasCounts = voices.some(v => (v.paragraph_count || 0) > 0);
            if (!hasCounts) return false;

            // Build a lookup: normalizedName -> voice object
            const byName = new Map(voices.map(v => [normalizeVoiceName(v.name), v]));

            let anySet = false;
            const processed = new Set(); // avoid processing the same pair twice

            for (const voice of voices) {
                const normA = normalizeVoiceName(voice.name);

                for (const other of voices) {
                    const normB = normalizeVoiceName(other.name);
                    if (normA === normB) continue;

                    const pairKey = [normA, normB].sort().join('||');
                    if (processed.has(pairKey)) continue;
                    processed.add(pairKey);

                    // Check if one name fully appears within the other
                    const aInB = normB.includes(normA);
                    const bInA = normA.includes(normB);
                    if (!aInB && !bInA) continue;

                    // The shorter/contained name is the candidate to alias;
                    // the one with more paragraphs is the target.
                    const countA = voice.paragraph_count || 0;
                    const countB = other.paragraph_count || 0;

                    let aliasName, targetName;
                    if (countA <= countB) {
                        aliasName = voice.name;   // fewer (or equal) → alias this one
                        targetName = other.name;
                    } else {
                        aliasName = other.name;
                        targetName = voice.name;
                    }

                    // Find the card for the alias candidate and fill its AKA only if empty
                    const card = Array.from(document.querySelectorAll('.voice-card'))
                        .find(el => el.dataset.voice === aliasName);
                    const aliasInput = card?.querySelector('.voice-alias-input');
                    if (!aliasInput || aliasInput.value.trim()) continue; // already set

                    aliasInput.value = targetName;
                    anySet = true;
                }
            }

            if (anySet) {
                updateVoiceAliasStates();
            }
            return anySet;
        }

        function getVoiceAliasTarget(card, nameMap) {
            const aliasInput = card.querySelector('.voice-alias-input');
            if (!aliasInput) return '';

            const alias = normalizeVoiceName(aliasInput.value);
            if (!alias) return '';

            const target = nameMap.get(alias);
            if (!target) return '';

            return normalizeVoiceName(target) === normalizeVoiceName(card.dataset.voice) ? '' : target;
        }

        function updateVoiceAliasStates() {
            const cards = Array.from(document.querySelectorAll('.voice-card'));
            const nameMap = new Map(cards.map(card => [normalizeVoiceName(card.dataset.voice), card.dataset.voice]));
            const narratorThreshold = getNarratorThresholdValue();
            const narratorName = getNarratorName(cards);

            cards.forEach(card => {
                const target = getVoiceAliasTarget(card, nameMap);
                const lineCount = Number(card.dataset.lineCount || 0);
                const isNarrator = normalizeVoiceName(card.dataset.voice) === normalizeVoiceName('NARRATOR');
                const thresholdTarget = (!target && narratorName && !isNarrator && lineCount < narratorThreshold)
                    ? narratorName
                    : '';
                const disabled = Boolean(target || thresholdTarget);
                const manualAliasActive = Boolean(target);
                const narratorThresholdActive = Boolean(thresholdTarget);

                card.classList.toggle('alias-active', manualAliasActive);
                card.classList.toggle('narrator-threshold-active', narratorThresholdActive);
                card.querySelectorAll('input, select, textarea, button').forEach(control => {
                    if (control.classList.contains('voice-alias-input')) {
                        control.disabled = false;
                        return;
                    }
                    control.disabled = disabled;
                });

                const aliasInput = card.querySelector('.voice-alias-input');
                if (aliasInput) {
                    if (manualAliasActive) {
                        aliasInput.title = `Aliased to ${target}. Generation will use that character's voice settings.`;
                    } else if (narratorThresholdActive) {
                        aliasInput.title = `Auto-aliased to ${thresholdTarget} because ${lineCount} line${lineCount === 1 ? '' : 's'} is below narrator threshold ${narratorThreshold}. Set AKA to override this.`;
                    } else {
                        aliasInput.title = 'Match another visible character name here to disable this row and reuse that voice.';
                    }
                }
            });
        }

        function getVoiceCardByName(name) {
            return Array.from(document.querySelectorAll('.voice-card'))
                .find(card => normalizeVoiceName(card.dataset.voice) === normalizeVoiceName(name));
        }

        function createVoiceCard(voice, index) {
            const config = voice.config || {};
            const voiceType = config.type || 'design';
            const safeName = escapeHtml(voice.name);
            const safeAlias = escapeHtml(config.alias || '');
            const designSampleText = config.ref_text || voice.suggested_sample_text || '';
            const designLoaded = Boolean((config.ref_audio || '').trim()) && !!voice.design_clone_loaded;
            const lineCount = Number(voice.line_count || 0);
            const lineLabel = `${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`;
            const isNarratorCard = normalizeVoiceName(voice.name) === normalizeVoiceName('NARRATOR');
            const narratesChecked = isNarratorCard || config.narrates === true;

            return `
                <div class="card voice-card mb-3" data-voice="${safeName}" data-line-count="${lineCount}" data-suggested-sample="${escapeHtml(voice.suggested_sample_text || '')}">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <h5 class="card-title voice-primary-content d-flex flex-wrap align-items-center gap-2">
                                    <span>${safeName}</span>
                                    <span class="badge text-bg-secondary" style="cursor:pointer;" title="Jump to first line in Editor" onclick="jumpToFirstChunkForVoice('${safeName}')">${lineLabel}</span>
                                    <div class="form-check form-check-inline mb-0">
                                        <input class="form-check-input voice-narrates" type="checkbox" id="narrates_${index}" ${narratesChecked ? 'checked' : ''} ${isNarratorCard ? 'disabled' : ''} onchange="debouncedSaveVoices()">
                                        <label class="form-check-label small" for="narrates_${index}">Narrates</label>
                                    </div>
                                </h5>
                                <div class="voice-alias-box mt-2">
                                    <label class="form-label small text-muted mb-1">AKA</label>
                                    <input
                                        type="text"
                                        class="form-control form-control-sm voice-alias-input"
                                        value="${safeAlias}"
                                        title="Match another visible character name here to disable this row and reuse that voice."
                                    >
                                </div>
                            </div>
                            <div class="col-md-9 voice-primary-content">
                                <div class="mb-2">
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="custom" ${voiceType === 'custom' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Custom Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="builtin_lora" ${voiceType === 'builtin_lora' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Built-in Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="clone" ${voiceType === 'clone' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Voice Clone</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="lora" ${voiceType === 'lora' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">LoRA Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="design" ${voiceType === 'design' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Voice Design</label>
                                    </div>
                                </div>

                                <!-- Custom Options -->
                                <div class="custom-opts" style="display: ${voiceType === 'custom' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select voice-select">
                                                ${AVAILABLE_VOICES.map(v => `<option value="${v}" ${config.voice === v ? 'selected' : ''}>${v}</option>`).join('')}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control character-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${config.character_style || config.default_style || ''}">
                                        </div>
                                    </div>
                                </div>

                                <!-- Built-in LoRA Options -->
                                <div class="builtin-lora-opts" style="display: ${voiceType === 'builtin_lora' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select builtin-lora-select">
                                                <option value="">-- Select built-in voice --</option>
                                                ${(() => {
                                                    const models = (window._loraModelsCache || []).filter(m => m.builtin);
                                                    const males = models.filter(m => m.gender === 'male');
                                                    const females = models.filter(m => m.gender === 'female');
                                                    let html = '';
                                                    if (males.length) {
                                                        html += '<optgroup label="Male">';
                                                        html += males.map(m => `<option value="${m.id}" ${config.adapter_id === m.id ? 'selected' : ''} ${m.downloaded === false ? 'disabled' : ''}>${m.name}${m.downloaded === false ? ' (not downloaded)' : ''} — ${m.description || ''}</option>`).join('');
                                                        html += '</optgroup>';
                                                    }
                                                    if (females.length) {
                                                        html += '<optgroup label="Female">';
                                                        html += females.map(m => `<option value="${m.id}" ${config.adapter_id === m.id ? 'selected' : ''} ${m.downloaded === false ? 'disabled' : ''}>${m.name}${m.downloaded === false ? ' (not downloaded)' : ''} — ${m.description || ''}</option>`).join('');
                                                        html += '</optgroup>';
                                                    }
                                                    return html;
                                                })()}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control builtin-lora-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${voiceType === 'builtin_lora' ? (config.character_style || '') : ''}">
                                        </div>
                                    </div>
                                    <small class="text-muted mt-1 d-block">Grayed-out voices need to be downloaded first. Go to the <strong>Training</strong> tab to download them.</small>
                                </div>

                                <!-- Clone Options -->
                                <div class="clone-opts" style="display: ${voiceType === 'clone' ? 'block' : 'none'}">
                                    <div class="row g-2 mb-2 align-items-center">
                                        <div class="col">
                                            <select class="form-select designed-voice-select" onchange="onDesignedVoiceSelect(this)">
                                                <option value="">-- Select voice or enter path manually --</option>
                                                ${(window._cloneVoicesCache || []).length ? `<optgroup label="Uploaded Voices">
                                                    ${(window._cloneVoicesCache || []).map(v => `<option value="clone:${v.id}" ${config.ref_audio && config.ref_audio.includes(v.filename) ? 'selected' : ''}>${v.name}</option>`).join('')}
                                                </optgroup>` : ''}
                                                ${(window._designedVoicesCache || []).length ? `<optgroup label="Designed Voices">
                                                    ${(window._designedVoicesCache || []).map(v => `<option value="design:${v.id}" ${config.ref_audio && config.ref_audio.includes(v.filename) ? 'selected' : ''}>${v.name}</option>`).join('')}
                                                </optgroup>` : ''}
                                                <option value="__manual__" ${config.ref_audio && !(window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) && !(window._designedVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) && config.ref_audio ? 'selected' : ''}>Custom path...</option>
                                            </select>
                                        </div>
                                        <div class="col-auto">
                                            <button class="btn btn-sm btn-outline-primary" onclick="uploadCloneVoice(this)" title="Upload audio file"><i class="fas fa-upload"></i> Upload</button>
                                            <input type="file" class="clone-voice-file-input" accept=".wav,.mp3,.flac,.ogg" style="display:none" onchange="handleCloneVoiceUpload(this)">
                                        </div>
                                    </div>
                                    <input type="text" class="form-control ref-text mb-2" placeholder="Reference Text" value="${config.ref_text || ''}">
                                    <div class="input-group">
                                        <input type="text" class="form-control ref-audio" placeholder="Path to audio file" value="${config.ref_audio || ''}" ${config.ref_audio && ((window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) || (window._designedVoicesCache || []).some(v => config.ref_audio.includes(v.filename))) ? 'readonly' : ''}>
                                        <button class="btn btn-sm btn-outline-secondary clone-play-btn" onclick="playCloneVoice(this)" title="Play reference audio" style="display:${config.ref_audio ? 'inline-block' : 'none'}"><i class="fas fa-play"></i></button>
                                        <button class="btn btn-sm btn-outline-danger clone-delete-btn" onclick="deleteCloneVoice(this)" title="Delete uploaded voice" style="display:${config.ref_audio && (window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) ? 'inline-block' : 'none'}"><i class="fas fa-trash"></i></button>
                                    </div>
                                </div>

                                <!-- LoRA Options -->
                                <div class="lora-opts" style="display: ${voiceType === 'lora' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select lora-adapter-select">
                                                <option value="">-- Select trained adapter --</option>
                                                ${(window._loraModelsCache || []).map(m => `<option value="${m.id}" ${config.adapter_id === m.id ? 'selected' : ''}>${m.name}</option>`).join('')}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control lora-character-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${voiceType === 'lora' ? (config.character_style || '') : ''}">
                                        </div>
                                    </div>
                                </div>

                                <!-- Voice Design Options -->
                                <div class="design-opts" style="display: ${voiceType === 'design' ? 'block' : 'none'}">
                                    <div class="input-group mb-2">
                                        <input type="text" class="form-control design-description" placeholder="Base voice description (e.g. Young strong soldier)" value="${config.description || ''}">
                                        <button class="btn btn-sm btn-outline-secondary design-suggest-btn" onclick="suggestVoiceDescription(this)">
                                            Suggest
                                        </button>
                                        <button class="btn btn-sm ${designLoaded ? 'btn-warning' : 'btn-outline-primary'} design-generate-btn" onclick="generateVoiceDesignClone(this)">
                                            ${designLoaded ? 'Retry' : 'Generate'}
                                        </button>
                                    </div>
                                    <div class="input-group mb-1">
                                        <input type="text" class="form-control design-sample-text" placeholder="Text sample used to create the reusable clone voice" value="${escapeHtml(designSampleText)}">
                                        <button class="btn btn-sm btn-outline-secondary design-play-btn" onclick="playVoiceDesignClone(this)" ${designLoaded ? '' : 'disabled'}>
                                            Play
                                        </button>
                                        <button class="btn btn-sm btn-outline-secondary design-download-btn" onclick="downloadVoiceDesignClone(this)" ${designLoaded ? '' : 'disabled'}>
                                            Download
                                        </button>
                                    </div>
                                    <input type="hidden" class="design-ref-audio" value="${config.ref_audio || ''}">
                                    <input type="hidden" class="design-generated-ref-text" value="${config.generated_ref_text || ''}">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        window.toggleVoiceType = (radio) => {
            const card = radio.closest('.card-body');
            card.querySelector('.custom-opts').style.display = radio.value === 'custom' ? 'block' : 'none';
            card.querySelector('.builtin-lora-opts').style.display = radio.value === 'builtin_lora' ? 'block' : 'none';
            card.querySelector('.clone-opts').style.display = radio.value === 'clone' ? 'block' : 'none';
            card.querySelector('.lora-opts').style.display = radio.value === 'lora' ? 'block' : 'none';
            card.querySelector('.design-opts').style.display = radio.value === 'design' ? 'block' : 'none';
            if (radio.value === 'design') {
                const voiceCard = radio.closest('.voice-card');
                const sampleInput = card.querySelector('.design-sample-text');
                if (sampleInput && !sampleInput.value.trim() && voiceCard?.dataset.suggestedSample) {
                    sampleInput.value = voiceCard.dataset.suggestedSample;
                }
            }
            debouncedSaveVoices();
        };

        async function loadVoices() {
            await loadVoiceSettings();
            // Refresh voice caches so dropdowns are populated
            try {
                window._designedVoicesCache = await API.get('/api/voice_design/list');
            } catch (e) { /* ignore if designer not available */ }
            try {
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
            } catch (e) { /* ignore if no uploads */ }
            try {
                window._loraModelsCache = await API.get('/api/lora/models');
            } catch (e) { /* ignore if no adapters */ }

            const voices = await API.get('/api/voices');
            window._narratingVoicesCache = voices
                .filter(v => (v.name || '').trim().toUpperCase() === 'NARRATOR' || v.config?.narrates === true)
                .map(v => v.name);
            const container = document.getElementById('voices-list');
            if (voices.length === 0) {
                container.innerHTML = '<div class="alert alert-info">No voices found. Generate a script first.</div>';
                layoutVoicesListContainer();
                return;
            }
            container.innerHTML = voices.map((v, i) => createVoiceCard(v, i)).join('');
            updateVoiceAliasStates();
            const suggestedAliases = suggestVoiceAliases(voices);
            layoutVoicesListContainer();

            let filledMissingDesignSample = false;
            voices.forEach(voice => {
                const config = voice.config || {};
                const voiceType = config.type || 'design';
                if (voiceType !== 'design') return;
                if ((config.ref_text || '').trim()) return;
                const card = Array.from(document.querySelectorAll('.voice-card')).find(el => el.dataset.voice === voice.name);
                const sampleInput = card?.querySelector('.design-sample-text');
                if (sampleInput && voice.suggested_sample_text) {
                    sampleInput.value = voice.suggested_sample_text;
                    filledMissingDesignSample = true;
                }
            });

            // If any voice has no saved config, or aliases were auto-suggested, save immediately
            if (suggestedAliases || filledMissingDesignSample || voices.some(v => !v.config || Object.keys(v.config).length === 0)) {
                clearTimeout(_voiceSaveTimer);
                saveVoicesNow({ promptConfirmation: false }).catch(() => {});
            }
        }

        function collectVoiceConfig() {
            const cards = document.querySelectorAll('.voice-card');
            const config = {};

            cards.forEach(card => {
                const name = card.dataset.voice;
                const type = card.querySelector('.voice-type:checked').value;
                const alias = (card.querySelector('.voice-alias-input')?.value || '').trim();
                const narratesEl = card.querySelector('.voice-narrates');
                const narrates = narratesEl ? narratesEl.checked : false;

                if (type === 'custom') {
                    config[name] = {
                        type: 'custom',
                        voice: card.querySelector('.voice-select').value,
                        character_style: card.querySelector('.character-style').value,
                        alias,
                        narrates,
                        seed: "-1"
                    };
                } else if (type === 'clone') {
                    config[name] = {
                        type: 'clone',
                        ref_text: card.querySelector('.ref-text').value,
                        ref_audio: card.querySelector('.ref-audio').value,
                        generated_ref_text: '',
                        alias,
                        narrates,
                        seed: "-1"
                    };
                } else if (type === 'builtin_lora') {
                    const adapterId = card.querySelector('.builtin-lora-select').value;
                    const adapterEntry = (window._loraModelsCache || []).find(m => m.id === adapterId);
                    config[name] = {
                        type: 'builtin_lora',
                        adapter_id: adapterId,
                        adapter_path: adapterEntry?.adapter_path || '',
                        character_style: card.querySelector('.builtin-lora-style').value,
                        alias,
                        narrates,
                        seed: "-1"
                    };
                } else if (type === 'lora') {
                    const adapterId = card.querySelector('.lora-adapter-select').value;
                    const adapterEntry = (window._loraModelsCache || []).find(m => m.id === adapterId);
                    config[name] = {
                        type: 'lora',
                        adapter_id: adapterId,
                        adapter_path: adapterEntry?.adapter_path || (adapterId ? `lora_models/${adapterId}` : ''),
                        character_style: card.querySelector('.lora-character-style').value,
                        alias,
                        narrates,
                        seed: "-1"
                    };
                } else if (type === 'design') {
                    config[name] = {
                        type: 'design',
                        description: card.querySelector('.design-description').value,
                        ref_text: card.querySelector('.design-sample-text').value,
                        ref_audio: card.querySelector('.design-ref-audio').value,
                        generated_ref_text: card.querySelector('.design-generated-ref-text').value,
                        alias,
                        narrates,
                        seed: "-1"
                    };
                }
            });
            return config;
        }

        let _voiceSaveTimer = null;
        let _voiceSaveInFlight = null;

        async function saveVoicesNow(options = {}) {
            const { promptConfirmation = true } = options;
            const cards = document.querySelectorAll('.voice-card');
            if (cards.length === 0) return;
            if (_voiceSaveInFlight) return _voiceSaveInFlight;

            const statusEl = document.getElementById('voice-save-status');
            statusEl.innerHTML = '<i class="fas fa-circle text-warning" style="font-size:0.5em;"></i> unsaved';

            _voiceSaveInFlight = (async () => {
                try {
                    const config = collectVoiceConfig();
                    let result = await API.post('/api/voices/save_config', {
                        config,
                        confirm_invalidation: false,
                    });

                    if (result.status === 'confirmation_required' && (result.invalidated_clips || 0) > 0) {
                        if (!promptConfirmation) {
                            statusEl.innerHTML = '<i class="fas fa-times text-danger me-1"></i>save cancelled';
                            throw new Error('Voice change cancelled');
                        }

                        const clipCount = Number(result.invalidated_clips || 0);
                        const confirmed = await showConfirm(
                            `This change will invalidate ${clipCount} generated clip${clipCount === 1 ? '' : 's'}. Continue?`
                        );
                        if (!confirmed) {
                            await loadVoices();
                            statusEl.innerHTML = '<i class="fas fa-times text-danger me-1"></i>save cancelled';
                            throw new Error('Voice change cancelled');
                        }

                        result = await API.post('/api/voices/save_config', {
                            config,
                            confirm_invalidation: true,
                        });

                        const invalidatedCount = Number(result.invalidated_clips || 0);
                        if (invalidatedCount > 0) {
                            showToast(
                                `Invalidated ${invalidatedCount} generated clip${invalidatedCount === 1 ? '' : 's'} for the new voice settings.`,
                                'warning',
                                5000
                            );
                        }
                    }

                    statusEl.innerHTML = '<i class="fas fa-check text-success me-1"></i>saved';
                    setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
                    window._narratingVoicesCache = null; // force refresh on next editor use
                    return result;
                } catch (e) {
                    if (!String(e?.message || '').toLowerCase().includes('cancelled')) {
                        const detail = String(e?.message || '').trim();
                        statusEl.innerHTML = `<i class="fas fa-times text-danger me-1"></i>save failed${detail ? `: ${detail}` : ''}`;
                    }
                    throw e;
                } finally {
                    _voiceSaveInFlight = null;
                }
            })();

            return _voiceSaveInFlight;
        }

        function debouncedSaveVoices() {
            clearTimeout(_voiceSaveTimer);
            _voiceSaveTimer = setTimeout(() => {
                saveVoicesNow().catch(() => {});
            }, 800);
        }

        window._sharedPreviewAudio = null;

        function stopSharedPreviewAudio() {
            if (!window._sharedPreviewAudio) return;
            try {
                window._sharedPreviewAudio.pause();
                window._sharedPreviewAudio.removeAttribute('src');
                window._sharedPreviewAudio.load();
            } catch (e) {
                console.warn('Could not stop preview audio cleanly', e);
            }
        }

        async function playSharedPreviewAudio(url) {
            if (!url) return;
            if (!window._sharedPreviewAudio) {
                window._sharedPreviewAudio = new Audio();
                window._sharedPreviewAudio.preload = 'auto';
            }

            stopSharedPreviewAudio();

            const audio = window._sharedPreviewAudio;
            audio.src = url;
            audio.currentTime = 0;
            audio.load();
            await audio.play();
        }

        function isPreviewAbortError(error) {
            if (!error) return false;
            return error.name === 'AbortError' || error.name === 'NotAllowedError' || String(error.message || '').toLowerCase().includes('aborted');
        }

        // Auto-save on any change inside the voices list
        document.getElementById('voices-list').addEventListener('change', (event) => {
            if (event.target.classList.contains('voice-alias-input')) {
                updateVoiceAliasStates();
            }
            debouncedSaveVoices();
        });
        document.getElementById('voices-list').addEventListener('input', (event) => {
            if (event.target.classList.contains('voice-alias-input')) {
                updateVoiceAliasStates();
                return;
            }
            if (event.target.classList.contains('ref-audio')) {
                return;
            }
            debouncedSaveVoices();
        });

        function syncDesignVoiceRow(card, { loaded = false, refAudio = '', generatedRefText = null } = {}) {
            if (!card) return;
            const generateBtn = card.querySelector('.design-generate-btn');
            const playBtn = card.querySelector('.design-play-btn');
            const downloadBtn = card.querySelector('.design-download-btn');
            const refAudioInput = card.querySelector('.design-ref-audio');
            const generatedRefTextInput = card.querySelector('.design-generated-ref-text');
            if (refAudioInput && typeof refAudio === 'string') {
                refAudioInput.value = refAudio;
            }
            if (generatedRefTextInput && typeof generatedRefText === 'string') {
                generatedRefTextInput.value = generatedRefText;
            }
            if (generateBtn) {
                generateBtn.className = `btn btn-sm ${loaded ? 'btn-warning' : 'btn-outline-primary'} design-generate-btn`;
                generateBtn.textContent = loaded ? 'Retry' : 'Generate';
                generateBtn.disabled = false;
            }
            if (playBtn) {
                playBtn.disabled = !loaded;
            }
            if (downloadBtn) {
                downloadBtn.disabled = !loaded;
            }
        }

        window.generateVoiceDesignClone = async (btn) => {
            const card = btn.closest('.card-body');
            const speaker = card.closest('.voice-card')?.dataset.voice;
            const description = card.querySelector('.design-description').value.trim();
            const sampleTextInput = card.querySelector('.design-sample-text');
            const loaded = Boolean((card.querySelector('.design-ref-audio')?.value || '').trim());

            if (!description) {
                showToast('Base voice description is required.', 'warning');
                return;
            }

            btn.disabled = true;
            btn.textContent = loaded ? 'Retrying...' : 'Generating...';
            try {
                if (!_bulkVoiceGenerationActive && window.setNavTaskSpinner) {
                    window.setNavTaskSpinner('voices');
                }
                let sampleText = sampleTextInput.value.trim();
                if (!sampleText) {
                    sampleText = card.closest('.voice-card')?.dataset.suggestedSample || '';
                    if (!sampleText) {
                        const voices = await API.get('/api/voices');
                        const voiceEntry = voices.find(v => normalizeVoiceName(v.name) === normalizeVoiceName(speaker));
                        sampleText = (voiceEntry?.suggested_sample_text || '').trim();
                    }
                    if (!sampleText) {
                        showToast(`No sample text available for ${speaker}.`, 'warning');
                        return;
                    }
                    sampleTextInput.value = sampleText;
                }
                await saveVoicesNow();

                const refreshedVoiceCard = getVoiceCardByName(speaker);
                const refreshedCard = refreshedVoiceCard?.querySelector('.card-body');
                const activeCard = refreshedCard || card;
                const freshSampleInput = activeCard.querySelector('.design-sample-text');
                const payload = {
                    speaker,
                    description: activeCard.querySelector('.design-description').value.trim(),
                    sample_text: freshSampleInput.value.trim(),
                    force: Boolean((activeCard.querySelector('.design-ref-audio')?.value || '').trim()),
                };
                const result = await API.post('/api/voices/design_generate', payload);
                if (freshSampleInput) {
                    freshSampleInput.value = result.ref_text || payload.sample_text;
                }
                syncDesignVoiceRow(activeCard, { loaded: true, refAudio: result.ref_audio, generatedRefText: result.generated_ref_text || '' });
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
                await saveVoicesNow();
                showToast(`Generated voice for ${speaker}.`, 'success');
            } catch (e) {
                if (!String(e?.message || '').toLowerCase().includes('cancelled')) {
                    showToast(`Voice generation failed: ${e.message}`, 'error');
                }
                syncDesignVoiceRow(card, { loaded: loaded, refAudio: card.querySelector('.design-ref-audio')?.value || '' });
            } finally {
                const activeCard = btn.closest('.card-body');
                const hasAudio = Boolean((activeCard.querySelector('.design-ref-audio')?.value || '').trim());
                syncDesignVoiceRow(activeCard, { loaded: hasAudio, refAudio: activeCard.querySelector('.design-ref-audio')?.value || '' });
                if (!_bulkVoiceGenerationActive && window.releaseNavTaskSpinner) {
                    window.releaseNavTaskSpinner('voices');
                }
            }
        };

        window.suggestVoiceDescription = async (btn) => {
            const card = btn.closest('.card-body');
            const voiceCard = btn.closest('.voice-card');
            const speaker = voiceCard?.dataset.voice;
            const descriptionInput = card.querySelector('.design-description');

            if (!speaker || !descriptionInput) {
                return '';
            }

            btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = 'Thinking...';

            try {
                const result = await API.post('/api/voices/suggest_description', { speaker });
                descriptionInput.value = result.voice || '';
                debouncedSaveVoices();
                showToast(`Suggested voice prompt for ${speaker}.`, 'success');
                return descriptionInput.value.trim();
            } catch (e) {
                showToast(`Voice prompt suggestion failed: ${e.message}`, 'error');
                throw e;
            } finally {
                btn.disabled = false;
                btn.textContent = originalText;
            }
        };

        async function suggestVoiceDescriptionsBulk(speakers, bulkBtn) {
            if (!Array.isArray(speakers) || speakers.length === 0) {
                return { results: [], failures: [] };
            }

            const result = await API.post('/api/voices/suggest_descriptions_bulk', { speakers });
            const results = Array.isArray(result?.results) ? result.results : [];
            const failures = Array.isArray(result?.failures) ? result.failures : [];

            results.forEach((item, index) => {
                const speaker = item?.speaker;
                const voiceCard = getVoiceCardByName(speaker);
                const descriptionInput = voiceCard?.querySelector('.design-description');
                if (!descriptionInput) return;
                if (bulkBtn) {
                    bulkBtn.textContent = `Suggesting ${index + 1}/${results.length}...`;
                }
                descriptionInput.value = item?.voice || '';
            });

            if (results.length > 0) {
                await saveVoicesNow();
            }

            return { results, failures };
        }

        window.playVoiceDesignClone = (btn) => {
            const card = btn.closest('.card-body');
            const refAudio = card.querySelector('.design-ref-audio')?.value;
            if (!refAudio) return;
            playSharedPreviewAudio(`/${refAudio}?t=${Date.now()}`).catch((e) => {
                if (isPreviewAbortError(e)) return;
                showToast(`Preview playback failed: ${e.message}`, 'error');
            });
        };

        window.downloadVoiceDesignClone = (btn) => {
            const card = btn.closest('.card-body');
            const voiceCard = btn.closest('.voice-card');
            const refAudio = card.querySelector('.design-ref-audio')?.value;
            if (!refAudio) {
                showToast('No reusable voice sample is available yet.', 'warning');
                return;
            }

            const speaker = (voiceCard?.dataset.voice || 'voice-sample').trim();
            const extensionMatch = refAudio.match(/(\.[a-z0-9]+)$/i);
            const extension = extensionMatch ? extensionMatch[1] : '.wav';
            const safeSpeaker = speaker.replace(/[^a-z0-9._-]+/gi, '_');
            const link = document.createElement('a');
            link.href = `/${refAudio}?t=${Date.now()}`;
            link.download = `${safeSpeaker}${extension}`;
            document.body.appendChild(link);
            link.click();
            link.remove();
        };

        window.generateOutstandingVoices = async () => {
            const bulkBtn = document.getElementById('generate-outstanding-voices-btn');

            // Convert any "Custom Voice" cards to "Voice Design" so they get voices generated
            Array.from(document.querySelectorAll('.voice-card'))
                .filter(card => !card.classList.contains('alias-active') && !card.classList.contains('narrator-threshold-active'))
                .filter(card => card.querySelector('.voice-type:checked')?.value === 'custom')
                .forEach(card => {
                    const designRadio = card.querySelector('.voice-type[value="design"]');
                    if (designRadio) {
                        designRadio.checked = true;
                        window.toggleVoiceType(designRadio);
                    }
                });

            const eligibleSpeakers = Array.from(document.querySelectorAll('.voice-card'))
                .filter(card => !card.classList.contains('alias-active') && !card.classList.contains('narrator-threshold-active'))
                .filter(card => card.querySelector('.voice-type:checked')?.value === 'design')
                .map(card => card.dataset.voice)
                .filter(Boolean);

            if (eligibleSpeakers.length === 0) {
                showToast('No voice-design rows are available to create.', 'info');
                return;
            }

            bulkBtn.disabled = true;
            const originalText = bulkBtn.textContent;
            _bulkVoiceGenerationActive = true;
            try {
                if (window.setNavTaskSpinner) {
                    window.setNavTaskSpinner('voices');
                }
                let generatedCount = 0;
                const failedSpeakers = new Map();
                const generationQueue = [];

                for (let i = 0; i < eligibleSpeakers.length; i += 1) {
                    const speaker = eligibleSpeakers[i];
                    const voiceCard = getVoiceCardByName(speaker);
                    if (!voiceCard || voiceCard.classList.contains('alias-active') || voiceCard.classList.contains('narrator-threshold-active')) {
                        continue;
                    }

                    const card = voiceCard.querySelector('.card-body');
                    const suggestButton = card?.querySelector('.design-suggest-btn');
                    const generateButton = card?.querySelector('.design-generate-btn');
                    const descriptionInput = card.querySelector('.design-description');
                    const sampleInput = card.querySelector('.design-sample-text');
                    const refAudioInput = card.querySelector('.design-ref-audio');
                    const hasAudio = Boolean((refAudioInput?.value || '').trim());

                    if (hasAudio) {
                        continue;
                    }

                    generationQueue.push({ speaker, voiceCard, card, descriptionInput, sampleInput, generateButton });
                }

                const suggestionQueue = generationQueue.filter(item => !item.descriptionInput?.value.trim());
                if (suggestionQueue.length > 0) {
                    bulkBtn.textContent = `Suggesting 0/${suggestionQueue.length}...`;
                    try {
                        const batchResult = await suggestVoiceDescriptionsBulk(
                            suggestionQueue.map(item => item.speaker),
                            bulkBtn,
                        );
                        batchResult.failures.forEach(({ speaker, error }) => {
                            failedSpeakers.set(speaker, `${speaker} (${error || 'suggestion failed'})`);
                        });
                    } catch (e) {
                        suggestionQueue.forEach(({ speaker }) => {
                            failedSpeakers.set(speaker, `${speaker} (${e.message || 'suggestion failed'})`);
                        });
                    }
                }

                for (let i = 0; i < generationQueue.length; i += 1) {
                    const { speaker, voiceCard, card, descriptionInput, sampleInput, generateButton } = generationQueue[i];
                    try {
                        if (sampleInput && !sampleInput.value.trim()) {
                            sampleInput.value = voiceCard.dataset.suggestedSample || '';
                            if (sampleInput.value.trim()) {
                                debouncedSaveVoices();
                            }
                        }

                        if (!descriptionInput?.value.trim() || !generateButton) {
                            if (!failedSpeakers.has(speaker)) {
                                failedSpeakers.set(speaker, `${speaker} (missing description)`);
                            }
                            continue;
                        }

                        bulkBtn.textContent = `Generating ${i + 1}/${generationQueue.length}...`;
                        await window.generateVoiceDesignClone(generateButton);
                        const nowHasAudio = Boolean((card.querySelector('.design-ref-audio')?.value || '').trim());
                        if (nowHasAudio) {
                            generatedCount += 1;
                        } else {
                            failedSpeakers.set(speaker, `${speaker} (generation failed)`);
                        }
                    } catch (e) {
                        failedSpeakers.set(speaker, `${speaker} (${e.message || 'unknown error'})`);
                    }
                }
                const failureList = Array.from(failedSpeakers.values());
                if (generatedCount === 0 && failureList.length === 0) {
                    showToast('No outstanding voices needed generation.', 'info');
                } else if (failureList.length > 0) {
                    const summary = failureList.slice(0, 4).join(', ');
                    const suffix = failureList.length > 4 ? `, and ${failureList.length - 4} more` : '';
                    showToast(`Created ${generatedCount} voice${generatedCount === 1 ? '' : 's'}; failed: ${summary}${suffix}.`, 'warning');
                } else {
                    showToast(`Created ${generatedCount} outstanding voice${generatedCount === 1 ? '' : 's'}.`, 'success');
                }
            } finally {
                try {
                    await API.post('/api/voices/unload_bulk_generation', {});
                } catch (e) {
                    console.warn('Failed to unload bulk voice generation state', e);
                }
                _bulkVoiceGenerationActive = false;
                bulkBtn.disabled = false;
                bulkBtn.textContent = originalText;
                if (window.releaseNavTaskSpinner) {
                    window.releaseNavTaskSpinner('voices');
                }
            }
        };

        window.clearOutstandingVoices = async () => {
            const clearBtn = document.getElementById('clear-outstanding-voices-btn');
            const confirmed = await showConfirm(
                'Delete all uploaded reusable voices tied to this script context and clear those references from the current voice cards?'
            );
            if (!confirmed) return;

            clearBtn.disabled = true;
            const originalText = clearBtn.textContent;
            clearBtn.textContent = 'Clearing...';
            try {
                const result = await API.post('/api/voices/clear_uploaded', {});
                await loadVoices();
                showToast(
                    `Cleared ${result.removed_manifest_entries || 0} voice entr${(result.removed_manifest_entries || 0) === 1 ? 'y' : 'ies'} for "${result.script_title || 'Project'}".`,
                    'success'
                );
            } catch (e) {
                showToast(`Clear failed: ${e.message}`, 'error');
            } finally {
                clearBtn.disabled = false;
                clearBtn.textContent = originalText;
            }
        };

        const narratorThresholdInput = document.getElementById('narrator-threshold-input');
        if (narratorThresholdInput) {
            narratorThresholdInput.addEventListener('input', () => {
                updateVoiceAliasStates();
                debouncedSaveVoiceSettings();
            });
            narratorThresholdInput.addEventListener('change', () => {
                updateVoiceAliasStates();
                debouncedSaveVoiceSettings();
            });
        }

        if (!_voicesListResizeBound) {
            window.addEventListener('resize', () => {
                layoutVoicesListContainer();
            });
            _voicesListResizeBound = true;
        }
