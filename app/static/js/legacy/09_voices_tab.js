        // --- Voices Tab ---
        const AVAILABLE_VOICES = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"];

        function normalizeVoiceName(value) {
            return (value || '').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
        }

        let _voiceSettingsSaveTimer = null;
        let _voicesListResizeBound = false;
        let _bulkVoiceGenerationActive = false;
        let _voiceSaveRetryTimer = null;
        let _voicePageUnloading = false;
        let _voiceAliasesPrimedForScript = false;
        let _voiceSaveFlushPromise = null;
        let _voiceSavePending = false;
        let _loadVoicesInFlight = null;
        let _voicesRenderSignature = '';
        let _dirtyVoiceNames = new Set();
        let _voiceSavePendingOptions = {
            promptConfirmation: false,
            retryOnNetworkFailure: false,
            includeAll: false,
            speakerNames: [],
        };

        function isVoiceSaveNetworkError(error) {
            const message = String(error?.message || '').toLowerCase();
            return message.includes('failed to fetch')
                || message.includes('networkerror')
                || message.includes('load failed')
                || message.includes('network request failed');
        }

        function scheduleVoiceSaveRetry(options = {}) {
            clearTimeout(_voiceSaveRetryTimer);
            _voiceSaveRetryTimer = setTimeout(() => {
                saveVoicesNow(options).catch(() => {});
            }, 1500);
        }

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
            if (!input) return 0;
            const parsed = parseInt(input.value, 10);
            return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
        }

        function getNarratorName(cards) {
            const narratorCard = cards.find(card => normalizeVoiceName(card.dataset.voice) === normalizeVoiceName('NARRATOR'));
            if (!narratorCard) return '';
            const narratesInput = narratorCard.querySelector('.voice-narrates');
            if (narratesInput && !narratesInput.checked) return '';
            return narratorCard.dataset.voice;
        }

        function captureVoicesListScrollState(container) {
            return {
                listScrollTop: container ? Number(container.scrollTop) || 0 : 0,
                windowScrollY: typeof window !== 'undefined' ? (Number(window.scrollY) || 0) : 0,
            };
        }

        function restoreVoicesListScrollState(container, scrollState) {
            if (!container || !scrollState) return;
            const restore = () => {
                const maxScrollTop = Math.max(0, (Number(container.scrollHeight) || 0) - (Number(container.clientHeight) || 0));
                container.scrollTop = Math.max(0, Math.min(Number(scrollState.listScrollTop) || 0, maxScrollTop));
                if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
                    window.scrollTo(0, Number(scrollState.windowScrollY) || 0);
                }
            };
            if (typeof requestAnimationFrame === 'function') {
                requestAnimationFrame(restore);
            } else {
                setTimeout(restore, 0);
            }
        }

        function buildVoicesRenderSignature(voices, extras = {}) {
            return JSON.stringify({
                narratorThreshold: Number(extras.narratorThreshold) || 0,
                voices: Array.isArray(voices) ? voices : [],
                designedVoices: Array.isArray(extras.designedVoices) ? extras.designedVoices : [],
                cloneVoices: Array.isArray(extras.cloneVoices) ? extras.cloneVoices : [],
                loraModels: Array.isArray(extras.loraModels) ? extras.loraModels : [],
            });
        }

        function isNarratorVoiceCard(card) {
            return normalizeVoiceName(card?.dataset?.voice) === normalizeVoiceName('NARRATOR');
        }

        async function loadVoiceSettings() {
            const input = document.getElementById('narrator-threshold-input');
            if (!input) return;
            try {
                const result = await API.get('/api/voices/settings');
                const value = Number(result?.narrator_threshold);
                input.value = Number.isFinite(value) && value >= 0 ? String(value) : '0';
            } catch (_e) {
                input.value = input.value || '0';
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
                    return loadVoices();
                }).catch((e) => {
                    showToast(`Failed to save narrator threshold: ${e.message}`, 'error');
                });
            }, 500);
        }

        function getVoiceAliasWeight(voiceLike) {
            const lineCount = Number(voiceLike?.line_count ?? voiceLike?.lineCount ?? 0);
            if (Number.isFinite(lineCount) && lineCount > 0) return lineCount;
            const paragraphCount = Number(voiceLike?.paragraph_count ?? voiceLike?.paragraphCount ?? 0);
            if (Number.isFinite(paragraphCount) && paragraphCount > 0) return paragraphCount;
            return 0;
        }

        function nameTokens(value) {
            return normalizeVoiceName(value).split(' ').filter(Boolean);
        }

        function containsNameTokens(containerName, candidateName) {
            const containerTokens = nameTokens(containerName);
            const candidateTokens = nameTokens(candidateName);
            if (!candidateTokens.length || candidateTokens.length > containerTokens.length) {
                return false;
            }
            for (let i = 0; i <= containerTokens.length - candidateTokens.length; i += 1) {
                const windowTokens = containerTokens.slice(i, i + candidateTokens.length);
                if (windowTokens.every((token, index) => token === candidateTokens[index])) {
                    return true;
                }
            }
            return false;
        }

        function buildVoiceAliasCandidatesFromCards(cards) {
            return cards.map(card => ({
                name: card.dataset.voice,
                lineCount: Number(card.dataset.lineCount || 0),
                paragraphCount: Number(card.dataset.paragraphCount || 0),
            }));
        }

        // Suggest aliases for names where one is a substring of another.
        // The name with fewer lines gets aliased to the one with more.
        // Falls back to paragraph count when line counts are unavailable.
        // Only fills empty AKA inputs — never overwrites a user-set alias.
        function suggestVoiceAliases(voices) {
            const hasCounts = voices.some(v => getVoiceAliasWeight(v) > 0);
            if (!hasCounts) return false;

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

                    // Check if one full name appears as whole tokens within the other
                    const aInB = containsNameTokens(other.name, voice.name);
                    const bInA = containsNameTokens(voice.name, other.name);
                    if (!aInB && !bInA) continue;

                    // The shorter/contained name is the candidate to alias;
                    // the one with more lines is the target.
                    const countA = getVoiceAliasWeight(voice);
                    const countB = getVoiceAliasWeight(other);

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
                    markVoiceDirty(card);
                    anySet = true;
                }
            }

            if (anySet) {
                updateVoiceAliasStates();
            }
            return anySet;
        }

        async function refreshAutomaticVoiceAliases(options = {}) {
            const { save = true } = options;
            const cards = Array.from(document.querySelectorAll('.voice-card'));
            if (cards.length === 0) return false;
            const anySet = suggestVoiceAliases(buildVoiceAliasCandidatesFromCards(cards));
            if (anySet && save) {
                await saveVoicesNow({ promptConfirmation: false, retryOnNetworkFailure: true });
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
                const storedAutoAliasTarget = (!target && normalizeVoiceName(card.dataset.autoAliasTarget) !== normalizeVoiceName(card.dataset.voice))
                    ? String(card.dataset.autoAliasTarget || '').trim()
                    : '';
                const computedThresholdTarget = (!target && narratorName && !isNarrator && lineCount < narratorThreshold)
                    ? narratorName
                    : '';
                const thresholdTarget = computedThresholdTarget || storedAutoAliasTarget;
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

        function getVoiceLineCount(card) {
            return Number(card?.dataset?.lineCount || 0);
        }

        function isZeroLineVoiceCard(card) {
            return getVoiceLineCount(card) <= 0;
        }

        function createVoiceCard(voice, index) {
            const config = voice.config || {};
            const voiceType = config.type || 'design';
            const safeName = escapeHtml(voice.name);
            const safeAlias = escapeHtml(config.alias || '');
            const safeAutoAliasTarget = escapeHtml(voice.auto_alias_target || '');
            const designSampleText = config.ref_text || voice.suggested_sample_text || '';
            const designLoaded = Boolean((config.ref_audio || '').trim()) && !!voice.design_clone_loaded;
            const lineCount = Number(voice.line_count || 0);
            const lineLabel = `${lineCount} ${lineCount === 1 ? 'line' : 'lines'}`;
            const userCreated = config.user_created === true;
            const isNarratorCard = normalizeVoiceName(voice.name) === normalizeVoiceName('NARRATOR');
            const narratesChecked = isNarratorCard ? config.narrates !== false : config.narrates === true;

            return `
                <div class="card voice-card mb-3" data-voice="${safeName}" data-line-count="${lineCount}" data-paragraph-count="${Number(voice.paragraph_count || 0)}" data-suggested-sample="${escapeHtml(voice.suggested_sample_text || '')}" data-auto-alias-target="${safeAutoAliasTarget}" data-user-created="${userCreated ? 'true' : 'false'}">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <h5 class="card-title voice-primary-content d-flex flex-wrap align-items-center gap-2">
                                    <span>${safeName}</span>
                                    <span class="badge text-bg-secondary" style="cursor:pointer;" title="Jump to first line in Editor" onclick="jumpToFirstChunkForVoice('${safeName}')">${lineLabel}</span>
                                    <div class="form-check form-check-inline mb-0">
                                        <input class="form-check-input voice-narrates" type="checkbox" id="narrates_${index}" ${narratesChecked ? 'checked' : ''}>
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
                                            <button class="btn btn-sm btn-outline-primary clone-action-btn" onclick="onCloneAction(this)" title="Upload audio file"><i class="fas fa-upload"></i> Upload</button>
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

        window.addVoiceCharacter = async () => {
            const rawName = typeof window.prompt === 'function'
                ? window.prompt('Character name:')
                : '';
            if (rawName === null) {
                return;
            }
            const speaker = String(rawName || '').trim().replace(/\s+/g, ' ');
            if (!speaker) {
                showToast('Character name is required.', 'warning');
                return;
            }
            if (getVoiceCardByName(speaker)) {
                showToast(`Character "${speaker}" already exists.`, 'warning');
                return;
            }

            const container = document.getElementById('voices-list');
            if (container && typeof container.insertAdjacentHTML === 'function') {
                const existingCards = Array.from(document.querySelectorAll('.voice-card'));
                if (existingCards.length === 0) {
                    container.innerHTML = '';
                }
                container.insertAdjacentHTML(
                    'beforeend',
                    createVoiceCard(
                        {
                            name: speaker,
                            config: {
                                type: 'design',
                                description: '',
                                ref_text: '',
                                ref_audio: '',
                                generated_ref_text: '',
                                alias: '',
                                narrates: false,
                                user_created: true,
                            },
                            suggested_sample_text: '',
                            design_clone_loaded: false,
                            line_count: 0,
                            paragraph_count: 0,
                            auto_narrator_alias: false,
                            auto_alias_target: '',
                        },
                        existingCards.length,
                    ),
                );
                _voicesRenderSignature = '';
                updateVoiceAliasStates();
                layoutVoicesListContainer();
            }

            try {
                const config = collectVoiceConfig();
                const existing = config[speaker] || {};
                config[speaker] = {
                    type: existing.type || 'design',
                    description: existing.description || '',
                    ref_text: existing.ref_text || '',
                    ref_audio: existing.ref_audio || '',
                    generated_ref_text: existing.generated_ref_text || '',
                    alias: existing.alias || '',
                    narrates: existing.narrates === true,
                    seed: existing.seed || '-1',
                    user_created: true,
                };
                await API.post('/api/voices/batch', {
                    config,
                    confirm_invalidation: false,
                });
                showToast(`Added ${speaker}.`, 'success');
            } catch (e) {
                showToast(`Failed to add character: ${e.message}`, 'error');
            } finally {
                try {
                    await loadVoices();
                } catch (_e) {
                    // Ignore refresh errors; save failure already surfaced above.
                }
            }
        };

        async function loadVoices() {
            if (_loadVoicesInFlight) {
                return _loadVoicesInFlight;
            }

            const run = (async () => {
                const container = document.getElementById('voices-list');
                if (!container) return;
                const scrollState = captureVoicesListScrollState(container);

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
                    .filter(v => v.config?.narrates === true)
                    .map(v => v.name);

                const renderSignature = buildVoicesRenderSignature(voices, {
                    narratorThreshold: getNarratorThresholdValue(),
                    designedVoices: window._designedVoicesCache,
                    cloneVoices: window._cloneVoicesCache,
                    loraModels: window._loraModelsCache,
                });

                if (voices.length === 0) {
                    const emptyMarkup = '<div class="alert alert-info">No voices found. Generate a script first.</div>';
                    if (container.innerHTML !== emptyMarkup) {
                        container.innerHTML = emptyMarkup;
                    }
                    _voicesRenderSignature = renderSignature;
                    layoutVoicesListContainer();
                    restoreVoicesListScrollState(container, scrollState);
                    return;
                }

                if (_voicesRenderSignature === renderSignature && container.innerHTML) {
                    layoutVoicesListContainer();
                    restoreVoicesListScrollState(container, scrollState);
                    return;
                }

                container.innerHTML = voices.map((v, i) => createVoiceCard(v, i)).join('');
                _voicesRenderSignature = renderSignature;
                _dirtyVoiceNames.clear();
                updateVoiceAliasStates();
                layoutVoicesListContainer();

                voices.forEach(voice => {
                    const config = voice.config || {};
                    const voiceType = config.type || 'design';
                    if (voiceType !== 'design') return;
                    if ((config.ref_text || '').trim()) return;
                    const card = Array.from(document.querySelectorAll('.voice-card')).find(el => el.dataset.voice === voice.name);
                    const sampleInput = card?.querySelector('.design-sample-text');
                    if (sampleInput && voice.suggested_sample_text) {
                        sampleInput.value = voice.suggested_sample_text;
                    }
                });

                document.querySelectorAll('.voice-card').forEach((card) => {
                    const cardType = card.querySelector('.voice-type:checked')?.value || '';
                    if (cardType === 'clone' && typeof window.updateCloneActionButtonForCard === 'function') {
                        window.updateCloneActionButtonForCard(card);
                    }
                });

                restoreVoicesListScrollState(container, scrollState);
            })();

            _loadVoicesInFlight = run;
            try {
                return await run;
            } finally {
                if (_loadVoicesInFlight === run) {
                    _loadVoicesInFlight = null;
                }
            }
        }

        function collectVoiceConfig(cardsInput = null) {
            const cards = cardsInput || document.querySelectorAll('.voice-card');
            const config = {};

            cards.forEach(card => {
                const name = card.dataset.voice;
                if (!name) return;
                const type = card.querySelector('.voice-type:checked').value;
                const alias = (card.querySelector('.voice-alias-input')?.value || '').trim();
                const narratesEl = card.querySelector('.voice-narrates');
                const narrates = narratesEl ? narratesEl.checked : false;
                const userCreated = card.dataset.userCreated === 'true';

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
                if (config[name] && userCreated) {
                    config[name].user_created = true;
                }
            });
            return config;
        }

        function getVoiceCardName(card) {
            return (card?.dataset?.voice || '').trim();
        }

        function markVoiceDirty(cardOrName) {
            const name = typeof cardOrName === 'string'
                ? cardOrName.trim()
                : getVoiceCardName(cardOrName);
            if (!name) return;
            _dirtyVoiceNames.add(name);
        }

        function getVoiceCardsForSave(options = {}) {
            const allCards = Array.from(document.querySelectorAll('.voice-card'));
            if (options.includeAll) {
                return allCards;
            }
            if (allCards.length === 0) {
                return [];
            }
            const dirtyNames = Array.from(_dirtyVoiceNames)
                .map(name => String(name || '').trim())
                .filter(Boolean);
            if (dirtyNames.length === 0 && (!options.speakerNames || options.speakerNames.length === 0)) {
                return allCards;
            }
            const effectiveNames = new Set(dirtyNames);
            for (const name of (options.speakerNames || [])) {
                const normalized = String(name || '').trim();
                if (normalized) {
                    effectiveNames.add(normalized);
                }
            }
            if (effectiveNames.size === 0) {
                return Array.from(document.querySelectorAll('.voice-card'));
            }
            return allCards
                .filter(card => effectiveNames.has(getVoiceCardName(card)));
        }

        let _voiceSaveTimer = null;
        let _voiceSaveInFlight = null;

        function mergeVoiceSaveOptions(options = {}) {
            const speakerNames = Array.from(new Set(
                Array.from(options.speakerNames || [])
                    .map(name => String(name || '').trim())
                    .filter(Boolean)
            ));
            return {
                promptConfirmation: Boolean(options.promptConfirmation),
                retryOnNetworkFailure: Boolean(options.retryOnNetworkFailure),
                includeAll: Boolean(options.includeAll),
                speakerNames,
            };
        }

        function queueVoiceSaveOptions(options = {}) {
            const merged = mergeVoiceSaveOptions(options);
            _voiceSavePending = true;
            _voiceSavePendingOptions = {
                promptConfirmation: _voiceSavePendingOptions.promptConfirmation || merged.promptConfirmation,
                retryOnNetworkFailure: _voiceSavePendingOptions.retryOnNetworkFailure || merged.retryOnNetworkFailure,
                includeAll: _voiceSavePendingOptions.includeAll || merged.includeAll,
                speakerNames: Array.from(new Set([
                    ...(_voiceSavePendingOptions.includeAll ? [] : _voiceSavePendingOptions.speakerNames),
                    ...(merged.includeAll ? [] : merged.speakerNames),
                ])),
            };
        }

        async function performVoiceSave(options = {}) {
            const { promptConfirmation = false, retryOnNetworkFailure = false } = options;
            const cards = getVoiceCardsForSave(options);
            if (cards.length === 0) return;

            const statusEl = document.getElementById('voice-save-status');
            statusEl.innerHTML = '<i class="fas fa-circle text-warning" style="font-size:0.5em;"></i> unsaved';
            const savedNames = cards.map(getVoiceCardName).filter(Boolean);

            _voiceSaveInFlight = (async () => {
                try {
                    const config = collectVoiceConfig(cards);
                    let result = await API.post('/api/voices/batch', {
                        config,
                        confirm_invalidation: false,
                    });

                    if (result.status === 'confirmation_required' && (result.invalidated_clips || 0) > 0) {
                        if (!promptConfirmation) {
                            statusEl.innerHTML = '<i class="fas fa-circle text-warning" style="font-size:0.5em;"></i> unsaved';
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

                        result = await API.post('/api/voices/batch', {
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

                    savedNames.forEach(name => _dirtyVoiceNames.delete(name));
                    statusEl.innerHTML = '<i class="fas fa-check text-success me-1"></i>saved';
                    setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
                    window._narratingVoicesCache = null; // force refresh on next editor use
                    return result;
                } catch (e) {
                    if (retryOnNetworkFailure && isVoiceSaveNetworkError(e)) {
                        if (!document.hidden && !_voicePageUnloading) {
                            statusEl.innerHTML = '<i class="fas fa-circle-notch fa-spin text-warning me-1"></i>waiting for reconnect';
                        }
                        scheduleVoiceSaveRetry({ promptConfirmation, retryOnNetworkFailure: true });
                        return { status: 'retry_scheduled' };
                    }
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

        async function saveVoicesNow(options = {}) {
            const cards = document.querySelectorAll('.voice-card');
            if (cards.length === 0) return;

            queueVoiceSaveOptions(options);
            if (_voiceSaveFlushPromise) {
                return _voiceSaveFlushPromise;
            }

            _voiceSaveFlushPromise = (async () => {
                try {
                    while (_voiceSavePending) {
                        const nextOptions = { ..._voiceSavePendingOptions };
                        _voiceSavePending = false;
                        _voiceSavePendingOptions = {
                            promptConfirmation: false,
                            retryOnNetworkFailure: false,
                            includeAll: false,
                            speakerNames: [],
                        };
                        await performVoiceSave(nextOptions);
                    }
                } finally {
                    _voiceSaveFlushPromise = null;
                }
            })();

            return _voiceSaveFlushPromise;
        }

        async function disableNarratorNarration(card, checkbox) {
            const statusEl = document.getElementById('voice-save-status');
            const previousChecked = true;
            const narratorName = getVoiceCardName(card);
            checkbox.disabled = true;
            statusEl.innerHTML = '<i class="fas fa-circle-notch fa-spin text-warning me-1"></i>saving';
            try {
                const config = collectVoiceConfig();
                const result = await API.post('/api/voices/narrator/disable', { config });
                _dirtyVoiceNames.clear();
                window._narratingVoicesCache = null;
                if (typeof syncNarratorSelectionsFromBackend === 'function') {
                    await syncNarratorSelectionsFromBackend();
                }
                await loadVoices();
                const chapterCount = Number(result?.changed_chapters || 0);
                const invalidatedCount = Number(result?.invalidated_clips || 0);
                showToast(
                    `Updated ${chapterCount} chapter narrator${chapterCount === 1 ? '' : 's'} and invalidated ${invalidatedCount} clip${invalidatedCount === 1 ? '' : 's'}.`,
                    'warning',
                    6000
                );
                statusEl.innerHTML = '<i class="fas fa-check text-success me-1"></i>saved';
                setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
                return result;
            } catch (e) {
                checkbox.checked = previousChecked;
                _dirtyVoiceNames.delete(narratorName);
                if (e?.detail?.code === 'narrator_disable_requires_other_narrator') {
                    statusEl.innerHTML = '';
                    showToast('Enable narration on another character before disabling the narrator.', 'warning', 5000);
                    return null;
                }
                statusEl.innerHTML = `<i class="fas fa-times text-danger me-1"></i>save failed${e?.message ? `: ${e.message}` : ''}`;
                showToast(`Failed to disable narrator narration: ${e.message}`, 'error');
                throw e;
            } finally {
                checkbox.disabled = false;
            }
        }

        function debouncedSaveVoices(options = {}) {
            clearTimeout(_voiceSaveTimer);
            queueVoiceSaveOptions({
                ...options,
                retryOnNetworkFailure: true,
            });
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
            const voiceCard = event.target.closest('.voice-card');
            if (voiceCard) {
                markVoiceDirty(voiceCard);
            }
            if (event.target.classList.contains('voice-narrates') && isNarratorVoiceCard(voiceCard) && !event.target.checked) {
                disableNarratorNarration(voiceCard, event.target).catch(() => {});
                return;
            }
            if (event.target.classList.contains('voice-alias-input')) {
                updateVoiceAliasStates();
                debouncedSaveVoices({
                    promptConfirmation: true,
                    speakerNames: voiceCard ? [getVoiceCardName(voiceCard)] : [],
                });
                return;
            }
            debouncedSaveVoices();
        });
        document.getElementById('voices-list').addEventListener('input', (event) => {
            const voiceCard = event.target.closest('.voice-card');
            if (voiceCard) {
                markVoiceDirty(voiceCard);
            }
            if (event.target.classList.contains('voice-narrates') && isNarratorVoiceCard(voiceCard) && !event.target.checked) {
                return;
            }
            if (event.target.classList.contains('voice-alias-input')) {
                updateVoiceAliasStates();
                debouncedSaveVoices({
                    promptConfirmation: true,
                    speakerNames: voiceCard ? [getVoiceCardName(voiceCard)] : [],
                });
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
                await saveVoicesNow({ promptConfirmation: true, speakerNames: [speaker] });

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
                markVoiceDirty(activeCard.closest('.voice-card') || speaker);
                await saveVoicesNow({ promptConfirmation: true, speakerNames: [speaker] });
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
            if (isZeroLineVoiceCard(voiceCard)) {
                showToast(`Skipping ${speaker}: 0 lines.`, 'info');
                return '';
            }

            btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = 'Thinking...';

            try {
                const result = await API.post('/api/voices/suggest_description', { speaker });
                descriptionInput.value = result.voice || '';
                markVoiceDirty(voiceCard);
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

            clearTimeout(_voiceSaveTimer);

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
                results.forEach(item => {
                    const voiceCard = getVoiceCardByName(item?.speaker);
                    if (voiceCard) {
                        markVoiceDirty(voiceCard);
                    }
                });
                await saveVoicesNow({
                    promptConfirmation: false,
                    retryOnNetworkFailure: true,
                    speakerNames: results.map(item => item?.speaker).filter(Boolean),
                });
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

            await refreshAutomaticVoiceAliases();

            // Convert any "Custom Voice" cards to "Voice Design" so they get voices generated
            Array.from(document.querySelectorAll('.voice-card'))
                .filter(card => !card.classList.contains('alias-active') && !card.classList.contains('narrator-threshold-active'))
                .filter(card => !isZeroLineVoiceCard(card))
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
                .filter(card => !isZeroLineVoiceCard(card))
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
                    if (isZeroLineVoiceCard(voiceCard)) {
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

                    generationQueue.push({ speaker });
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
                    const { speaker } = generationQueue[i];
                    try {
                        const voiceCard = getVoiceCardByName(speaker);
                        if (!voiceCard || voiceCard.classList.contains('alias-active') || voiceCard.classList.contains('narrator-threshold-active')) {
                            if (!failedSpeakers.has(speaker)) {
                                failedSpeakers.set(speaker, `${speaker} (no longer eligible)`);
                            }
                            continue;
                        }

                        const card = voiceCard.querySelector('.card-body');
                        const descriptionInput = card?.querySelector('.design-description');
                        const sampleInput = card?.querySelector('.design-sample-text');
                        const generateButton = card?.querySelector('.design-generate-btn');
                        if (sampleInput && !sampleInput.value.trim()) {
                            sampleInput.value = voiceCard.dataset.suggestedSample || '';
                            if (sampleInput.value.trim()) {
                                markVoiceDirty(voiceCard);
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

        function flushPendingVoiceSavesOnUnload() {
            clearTimeout(_voiceSaveTimer);
            const pendingSpeakerNames = Array.from(new Set([
                ...Array.from(_dirtyVoiceNames),
                ...Array.from(_voiceSavePendingOptions.speakerNames || []),
            ]));
            const includeAll = Boolean(_voiceSavePendingOptions.includeAll);
            const promptConfirmation = Boolean(_voiceSavePendingOptions.promptConfirmation);
            if (!includeAll && pendingSpeakerNames.length === 0) {
                return;
            }
            const cards = getVoiceCardsForSave({
                includeAll,
                speakerNames: pendingSpeakerNames,
            });
            if (cards.length === 0) {
                return;
            }
            const payload = JSON.stringify({
                config: collectVoiceConfig(cards),
                confirm_invalidation: promptConfirmation,
            });
            try {
                if (typeof navigator !== 'undefined' && navigator.sendBeacon) {
                    const blob = new Blob([payload], { type: 'application/json' });
                    navigator.sendBeacon('/api/voices/batch', blob);
                    return;
                }
            } catch (_e) {
                // Fall back to keepalive fetch below.
            }
            try {
                fetch('/api/voices/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload,
                    keepalive: true,
                });
            } catch (_e) {
                // Ignore unload-time network failures.
            }
        }

        if (!_voicesListResizeBound) {
            window.addEventListener('resize', () => {
                layoutVoicesListContainer();
            });
            window.addEventListener('beforeunload', () => {
                _voicePageUnloading = true;
                clearTimeout(_voiceSaveRetryTimer);
                flushPendingVoiceSavesOnUnload();
            });
            window.addEventListener('pageshow', () => {
                _voicePageUnloading = false;
            });
            _voicesListResizeBound = true;
        }

        window.focusVoiceCard = async (voiceName) => {
            const normalizedTarget = normalizeVoiceName(voiceName);
            if (!normalizedTarget) return false;

            const voicesTab = document.getElementById('voices-tab');
            if (voicesTab && voicesTab.style.display === 'none') {
                const voicesLink = document.querySelector('.nav-link[data-tab="voices"]');
                if (voicesLink && typeof voicesLink.click === 'function') {
                    voicesLink.click();
                } else if (voicesTab) {
                    document.querySelectorAll('.tab-content').forEach(tab => { tab.style.display = 'none'; });
                    voicesTab.style.display = 'block';
                }
            }

            await loadVoices();

            const card = Array.from(document.querySelectorAll('.voice-card'))
                .find((element) => normalizeVoiceName(element?.dataset?.voice) === normalizedTarget);
            if (!card) return false;

            card.classList.add('table-warning');
            setTimeout(() => card.classList.remove('table-warning'), 2500);
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            const preferredControl = card.querySelector('.voice-alias-input')
                || card.querySelector('.voice-select')
                || card.querySelector('.designed-voice-select')
                || card.querySelector('input, select, textarea');
            if (preferredControl && typeof preferredControl.focus === 'function') {
                preferredControl.focus();
            }
            return true;
        };

        window.refreshAutomaticVoiceAliases = refreshAutomaticVoiceAliases;
        window.primeVoicesForScriptWorkflow = async () => {
            if (_voiceAliasesPrimedForScript) return;
            _voiceAliasesPrimedForScript = true;
            try {
                await loadVoices();
            } catch (e) {
                console.warn('Failed to prime voice aliases after script creation', e);
            }
        };
        window.resetVoiceAliasWorkflowPrime = () => {
            _voiceAliasesPrimedForScript = false;
        };
