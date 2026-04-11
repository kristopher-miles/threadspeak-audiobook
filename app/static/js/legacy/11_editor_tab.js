        // --- Editor Tab ---
        const WHOLE_PROJECT_CHAPTER_ID = '__whole_project__';
        let isPlayingSequence = false;
        let cachedChunks = []; // Cache to track changes
        let cachedVisibleChunkIds = [];
        let cachedProofreadVisibleChunkIds = [];
        let selectedEditorChapter = WHOLE_PROJECT_CHAPTER_ID;
        let editorChapterAutoSelected = false;
        let selectedProofreadChapter = WHOLE_PROJECT_CHAPTER_ID;
        let proofreadChapterAutoSelected = false;
        let audioQueuePollTimer = null;
        let latestAudioState = null;
        let latestProofreadStatus = null;
        let renderPrepComplete = false;
        let _queueStatusToastShown = false;
        const activeChunkStatusPolls = new Map();
        const singleChunkPollIntervalMs = 1000;
        const singleChunkPollTimeoutMs = 180000;
        const NARRATOR_SELECTION_KEY = 'threadspeak-narrator-selection';

        // Check if any audio is currently playing
        function isAudioPlaying() {
            const audios = document.querySelectorAll('audio');
            for (const audio of audios) {
                if (!audio.paused && !audio.ended) return true;
            }
            return false;
        }

        // Update only changed rows instead of full redraw
        function escapeHtml(value) {
            return (value || '').replace(/[&<>"']/g, ch => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            }[ch]));
        }

        function getChunkStatusMeta(chunk) {
            const validationError = chunk.audio_validation && chunk.audio_validation.error;
            const statusColor = chunk.status === 'done' ? 'success' :
                              chunk.status === 'generating' ? 'warning' :
                              chunk.status === 'error' ? 'danger' : 'secondary';
            const statusDetail = validationError
                ? `<div class="small text-danger mt-1" title="${escapeHtml(validationError)}">${escapeHtml(validationError)}</div>`
                : '';
            return { statusColor, statusDetail };
        }

        function getChunkChapterName(chunk) {
            if (!chunk || typeof chunk.chapter !== 'string') return '';
            return chunk.chapter.trim();
        }

        function getChunkAudioFingerprint(chunk) {
            if (!chunk || !chunk.audio_path) return '';
            const validation = chunk.audio_validation || {};
            return [
                chunk.audio_path,
                validation.file_size_bytes || 0,
                validation.actual_duration_sec || 0,
                chunk.status || '',
            ].join('|');
        }

        function buildChunkAudioSrc(chunk, fallbackToken = '') {
            const rawPath = String(chunk?.audio_path || '').trim();
            if (!rawPath) return '';
            if (/^https?:\/\//i.test(rawPath)) {
                return rawPath;
            }
            const normalizedPath = rawPath.startsWith('/') ? rawPath : `/${rawPath}`;
            const fingerprint = getChunkAudioFingerprint(chunk) || fallbackToken || Date.now().toString();
            return `${normalizedPath}?t=${encodeURIComponent(String(fingerprint))}`;
        }

        function updateCachedChunk(updatedChunk) {
            if (!updatedChunk) return '';
            const chunkRef = getChunkRef(updatedChunk);
            const index = cachedChunks.findIndex(chunk => getChunkRef(chunk) === chunkRef);
            if (index >= 0) {
                cachedChunks[index] = updatedChunk;
            } else if (Array.isArray(cachedChunks) && cachedChunks.length > 0) {
                cachedChunks.push(updatedChunk);
            } else {
                cachedChunks = [updatedChunk];
            }
            return chunkRef;
        }

        function applyTrackedChunkUpdate(updatedChunk) {
            const chunkRef = updateCachedChunk(updatedChunk);
            if (!chunkRef) {
                return { chunkRef: '', editorUpdated: false, proofreadUpdated: false };
            }
            const editorUpdated = updateChunkRow(updatedChunk);
            const proofreadUpdated = updateProofreadRow(updatedChunk);
            renderEditorProgressBar(cachedChunks, latestAudioState);
            return { chunkRef, editorUpdated, proofreadUpdated };
        }

        function markChunkGeneratingLocally(chunkRef) {
            const normalizedRef = String(chunkRef || '');
            if (!normalizedRef) return null;
            const current = (cachedChunks || []).find(chunk => getChunkRef(chunk) === normalizedRef);
            if (!current) return null;
            const updatedChunk = {
                ...current,
                status: 'generating',
            };
            applyTrackedChunkUpdate(updatedChunk);
            return updatedChunk;
        }

        function stopTrackedChunkStatusPolling(chunkRef) {
            const normalizedRef = String(chunkRef || '');
            const existing = activeChunkStatusPolls.get(normalizedRef);
            if (!existing) return;
            existing.cancelled = true;
            if (existing.timer) {
                clearTimeout(existing.timer);
            }
            activeChunkStatusPolls.delete(normalizedRef);
        }

        async function pollTrackedChunkStatus(chunkRef) {
            const normalizedRef = String(chunkRef || '');
            const pollState = activeChunkStatusPolls.get(normalizedRef);
            if (!pollState || pollState.cancelled) return;

            try {
                const updatedChunk = await API.get(`/api/chunks/${encodeURIComponent(normalizedRef)}`);
                if (pollState.cancelled) return;

                pollState.attempts += 1;
                const updateOutcome = applyTrackedChunkUpdate(updatedChunk);
                const reachedTerminalState = updatedChunk?.status !== 'generating';
                const hasVisibleRow = updateOutcome.editorUpdated || updateOutcome.proofreadUpdated;
                const timedOut = (Date.now() - pollState.startedAt) >= singleChunkPollTimeoutMs;

                if (reachedTerminalState || timedOut || !hasVisibleRow) {
                    stopTrackedChunkStatusPolling(normalizedRef);
                    return;
                }

                pollState.timer = setTimeout(() => {
                    pollTrackedChunkStatus(normalizedRef).catch((error) => {
                        console.warn(`Single-chunk status poll failed for ${normalizedRef}`, error);
                        stopTrackedChunkStatusPolling(normalizedRef);
                    });
                }, singleChunkPollIntervalMs);
            } catch (e) {
                console.warn(`Failed to fetch chunk status for ${normalizedRef}`, e);
                stopTrackedChunkStatusPolling(normalizedRef);
            }
        }

        function startTrackedChunkStatusPolling(chunkRef, options = {}) {
            const normalizedRef = String(chunkRef || '');
            if (!normalizedRef || activeChunkStatusPolls.has(normalizedRef)) return;

            const pollState = {
                attempts: 0,
                startedAt: Date.now(),
                timer: null,
                cancelled: false,
            };
            activeChunkStatusPolls.set(normalizedRef, pollState);

            const initialDelay = Number.isFinite(Number(options.initialDelayMs))
                ? Math.max(0, Number(options.initialDelayMs))
                : 0;
            pollState.timer = setTimeout(() => {
                pollTrackedChunkStatus(normalizedRef).catch((error) => {
                    console.warn(`Single-chunk status poll failed for ${normalizedRef}`, error);
                    stopTrackedChunkStatusPolling(normalizedRef);
                });
            }, initialDelay);
        }

        function syncTrackedChunkPollers(visibleChunks) {
            (visibleChunks || []).forEach(chunk => {
                const chunkRef = getChunkRef(chunk);
                if (chunk.status === 'generating') {
                    if (!activeChunkStatusPolls.has(chunkRef)) {
                        startTrackedChunkStatusPolling(chunkRef, { initialDelayMs: singleChunkPollIntervalMs });
                    }
                } else {
                    stopTrackedChunkStatusPolling(chunkRef);
                }
            });
        }

        function buildEditorChapterOptions(chunks) {
            const options = [{
                id: WHOLE_PROJECT_CHAPTER_ID,
                label: 'Whole Project',
                count: chunks.length
            }];
            const counts = new Map();

            chunks.forEach(chunk => {
                const chapter = getChunkChapterName(chunk);
                if (!chapter) return;
                counts.set(chapter, (counts.get(chapter) || 0) + 1);
            });

            counts.forEach((count, chapter) => {
                options.push({
                    id: chapter,
                    label: chapter,
                    count
                });
            });

            return options;
        }

        function getVisibleChunks(chunks) {
            if (selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) {
                return chunks;
            }
            return chunks.filter(chunk => getChunkChapterName(chunk) === selectedEditorChapter);
        }

        function isChapterOnlyEnabled() {
            const toggle = document.getElementById('editor-chapter-only');
            return !toggle || toggle.checked;
        }

        function getActionTargetChunks(chunks) {
            return isChapterOnlyEnabled() ? getVisibleChunks(chunks) : chunks;
        }

        function getActionScopeLabel() {
            if (!isChapterOnlyEnabled()) {
                return 'the entire book';
            }
            if (selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) {
                return 'the whole project';
            }
            return `chapter "${selectedEditorChapter}"`;
        }

        function syncEditorChapterState(chunks) {
            const select = document.getElementById('editor-chapter-select');
            if (!select) return;

            const options = buildEditorChapterOptions(chunks);
            const firstRealChapter = options.find(option => option.id !== WHOLE_PROJECT_CHAPTER_ID);
            if (!editorChapterAutoSelected && selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID && firstRealChapter) {
                selectedEditorChapter = firstRealChapter.id;
                editorChapterAutoSelected = true;
            }
            if (!options.some(option => option.id === selectedEditorChapter)) {
                selectedEditorChapter = firstRealChapter ? firstRealChapter.id : WHOLE_PROJECT_CHAPTER_ID;
            }

            const optionsHtml = options.map(option => `
                <option value="${escapeHtml(option.id)}"${option.id === selectedEditorChapter ? ' selected' : ''}>
                    ${escapeHtml(option.label)} (${option.count})
                </option>
            `).join('');
            const optionsSignature = JSON.stringify(options.map(option => ({
                id: option.id,
                count: option.count,
                selected: option.id === selectedEditorChapter,
            })));

            if (select.dataset.optionsSignature !== optionsSignature) {
                select.innerHTML = optionsHtml;
                select.dataset.optionsSignature = optionsSignature;
            } else if (select.value !== selectedEditorChapter) {
                select.value = selectedEditorChapter;
            }

            updateDeleteChapterButtonVisibility();
            updateNarratorSelector(chunks); // async, fire-and-forget
        }

        function updateChunkRow(chunk) {
            const chunkRef = getChunkRef(chunk);
            const tr = document.querySelector(`tr[data-id="${chunkRef}"]`);
            if (!tr) return false;

            const { statusColor, statusDetail } = getChunkStatusMeta(chunk);

            // Update status badge
            const statusCell = tr.querySelector('.chunk-status-cell');
            const badge = statusCell ? statusCell.querySelector('.badge') : tr.querySelector('.badge');
            if (badge && statusCell) {
                badge.className = `badge bg-${statusColor}`;
                badge.innerText = chunk.status;
                statusCell.innerHTML = `<span class="badge bg-${statusColor}">${chunk.status}</span>${statusDetail}`;
            }

            // Update action area (button/progress)
            const actionContainer = tr.querySelector('.d-flex');
            if (actionContainer) {
                const existingBtn = actionContainer.querySelector('button');
                const existingProgress = actionContainer.querySelector('.progress');

                if (chunk.status === 'generating') {
                    if (existingBtn && !existingProgress) {
                        const progressBar = document.createElement('div');
                        progressBar.className = 'progress';
                        progressBar.style.width = '100px';
                        progressBar.style.height = '20px';
                        progressBar.innerHTML = '<div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>';
                        actionContainer.replaceChild(progressBar, existingBtn);
                    }
                } else {
                    if (existingProgress && !existingBtn) {
                        const btn = document.createElement('button');
                        btn.className = 'btn btn-sm btn-primary';
                        btn.onclick = () => generateChunk(chunkRef);
                        btn.innerHTML = '<i class="fas fa-play"></i> Gen';
                        actionContainer.replaceChild(btn, existingProgress);
                    }
                }

                // Update audio player whenever an audio file exists. If the clip
                // was invalidated, swap back to the "No audio" placeholder right away.
                const existingAudio = actionContainer.querySelector('audio');
                const existingNoAudio = actionContainer.querySelector('.text-muted.small');
                if (chunk.audio_path) {
                    const newSrc = buildChunkAudioSrc(chunk, Date.now().toString());
                    const nextFingerprint = getChunkAudioFingerprint(chunk);

                    if (existingNoAudio) {
                        const audioHtml = `<audio class="chunk-audio" data-id="${escapeHtml(chunkRef)}" data-audio-path="${escapeHtml(chunk.audio_path)}" data-audio-fingerprint="${escapeHtml(nextFingerprint)}" controls preload="none" src="${newSrc}" style="width: 200px; height: 30px;" onplay='stopOthers(${JSON.stringify(chunkRef)})'></audio>`;
                        existingNoAudio.outerHTML = audioHtml;
                    } else if (existingAudio) {
                        const currentFingerprint = existingAudio.dataset.audioFingerprint || '';
                        if (currentFingerprint !== nextFingerprint || existingAudio.dataset.audioPath !== chunk.audio_path) {
                            existingAudio.dataset.audioPath = chunk.audio_path;
                            existingAudio.dataset.audioFingerprint = nextFingerprint;
                            existingAudio.src = newSrc;
                            existingAudio.load();
                        }
                    }
                } else if (existingAudio) {
                    existingAudio.outerHTML = '<span class="text-muted small">No audio</span>';
                } else if (!existingNoAudio) {
                    actionContainer.insertAdjacentHTML('beforeend', '<span class="text-muted small">No audio</span>');
                }
            }
            return true;
        }

        window.changeEditorChapter = async (chapterId) => {
            await flushPendingEditorChunkSaves().catch(err => console.error('Failed to flush editor saves before chapter change:', err));
            selectedEditorChapter = chapterId || WHOLE_PROJECT_CHAPTER_ID;
            stopSequence();
            updateDeleteChapterButtonVisibility();
            await loadChunks(false);
            updateNarratorSelector(cachedChunks); // syncEditorChapterState is skipped when cache is warm
        };


        window.changeEditorScope = () => {
            const toggle = document.getElementById('editor-chapter-only');
            const lbl = document.getElementById('editor-chapter-only-label');
            if (lbl) lbl.textContent = toggle.checked ? 'Generate Chapter' : 'Generate Book';
            syncEditorChapterState(cachedChunks);
        };

        function updateAudioQueueControls(audioState) {
            const cancelBtn = document.getElementById('btn-cancel-render');
            if (!cancelBtn) return;
            const hasAudioWork = Boolean(audioState?.running) || (audioState?.queue || []).length > 0;
            cancelBtn.style.display = hasAudioWork ? 'inline-block' : 'none';
        }

        function getChunkRef(chunk) {
            return String(chunk?.uid || chunk?.id || '');
        }

        function formatDuration(seconds) {
            if (seconds == null || !Number.isFinite(seconds)) return 'Waiting for data';
            const rounded = Math.max(0, Math.round(seconds));
            const hours = Math.floor(rounded / 3600);
            const minutes = Math.floor((rounded % 3600) / 60);
            const secs = rounded % 60;
            if (hours > 0) return `${hours}h ${minutes}m`;
            if (minutes > 0) return `${minutes}m ${secs}s`;
            return `${secs}s`;
        }

        function formatNumber(value) {
            return new Intl.NumberFormat('en-US').format(value || 0);
        }

        function formatBytes(bytes) {
            const value = Number(bytes) || 0;
            if (value < 1024) return `${value} B`;
            const units = ['KB', 'MB', 'GB', 'TB'];
            let size = value / 1024;
            let unitIndex = 0;
            while (size >= 1024 && unitIndex < units.length - 1) {
                size /= 1024;
                unitIndex += 1;
            }
            return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
        }

        function getProofreadVisibleChunks(chunks) {
            const nonSilence = chunks.filter(chunk => chunk.type !== 'silence');
            if (selectedProofreadChapter === WHOLE_PROJECT_CHAPTER_ID) {
                return nonSilence;
            }
            return nonSilence.filter(chunk => getChunkChapterName(chunk) === selectedProofreadChapter);
        }

        function mergeChunkSnapshots(existingChunks, freshChunks) {
            const nextByRef = new Map((freshChunks || []).map(chunk => [getChunkRef(chunk), chunk]));
            const existingRefs = new Set();
            const merged = (existingChunks || []).map(chunk => {
                const ref = getChunkRef(chunk);
                existingRefs.add(ref);
                return nextByRef.get(ref) || chunk;
            });

            (freshChunks || []).forEach(chunk => {
                const ref = getChunkRef(chunk);
                if (!existingRefs.has(ref)) {
                    merged.push(chunk);
                }
            });

            return merged;
        }

        function syncProofreadChapterState(chunks) {
            const select = document.getElementById('proofread-chapter-select');
            const summary = document.getElementById('proofread-chapter-summary');
            if (!select || !summary) return;

            const options = buildEditorChapterOptions(chunks);
            const firstRealChapter = options.find(option => option.id !== WHOLE_PROJECT_CHAPTER_ID);
            if (!proofreadChapterAutoSelected && selectedProofreadChapter === WHOLE_PROJECT_CHAPTER_ID && firstRealChapter) {
                selectedProofreadChapter = firstRealChapter.id;
                proofreadChapterAutoSelected = true;
            }
            if (!options.some(option => option.id === selectedProofreadChapter)) {
                selectedProofreadChapter = firstRealChapter ? firstRealChapter.id : WHOLE_PROJECT_CHAPTER_ID;
            }

            const optionsHtml = options.map(option => `
                <option value="${escapeHtml(option.id)}"${option.id === selectedProofreadChapter ? ' selected' : ''}>
                    ${escapeHtml(option.label)} (${option.count})
                </option>
            `).join('');
            const optionsSignature = JSON.stringify(options.map(option => ({
                id: option.id,
                count: option.count,
                selected: option.id === selectedProofreadChapter,
            })));

            if (select.dataset.optionsSignature !== optionsSignature) {
                select.innerHTML = optionsHtml;
                select.dataset.optionsSignature = optionsSignature;
            } else if (select.value !== selectedProofreadChapter) {
                select.value = selectedProofreadChapter;
            }

            const visibleCount = getProofreadVisibleChunks(chunks).length;
            const viewLabel = selectedProofreadChapter === WHOLE_PROJECT_CHAPTER_ID
                ? `Showing all ${visibleCount} segments.`
                : `Showing ${visibleCount} segments from ${selectedProofreadChapter}.`;
            summary.textContent = viewLabel;
        }

        function getProofreadThreshold() {
            const input = document.getElementById('proofread-threshold');
            const value = parseFloat(input?.value || '0.7');
            return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.7;
        }

        function getProofreadRowStyle(chunk, threshold) {
            const proofread = chunk?.proofread || null;
            if (!proofread || !proofread.checked) return '';
            if (proofread.speaker_match === false || proofread.auto_failed_reason === 'speaker_mismatch') {
                return '--proofread-row-bg: var(--proofread-mismatch-bg); --proofread-row-text: var(--proofread-mismatch-text);';
            }
            const score = Number(proofread.score);
            if (!Number.isFinite(score)) return '';
            if (score >= threshold) {
                return '--proofread-row-bg: var(--proofread-pass-bg);';
            }
            const ratio = threshold > 0 ? Math.max(0, Math.min(1, 1 - (score / threshold))) : 1;
            const alpha = 0.12 + (ratio * 0.62);
            return `--proofread-row-bg: rgba(var(--proofread-fail-rgb), ${alpha.toFixed(3)});`;
        }

        function getProofreadBadge(chunk, threshold) {
            const proofread = chunk?.proofread || null;
            if (!proofread || !proofread.checked) {
                return '<span class="badge bg-secondary">Not checked</span>';
            }
            if (proofread.manual_failed) {
                return '<span class="badge bg-danger">Rejected</span>';
            }
            if (proofread.manual_validated) {
                return '<span class="badge bg-primary">Validated</span>';
            }
            if (proofread.speaker_match === false) {
                return '<span class="badge bg-danger">Speaker mismatch</span>';
            }
            if (proofread.auto_failed_reason === 'duration_outlier') {
                return '<span class="badge bg-danger">Length outlier</span>';
            }
            if (Number(proofread.score) >= threshold) {
                return '<span class="badge bg-success">Passed</span>';
            }
            return '<span class="badge bg-warning text-dark">Needs review</span>';
        }

        function shouldShowProofreadRegenerate(chunk) {
            const hasAudio = Boolean((chunk?.audio_path || '').trim());
            if (!hasAudio) return true;
            const proofread = chunk?.proofread || null;
            return Boolean(proofread && proofread.checked && !proofread.passed);
        }

        function getProofreadGenerateButtonHtml(chunk, chunkRef) {
            if (!shouldShowProofreadRegenerate(chunk)) return '';
            const hasAudio = Boolean((chunk?.audio_path || '').trim());
            const label = hasAudio ? 'Regenerate' : 'Generate';
            const icon = hasAudio ? 'fa-redo' : 'fa-play';
            return `<button class="btn btn-sm btn-outline-danger mt-2" onclick='regenerateProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas ${icon} me-1"></i>${label}</button>`;
        }

        function getProofreadValidateButtonHtml(chunk, chunkRef) {
            // Show both buttons on any clip that has been scored (checked) or has audio.
            // Clips without audio or without a score yet get no buttons.
            const hasAudio = Boolean((chunk?.audio_path || '').trim());
            const proofread = chunk?.proofread || {};
            if (!hasAudio || !proofread.checked) return '';

            const isValidated = Boolean(proofread.manual_validated);
            const isRejected  = Boolean(proofread.manual_failed);

            const validateClass = isValidated ? 'btn-primary'         : 'btn-outline-primary';
            const rejectClass   = isRejected  ? 'btn-warning'         : 'btn-outline-warning';

            const validateBtn = `<button class="btn btn-sm ${validateClass} mt-2 ms-2" onclick='validateProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas fa-check me-1"></i>Validate</button>`;
            const rejectBtn   = `<button class="btn btn-sm ${rejectClass} mt-2 ms-2" onclick='rejectProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas fa-xmark me-1"></i>Reject</button>`;

            return validateBtn + rejectBtn;
        }

        function getProofreadEditButtonHtml(chunk, chunkRef) {
            return `<button class="btn btn-sm btn-outline-secondary mt-2 ms-2" onclick='editProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas fa-pen me-1"></i>Edit</button>`;
        }

        function getChunkCachedTranscript(chunk) {
            const proofreadTranscript = String(chunk?.proofread?.transcript_text || '').trim();
            if (proofreadTranscript) return proofreadTranscript;
            const validationTranscript = String(chunk?.audio_validation?.transcript_text || '').trim();
            if (validationTranscript) return validationTranscript;
            return '';
        }

        function shouldShowProofreadCompare(chunk) {
            if (!Boolean((chunk?.audio_path || '').trim())) return false;
            return !getChunkCachedTranscript(chunk);
        }

        function getProofreadFingerprint(chunk, threshold) {
            const proofread = chunk?.proofread || {};
            return [
                getChunkRef(chunk),
                chunk?.audio_path || '',
                chunk?.status || '',
                threshold,
                proofread.checked ? 1 : 0,
                proofread.score ?? '',
                proofread.passed ? 1 : 0,
                proofread.error || '',
                getChunkCachedTranscript(chunk),
                proofread.actual_duration_sec ?? '',
                proofread.expected_duration_sec ?? '',
                proofread.auto_failed_reason || '',
                proofread.speaker_match ? 1 : 0,
                proofread.manual_validated ? 1 : 0,
                proofread.manual_failed ? 1 : 0,
                proofread.validated_at ?? '',
                proofread.failed_at ?? '',
            ].join('|');
        }

        function updateProofreadRow(chunk, threshold = getProofreadThreshold()) {
            const chunkRef = getChunkRef(chunk);
            const row = document.querySelector(`tr[data-proofread-id="${chunkRef}"]`);
            if (!row) return false;

            const proofread = chunk.proofread || {};
            const score = Number.isFinite(Number(proofread.score)) ? Number(proofread.score).toFixed(3) : '—';
            const transcriptText = getChunkCachedTranscript(chunk);
            const transcript = transcriptText
                ? escapeHtml(transcriptText)
                : '<span class="text-muted small">No transcript captured</span>';
            const detail = proofread.error
                ? `<div class="small ${Number(proofread.score) >= threshold ? 'text-muted' : 'text-danger'} mt-1">${escapeHtml(proofread.error)}</div>`
                : '';
            const manualValidationDetail = proofread.manual_validated
                ? '<div class="small text-primary mt-1">Manually validated by user.</div>'
                : '';
            const manualFailureDetail = proofread.manual_failed
                ? '<div class="small text-danger mt-1">Manually marked as failed by user.</div>'
                : '';
            const cachedTranscriptDetail = (!proofread.checked && transcriptText)
                ? '<div class="small text-muted mt-1">Cached transcript from repair.</div>'
                : '';
            const _actual = proofread.actual_duration_sec;
            const _expected = proofread.expected_duration_sec || 0;
            const durationDetail = (_actual != null && (_actual < 0.1 || Math.abs(_actual - _expected) >= 2))
                ? `<div class="small text-muted mt-1">Duration ${_actual}s vs expected ${_expected}s</div>`
                : '';
            const regenButton = getProofreadGenerateButtonHtml(chunk, chunkRef);
            const compareButton = shouldShowProofreadCompare(chunk)
                ? `<button class="btn btn-sm btn-outline-secondary mt-2 ms-2" onclick='compareProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas fa-scale-balanced me-1"></i>Compare</button>`
                : '';
            const validateButton = getProofreadValidateButtonHtml(chunk, chunkRef);
            const editButton = getProofreadEditButtonHtml(chunk, chunkRef);

            row.style.cssText = getProofreadRowStyle(chunk, threshold);

            const scoreCell = row.children[2];
            const transcriptCell = row.children[3];
            const audioCell = row.children[4];
            if (scoreCell) {
                scoreCell.innerHTML = `
                    ${getProofreadBadge(chunk, threshold)}
                    <div class="small mt-1">Score: ${score}</div>
                    ${detail}
                    ${manualValidationDetail}
                    ${manualFailureDetail}
                    ${cachedTranscriptDetail}
                `;
            }
            if (transcriptCell) {
                transcriptCell.innerHTML = `<div>${transcript}</div>${durationDetail}`;
            }
            if (audioCell) {
                const newAudioSrc = chunk.audio_path ? buildChunkAudioSrc(chunk) : '';
                const existingAudio = audioCell.querySelector('audio');
                const existingSrc = existingAudio ? existingAudio.getAttribute('src') : null;
                if (existingSrc !== newAudioSrc) {
                    const audioHtml = newAudioSrc
                        ? `<audio controls preload="none" src="${newAudioSrc}" style="width: 210px; height: 30px;"></audio>`
                        : '<span class="text-muted small">No audio</span>';
                    audioCell.innerHTML = `${audioHtml}<div></div>`;
                }
                const btnDiv = audioCell.querySelector('div') || (() => {
                    const d = document.createElement('div');
                    audioCell.appendChild(d);
                    return d;
                })();
                btnDiv.innerHTML = `${regenButton}${compareButton}${validateButton}${editButton}`;
            }
            row.dataset.proofreadFingerprint = getProofreadFingerprint(chunk, threshold);
            return true;
        }

        function renderProofreadTable(chunks) {
            const tbody = document.getElementById('proofread-table-body');
            if (!tbody) return;
            const visibleChunks = getProofreadVisibleChunks(chunks);
            const threshold = getProofreadThreshold();

            if (!visibleChunks.length) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No segments in this chapter yet.</td></tr>';
                return;
            }

            // Save live audio elements before wiping — avoids re-requesting already-cached files
            const savedAudio = new Map();
            tbody.querySelectorAll('tr[data-proofread-id]').forEach(row => {
                const audio = row.children[4]?.querySelector('audio');
                if (audio) savedAudio.set(row.dataset.proofreadId, audio);
            });

            tbody.innerHTML = visibleChunks.map(chunk => {
                const proofread = chunk.proofread || {};
                const chunkRef = getChunkRef(chunk);
                const score = Number.isFinite(Number(proofread.score)) ? Number(proofread.score).toFixed(3) : '—';
                const transcriptText = getChunkCachedTranscript(chunk);
                const transcript = transcriptText
                    ? escapeHtml(transcriptText)
                    : '<span class="text-muted small">No transcript captured</span>';
                const detail = proofread.error
                    ? `<div class="small ${Number(proofread.score) >= threshold ? 'text-muted' : 'text-danger'} mt-1">${escapeHtml(proofread.error)}</div>`
                    : '';
                const manualValidationDetail = proofread.manual_validated
                    ? '<div class="small text-primary mt-1">Manually validated by user.</div>'
                    : '';
                const manualFailureDetail = proofread.manual_failed
                    ? '<div class="small text-danger mt-1">Manually marked as failed by user.</div>'
                    : '';
                const cachedTranscriptDetail = (!proofread.checked && transcriptText)
                    ? '<div class="small text-muted mt-1">Cached transcript from repair.</div>'
                    : '';
                const _actual = proofread.actual_duration_sec;
                const _expected = proofread.expected_duration_sec || 0;
                const durationDetail = (_actual != null && (_actual < 0.1 || Math.abs(_actual - _expected) >= 2))
                    ? `<div class="small text-muted mt-1">Duration ${_actual}s vs expected ${_expected}s</div>`
                    : '';
                const audioHtml = chunk.audio_path
                    ? `<audio controls preload="none" src="${buildChunkAudioSrc(chunk, Date.now().toString())}" style="width: 210px; height: 30px;"></audio>`
                    : '<span class="text-muted small">No audio</span>';
                const regenButton = getProofreadGenerateButtonHtml(chunk, chunkRef);
                const compareButton = shouldShowProofreadCompare(chunk)
                    ? `<button class="btn btn-sm btn-outline-secondary mt-2 ms-2" onclick='compareProofreadChunk(${JSON.stringify(chunkRef)})'><i class="fas fa-scale-balanced me-1"></i>Compare</button>`
                    : '';
                const validateButton = getProofreadValidateButtonHtml(chunk, chunkRef);
                const editButton = getProofreadEditButtonHtml(chunk, chunkRef);
                return `
                    <tr data-proofread-id="${escapeHtml(chunkRef)}" data-proofread-fingerprint="${escapeHtml(getProofreadFingerprint(chunk, threshold))}" style="${getProofreadRowStyle(chunk, threshold)}">
                        <td>${escapeHtml(chunk.speaker || '')}</td>
                        <td>${escapeHtml(chunk.text || '')}</td>
                        <td>
                            ${getProofreadBadge(chunk, threshold)}
                            <div class="small mt-1">Score: ${score}</div>
                            ${detail}
                            ${manualValidationDetail}
                            ${manualFailureDetail}
                            ${cachedTranscriptDetail}
                        </td>
                        <td>
                            <div>${transcript}</div>
                            ${durationDetail}
                        </td>
                        <td>${audioHtml}<div>${regenButton}${compareButton}${validateButton}${editButton}</div></td>
                    </tr>
                `;
            }).join('');

            // Restore audio elements whose src hasn't changed — preserves in-progress loads
            if (savedAudio.size > 0) {
                tbody.querySelectorAll('tr[data-proofread-id]').forEach(newRow => {
                    const oldAudio = savedAudio.get(newRow.dataset.proofreadId);
                    if (!oldAudio) return;
                    const newAudio = newRow.children[4]?.querySelector('audio');
                    if (newAudio && newAudio.getAttribute('src') === oldAudio.getAttribute('src')) {
                        newAudio.replaceWith(oldAudio);
                    }
                });
            }

            cachedProofreadVisibleChunkIds = visibleChunks.map(chunk => getChunkRef(chunk));
        }

        function getProofreadDerivedStats(chunks, threshold = getProofreadThreshold(), chapterId = selectedProofreadChapter) {
            const sourceChunks = Array.isArray(chunks) ? chunks : [];
            const scopedChunks = chapterId === WHOLE_PROJECT_CHAPTER_ID
                ? sourceChunks
                : sourceChunks.filter(chunk => getChunkChapterName(chunk) === chapterId);
            let passed = 0;
            let failed = 0;
            let autoFailed = 0;
            let skipped = 0;

            scopedChunks.forEach(chunk => {
                const hasAudio = Boolean((chunk?.audio_path || '').trim());
                const proofread = chunk?.proofread || null;
                if (!hasAudio) {
                    skipped += 1;
                    return;
                }
                if (!proofread || !proofread.checked) {
                    return;
                }
                const score = Number(proofread.score);
                const isPassed = proofread.passed === true && Number.isFinite(score) && score >= threshold;
                if (isPassed) {
                    passed += 1;
                } else {
                    failed += 1;
                    if (proofread.auto_failed_reason) {
                        autoFailed += 1;
                    }
                }
            });

            return {
                total: scopedChunks.length,
                passed,
                failed,
                auto_failed: autoFailed,
                skipped,
            };
        }

        function renderProofreadTaskStatus(status) {
            latestProofreadStatus = status || latestProofreadStatus || { running: false, progress: {}, logs: [] };
            status = latestProofreadStatus;
            const progress = status?.progress || {};
            const running = !!status?.running;
            const derived = getProofreadDerivedStats(cachedChunks);
            const projectDerived = getProofreadDerivedStats(cachedChunks, getProofreadThreshold(), WHOLE_PROJECT_CHAPTER_ID);
            const progressCompleted = Number.isFinite(Number(progress.processed))
                ? Number(progress.processed)
                : (Number.isFinite(Number(progress.transcribed)) ? Number(progress.transcribed) : 0);
            const progressTotal = Number.isFinite(Number(progress.pending_total))
                ? Number(progress.pending_total)
                : (Number.isFinite(Number(progress.transcribe_total))
                    ? Number(progress.transcribe_total)
                    : (Number.isFinite(Number(progress.pending_clips)) ? Number(progress.pending_clips) : 0));
            document.getElementById('proofread-phase').textContent = progress.phase || (status?.running ? 'proofreading' : 'Idle');
            document.getElementById('proofread-progress').textContent = running
                ? `${formatNumber(progressCompleted)} / ${formatNumber(progressTotal)}`
                : `${formatNumber(derived.passed + derived.failed)} / ${formatNumber(Math.max(derived.total - derived.skipped, 0))}`;
            document.getElementById('proofread-eta').textContent = formatDuration(progress.eta_seconds);
            document.getElementById('proofread-elapsed').textContent = formatDuration(progress.elapsed_seconds || 0);
            document.getElementById('proofread-passed').textContent = formatNumber(running ? (progress.passed || 0) : derived.passed);
            document.getElementById('proofread-failed').textContent = formatNumber(running ? (progress.failed || 0) : projectDerived.failed);
            document.getElementById('proofread-auto-failed').textContent = formatNumber(running ? (progress.auto_failed || 0) : derived.auto_failed);
            document.getElementById('proofread-skipped').textContent = formatNumber(running ? (progress.skipped || 0) : derived.skipped);

            const summary = document.getElementById('proofread-run-summary');
            const chapterLabel = selectedProofreadChapter === WHOLE_PROJECT_CHAPTER_ID ? 'the whole project' : selectedProofreadChapter;
            if (status?.running) {
                summary.textContent = `Proofreading ${chapterLabel} at certainty ${getProofreadThreshold().toFixed(2)}.`;
            } else if ((status?.logs || []).length) {
                summary.textContent = status.logs[status.logs.length - 1] || '';
            } else {
                summary.textContent = '';
            }

            document.getElementById('btn-proofread-sequence').disabled = running;
            document.getElementById('btn-proofread-book').disabled = running;
            document.getElementById('btn-proofread-discard-selection').disabled = running;
            document.getElementById('btn-proofread-regrade-book').disabled = running;
        }

        window.changeProofreadChapter = async (chapterId) => {
            selectedProofreadChapter = chapterId || WHOLE_PROJECT_CHAPTER_ID;
            if (!Array.isArray(cachedChunks) || cachedChunks.length === 0) {
                await loadChunks(true);
                return;
            }
            renderProofreadTable(cachedChunks);
            syncProofreadChapterState(cachedChunks);
            renderProofreadTaskStatus(await API.get('/api/status/proofread'));
        };

        async function jumpToNextProofreadFailure(afterChunkRef) {
            let chunks = Array.isArray(cachedChunks) ? cachedChunks : [];
            if (!chunks.length) {
                chunks = await API.get('/api/chunks/view');
                cachedChunks = chunks;
                syncEditorChapterState(chunks);
                syncProofreadChapterState(chunks);
            }

            const isFailure = chunk => {
                const p = chunk?.proofread || {};
                return Boolean(p.checked) && !Boolean(p.passed) && !Boolean(p.manual_validated) && !Boolean(p.manual_failed);
            };

            const currentIndex = afterChunkRef != null
                ? chunks.findIndex(c => getChunkRef(c) === String(afterChunkRef))
                : -1;

            // Search from the chunk after the current one, then wrap around
            const candidates = currentIndex >= 0
                ? [...chunks.slice(currentIndex + 1), ...chunks.slice(0, currentIndex + 1)]
                : chunks;

            const nextFailure = candidates.find(isFailure);
            if (!nextFailure) return; // no failures left, stay put

            const chapterId = getChunkChapterName(nextFailure) || WHOLE_PROJECT_CHAPTER_ID;
            selectedProofreadChapter = chapterId;
            renderProofreadTable(cachedChunks);
            syncProofreadChapterState(chunks);

            await new Promise(resolve => requestAnimationFrame(() => resolve()));

            const chunkRef = getChunkRef(nextFailure);
            const row = document.querySelector(`#proofread-table-body tr[data-proofread-id="${CSS.escape(chunkRef)}"]`);
            if (!row) return;

            document.querySelectorAll('#proofread-table-body tr.table-primary').forEach(tr => tr.classList.remove('table-primary'));
            row.classList.add('table-primary');
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        window.jumpToFirstProofreadFailure = async () => {
            try {
                let chunks = Array.isArray(cachedChunks) ? cachedChunks : [];
                if (!chunks.length) {
                    chunks = await API.get('/api/chunks/view');
                    cachedChunks = chunks;
                    syncEditorChapterState(chunks);
                    syncProofreadChapterState(chunks);
                }

                const firstFailure = chunks.find(chunk => {
                    const proofread = chunk?.proofread || {};
                    return Boolean(proofread.checked) && !Boolean(proofread.passed) && !Boolean(proofread.manual_validated) && !Boolean(proofread.manual_failed);
                });
                if (!firstFailure) {
                    showToast('No proofread failures were found.', 'info', 2500);
                    return;
                }

                const chapterId = getChunkChapterName(firstFailure) || WHOLE_PROJECT_CHAPTER_ID;
                selectedProofreadChapter = chapterId;
                renderProofreadTable(cachedChunks);
                syncProofreadChapterState(chunks);
                renderProofreadTaskStatus(await API.get('/api/status/proofread'));

                await new Promise(resolve => requestAnimationFrame(() => resolve()));

                const chunkRef = getChunkRef(firstFailure);
                const row = document.querySelector(`#proofread-table-body tr[data-proofread-id="${CSS.escape(chunkRef)}"]`);
                if (!row) {
                    showToast('Found a proofread failure, but could not scroll to it.', 'warning', 3000);
                    return;
                }

                document.querySelectorAll('#proofread-table-body tr.table-primary').forEach(tr => tr.classList.remove('table-primary'));
                row.classList.add('table-primary');
                row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } catch (e) {
                showToast('Failed to jump to the first proofread failure: ' + e.message, 'error');
            }
        };

        window.saveProofreadThreshold = async () => {
            const thresholdInput = document.getElementById('proofread-threshold');
            const threshold = getProofreadThreshold();
            thresholdInput.value = threshold.toFixed(2).replace(/\.00$/, '.0');
            try {
                const config = await API.get('/api/config');
                config.proofread = { ...(config.proofread || {}), certainty_threshold: threshold };
                await API.post('/api/config', config);
                renderProofreadTable(cachedChunks);
                renderProofreadTaskStatus(latestProofreadStatus || { running: false, progress: {}, logs: [] });
            } catch (e) {
                showToast('Failed to save proofread certainty: ' + e.message, 'error');
            }
        };

        window.previewProofreadThreshold = () => {
            getProofreadThreshold();
            renderProofreadTable(cachedChunks);
            renderProofreadTaskStatus(latestProofreadStatus || { running: false, progress: {}, logs: [] });
        };

        async function startProofreadRun(chapter) {
            const threshold = getProofreadThreshold();
            const payload = {
                threshold,
                chapter: chapter === WHOLE_PROJECT_CHAPTER_ID ? null : chapter,
            };
            await API.post('/api/proofread', payload);
            await pollLogs('proofread', 'proofread-logs');
            await loadChunks(false);
        }

        window.startProofreadSequence = async () => {
            try {
                await startProofreadRun(selectedProofreadChapter);
                showToast('Proofread run finished.', 'success', 4000);
            } catch (e) {
                showToast('Proofread failed: ' + e.message, 'error');
            }
        };

        window.startProofreadEntireBook = async () => {
            try {
                await startProofreadRun(null);
                showToast('Full-book proofread finished.', 'success', 4000);
            } catch (e) {
                showToast('Proofread failed: ' + e.message, 'error');
            }
        };

        window.regradeBook = async () => {
            try {
                // Discard all proofread results for the whole book, preserving transcripts
                await API.post('/api/proofread/discard_selection', { chapter: null });
                // Re-run proofread on the whole book — transcribes missing clips, re-scores all
                await startProofreadRun(null);
                showToast('Book regrade complete.', 'success', 4000);
            } catch (e) {
                showToast('Regrade failed: ' + e.message, 'error');
            }
        };

        window.discardProofreadSelection = async () => {
            try {
                const result = await API.post('/api/proofread/discard_selection', {
                    chapter: selectedProofreadChapter === WHOLE_PROJECT_CHAPTER_ID ? null : selectedProofreadChapter,
                });
                await loadChunks(false);
                showToast(
                    `Discarded ${result.discarded || 0} proofread result(s). Preserved ${result.preserved_transcripts || 0} transcript cache(s).`,
                    'success',
                    4000
                );
            } catch (e) {
                showToast('Failed to discard proofread selection: ' + e.message, 'error');
            }
        };

        window.regenerateProofreadChunk = async (chunkRef) => {
            try {
                await API.post(`/api/chunks/${encodeURIComponent(chunkRef)}/regenerate`, {});
                markChunkGeneratingLocally(chunkRef);
                startTrackedChunkStatusPolling(chunkRef);
                showToast('Regeneration started for the selected clip.', 'success', 3000);
            } catch (e) {
                showToast('Failed to regenerate clip: ' + e.message, 'error');
            }
        };

        function applyProofreadChunkUpdate(updatedChunk) {
            // Patch cachedChunks in-place so jumpToNextProofreadFailure and
            // renderProofreadTable see the correct state even when loadChunks(false)
            // used chapter scope and didn't fetch this chunk's chapter.
            if (!updatedChunk) return;
            const ref = getChunkRef(updatedChunk);
            const idx = cachedChunks.findIndex(c => getChunkRef(c) === ref);
            if (idx >= 0) cachedChunks[idx] = updatedChunk;
            updateProofreadRow(updatedChunk);
        }

        window.validateProofreadChunk = async (chunkRef) => {
            try {
                const result = await API.post(`/api/proofread/${encodeURIComponent(chunkRef)}/validate`, {
                    threshold: getProofreadThreshold(),
                });
                applyProofreadChunkUpdate(result.chunk);
                showToast('Clip validated.', 'success', 2000);
                await loadChunks(false);
                await jumpToNextProofreadFailure(chunkRef);
            } catch (e) {
                showToast('Failed to validate clip: ' + e.message, 'error');
            }
        };

        window.rejectProofreadChunk = async (chunkRef) => {
            try {
                const result = await API.post(`/api/proofread/${encodeURIComponent(chunkRef)}/reject`, {
                    threshold: getProofreadThreshold(),
                });
                applyProofreadChunkUpdate(result.chunk);
                showToast('Clip rejected.', 'warning', 2000);
                await loadChunks(false);
                await jumpToNextProofreadFailure(chunkRef);
            } catch (e) {
                showToast('Failed to reject clip: ' + e.message, 'error');
            }
        };

        window.compareProofreadChunk = async (chunkRef) => {
            try {
                await API.post(`/api/proofread/${encodeURIComponent(chunkRef)}/compare`, {
                    threshold: getProofreadThreshold(),
                });
                showToast('Clip compared successfully.', 'success', 3000);
                await loadChunks(false);
            } catch (e) {
                showToast('Failed to compare clip: ' + e.message, 'error');
            }
        };

        window.editProofreadChunk = async (chunkRef) => {
            try {
                const chunk = (cachedChunks || []).find(candidate => getChunkRef(candidate) === String(chunkRef))
                    || (await API.get('/api/chunks/view')).find(candidate => getChunkRef(candidate) === String(chunkRef));
                if (!chunk) {
                    showToast('Could not find that clip in the editor.', 'warning');
                    return;
                }

                const chapter = getChunkChapterName(chunk);
                const chapterToggle = document.getElementById('editor-chapter-only');
                if (chapterToggle) {
                    chapterToggle.checked = true;
                }
                selectedEditorChapter = chapter || WHOLE_PROJECT_CHAPTER_ID;

                const editorTab = document.querySelector('[data-tab="editor"]');
                if (!editorTab) {
                    showToast('Editor tab is unavailable.', 'error');
                    return;
                }
                editorTab.click();

                const focused = await focusChunkInEditor(String(chunkRef));
                if (!focused) {
                    showToast('Opened the editor, but could not focus that clip.', 'warning');
                    return;
                }
            } catch (e) {
                showToast('Failed to open clip in editor: ' + e.message, 'error');
            }
        };

        async function clearProofreadFailures(chapter) {
            const threshold = getProofreadThreshold();
            const payload = {
                threshold,
                chapter: chapter === WHOLE_PROJECT_CHAPTER_ID ? null : chapter,
            };
            const result = await API.post('/api/proofread/clear_failures', payload);
            if (result.ungraded_with_audio > 0) {
                showToast('Only graded audio will be deleted. Grade your complete book for a more exhaustive result.', 'warning', 7000);
            }
            await loadChunks(false);
            if (result.cleared > 0) {
                showToast(`Cleared ${result.cleared} failed clip${result.cleared === 1 ? '' : 's'}.`, 'success', 4000);
            } else {
                showToast('No graded failed clips were found in the selected scope.', 'info', 4000);
            }
        }

        window.clearProofreadFailuresLocal = async () => {
            try {
                await clearProofreadFailures(selectedProofreadChapter);
            } catch (e) {
                showToast('Failed to clear local failures: ' + e.message, 'error');
            }
        };

        window.clearProofreadFailuresBook = async () => {
            try {
                await clearProofreadFailures(null);
            } catch (e) {
                showToast('Failed to clear book failures: ' + e.message, 'error');
            }
        };

        function renderEditorProgressBar(chunks, audioState = latestAudioState) {
            const progressBar = document.getElementById('full-progress-bar');
            if (!progressBar) return;

            const allChunks = Array.isArray(chunks) ? chunks : [];
            const nonEmptyChunks = allChunks.filter(chunk => (chunk.text || '').trim());
            const completed = nonEmptyChunks.filter(chunk => chunk.status === 'done').length;
            const total = nonEmptyChunks.length;
            const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
            const currentJob = audioState?.current_job || null;
            const hasActiveAudioWork = Boolean(audioState?.running) && currentJob && (currentJob.total_chunks || 0) > 0;

            if (hasActiveAudioWork) {
                const processed = Number(currentJob.processed_clips) || 0;
                const jobTotal = Number(currentJob.total_chunks) || 0;
                const jobPercentage = jobTotal > 0 ? Math.round((processed / jobTotal) * 100) : 0;
                progressBar.style.width = `${jobPercentage}%`;
                progressBar.classList.add('progress-bar-animated', 'progress-bar-striped', 'bg-warning');
                progressBar.classList.remove('bg-success');
                progressBar.innerText = `${jobPercentage}% (${processed}/${jobTotal})`;
                progressBar.title = `${currentJob.label || 'Rendering audio'} • current job ${processed}/${jobTotal}`;
                return;
            }

            progressBar.style.width = `${percentage}%`;
            progressBar.classList.remove('progress-bar-animated', 'bg-warning');
            progressBar.classList.add('progress-bar-striped', 'bg-success');
            progressBar.innerText = `${percentage}% (${completed}/${total})`;
            progressBar.title = 'Completed audio coverage';
        }

        function renderAudioEstimatePanel(audioState) {
            const metrics = audioState?.metrics || {};
            const etaEl = document.getElementById('editor-estimate-eta');
            const clipTimeEl = document.getElementById('editor-estimate-clip-time');
            const speedEl = document.getElementById('editor-estimate-speed');
            const wordsEl = document.getElementById('editor-estimate-words');
            const errorsEl = document.getElementById('editor-estimate-errors');
            if (!etaEl || !clipTimeEl || !speedEl || !wordsEl || !errorsEl) return;

            etaEl.textContent = formatDuration(metrics.estimated_remaining_seconds);
            const sampleCount = metrics.processed_clips || 0;
            clipTimeEl.textContent = sampleCount > 0
                ? formatDuration((metrics.total_elapsed_seconds || 0) / sampleCount)
                : 'Waiting for data';
            speedEl.textContent = metrics.words_per_minute
                ? `${Math.round(metrics.words_per_minute)} words/min`
                : 'Not enough samples';
            wordsEl.textContent = formatNumber(metrics.remaining_words);
            const errorCount = metrics.error_clips || 0;
            const errorRate = ((metrics.error_rate || 0) * 100).toFixed(1);
            errorsEl.textContent = `${errorCount} clip${errorCount === 1 ? '' : 's'}, ${errorRate}%`;
        }

        function renderAudioQueueStatus(audioState) {
            if (_queueStatusToastShown) return;

            const current = audioState?.current_job;
            const queued = (audioState?.queue || []).length;
            let msg = '';
            if (current) {
                msg = `Running job #${current.id}: ${current.label}. ${queued} queued behind it.`;
            } else if (queued > 0) {
                msg = `${queued} audio job${queued === 1 ? '' : 's'} queued.`;
            }
            if (msg) {
                showToast(msg, 'info', 6000);
                _queueStatusToastShown = true;
            }
        }

        function renderAudioMergeProgress(status) {
            const panel = document.getElementById('audio-merge-progress');
            if (!panel) return;

            const progress = status?.merge_progress || {};
            const shouldShow = Boolean(status?.merge_running) || Boolean(progress?.running) || (progress?.total_chapters || 0) > 0;
            panel.style.display = shouldShow ? '' : 'none';
            if (!shouldShow) return;

            const stageMap = {
                preparing: 'Preparing Inputs',
                starting: 'Starting',
                assembling: 'Assembling Chapters',
                packing: 'Packing Parts',
                bundling: 'Writing Zip',
                exporting: 'Exporting File',
                complete: 'Complete',
                idle: 'Idle',
            };

            document.getElementById('audio-merge-stage').textContent = stageMap[progress.stage] || 'Working';
            document.getElementById('audio-merge-chapter-progress').textContent = `${progress.chapter_index || 0} / ${progress.total_chapters || 0}`;
            document.getElementById('audio-merge-elapsed').textContent = formatDuration(progress.elapsed_seconds || 0);
            document.getElementById('audio-merge-duration').textContent = formatDuration(progress.merged_duration_seconds || 0);
            document.getElementById('audio-merge-current-chapter').textContent = progress.chapter_label || 'Waiting for merge to start';
            document.getElementById('audio-merge-size').textContent = formatBytes(progress.estimated_size_bytes || 0);
            document.getElementById('audio-merge-output-size').textContent = formatBytes(progress.output_file_size_bytes || 0);
        }

        function getNarratorSelections() {
            try { return JSON.parse(localStorage.getItem(NARRATOR_SELECTION_KEY) || '{}'); }
            catch { return {}; }
        }

        window.onNarratorSelectorChange = async () => {
            const select = document.getElementById('editor-narrator-select');
            if (!select || selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) return;

            const newValue = select.value;
            const oldValue = getNarratorSelections()[selectedEditorChapter] || null;
            if (newValue === oldValue) return;

            // Check for existing NARRATOR audio in this chapter
            const withAudio = cachedChunks.filter(c =>
                (c.speaker || '').trim().toUpperCase() === 'NARRATOR' &&
                getChunkChapterName(c) === selectedEditorChapter &&
                c.audio_path
            );

            if (withAudio.length > 0) {
                const n = withAudio.length;
                const confirmed = await showConfirm(
                    `Changing the narrator for "${selectedEditorChapter}" will delete ${n} generated clip${n === 1 ? '' : 's'}. Continue?`
                );
                if (!confirmed) {
                    select.value = oldValue || 'NARRATOR';
                    return;
                }
                await API.post('/api/narrator_overrides', {
                    chapter: selectedEditorChapter,
                    voice: newValue,
                    invalidate_audio: true,
                });
                await loadChunks(true);
            } else {
                await API.post('/api/narrator_overrides', {
                    chapter: selectedEditorChapter,
                    voice: newValue,
                    invalidate_audio: false,
                });
            }

            const selections = getNarratorSelections();
            selections[selectedEditorChapter] = newValue;
            localStorage.setItem(NARRATOR_SELECTION_KEY, JSON.stringify(selections));
        };

        async function syncNarratorSelectionsFromBackend() {
            try {
                const overrides = await API.get('/api/narrator_overrides');
                const selections = getNarratorSelections();
                // Backend is authoritative — merge, backend wins on conflict
                Object.assign(selections, overrides);
                localStorage.setItem(NARRATOR_SELECTION_KEY, JSON.stringify(selections));
            } catch (e) {
                // Non-fatal; localStorage remains as fallback
            }
        }

        async function getNarratingVoices() {
            if (window._narratingVoicesCache) return window._narratingVoicesCache;
            try {
                const voices = await API.get('/api/voices');
                window._narratingVoicesCache = voices
                    .filter(v => (v.name || '').trim().toUpperCase() === 'NARRATOR' || v.config?.narrates === true)
                    .map(v => v.name);
            } catch (e) {
                window._narratingVoicesCache = ['NARRATOR'];
            }
            return window._narratingVoicesCache;
        }

        function setNarratorSelectorVisible(group, visible) {
            group.style.display = visible ? 'flex' : 'none';
        }

        async function updateNarratorSelector(chunks) {
            const group = document.getElementById('narrator-selector-group');
            if (!group) return;

            if (selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) {
                setNarratorSelectorVisible(group, false);
                return;
            }

            const narratingVoices = await getNarratingVoices();
            const hasExtra = narratingVoices.some(v => v.trim().toUpperCase() !== 'NARRATOR');
            if (!hasExtra) {
                setNarratorSelectorVisible(group, false);
                return;
            }

            const sourceChunks = chunks || cachedChunks;
            const chapterText = sourceChunks
                .filter(c => getChunkChapterName(c) === selectedEditorChapter)
                .map(c => c.text || '')
                .join(' ');

            const narratorName = narratingVoices.find(v => v.trim().toUpperCase() === 'NARRATOR') || 'NARRATOR';
            const others = narratingVoices.filter(v => v.trim().toUpperCase() !== 'NARRATOR');

            const countMentions = name => {
                try {
                    const re = new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                    return (chapterText.match(re) || []).length;
                } catch { return 0; }
            };
            others.sort((a, b) => countMentions(b) - countMentions(a));

            const ordered = [narratorName, ...others];
            const saved = getNarratorSelections()[selectedEditorChapter];
            const selected = ordered.includes(saved) ? saved : narratorName;

            const select = document.getElementById('editor-narrator-select');
            if (!select) return;
            select.innerHTML = ordered.map(v =>
                `<option value="${escapeHtml(v)}"${v === selected ? ' selected' : ''}>${escapeHtml(v)}</option>`
            ).join('');

            setNarratorSelectorVisible(group, true);
        }

        async function focusChunkInEditor(chunkId) {
            await loadChunks(false);
            const row = document.querySelector(`tr[data-id="${chunkId}"]`);
            if (!row) return false;
            document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));
            row.classList.add('table-primary');
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return true;
        }

        window.jumpToFirstEditorError = async () => {
            try {
                const chunks = await API.get('/api/chunks/view');
                const firstError = chunks.find(chunk => chunk.status === 'error');
                if (!firstError) {
                    showToast('No errored segments found.', 'warning');
                    return;
                }

                const chapter = getChunkChapterName(firstError);
                const chapterToggle = document.getElementById('editor-chapter-only');
                if (chapterToggle) {
                    chapterToggle.checked = true;
                }
                selectedEditorChapter = chapter || WHOLE_PROJECT_CHAPTER_ID;
                syncEditorChapterState(chunks);

                const focused = await focusChunkInEditor(getChunkRef(firstError));
                if (!focused) {
                    showToast('Found an errored segment, but it could not be focused in the editor.', 'warning');
                    return;
                }

                const scopeLabel = chapter ? `chapter "${chapter}"` : 'the current script';
                showToast(`Jumped to the first errored segment in ${scopeLabel}.`, 'info', 2500);
            } catch (e) {
                showToast('Failed to jump to the first errored segment: ' + e.message, 'error');
            }
        };

        window.jumpToFirstChunkForVoice = async (voiceName) => {
            try {
                const chunks = await API.get('/api/chunks/view');
                const norm = s => (s || '').trim().toLowerCase();
                const first = chunks.find(c => norm(c.speaker) === norm(voiceName));
                if (!first) {
                    showToast(`No lines found for "${voiceName}".`, 'warning');
                    return;
                }

                // Switch to editor tab
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                const editorLink = document.querySelector('.nav-link[data-tab="editor"]');
                if (editorLink) editorLink.classList.add('active');
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                const editorTabEl = document.getElementById('editor-tab');
                if (editorTabEl) editorTabEl.style.display = 'block';

                // Scope to the chunk's chapter so only that chapter's rows are rendered
                const chapter = getChunkChapterName(first);
                const chapterToggle = document.getElementById('editor-chapter-only');
                if (chapterToggle) chapterToggle.checked = true;
                const lbl = document.getElementById('editor-chapter-only-label');
                if (lbl) lbl.textContent = 'Generate Chapter';
                selectedEditorChapter = chapter || WHOLE_PROJECT_CHAPTER_ID;
                syncEditorChapterState(chunks);

                ensureAudioQueuePolling();

                const focused = await focusChunkInEditor(getChunkRef(first));
                if (!focused) {
                    showToast(`Found a line for "${voiceName}" but could not focus it in the editor.`, 'warning');
                }
            } catch (e) {
                showToast('Failed to jump to voice: ' + e.message, 'error');
            }
        };

        async function refreshAudioQueueUI() {
            const audioState = await API.get('/api/status/audio');
            latestAudioState = audioState;
            updateAudioQueueControls(audioState);
            renderAudioQueueStatus(audioState);
            renderAudioEstimatePanel(audioState);
            renderEditorProgressBar(cachedChunks, audioState);
            return audioState;
        }

        async function pollAudioQueueOnce() {
            try {
                const audioState = await refreshAudioQueueUI();
                await loadChunks(false);
                const hasAudioWork = Boolean(audioState?.running) || (audioState?.queue || []).length > 0;
                if (hasAudioWork && window.setNavTaskSpinner && !window.getNavTaskSpinnerTab?.()) {
                    window.setNavTaskSpinner('editor');
                }
                if (!hasAudioWork && audioQueuePollTimer) {
                    clearInterval(audioQueuePollTimer);
                    audioQueuePollTimer = null;
                    if (window.releaseNavTaskSpinner) {
                        window.releaseNavTaskSpinner('editor');
                    }
                }
                // Auto Proofread: trigger every 25 completed clips while audio is running
                if (autoProofreadEnabled && audioState?.running) {
                    const processed = Number(audioState?.metrics?.processed_clips) || 0;
                    if (processed - clipsAtLastAutoProofread >= 25) {
                        const proofreadRunning = !!latestProofreadStatus?.running;
                        if (!proofreadRunning) {
                            clipsAtLastAutoProofread = processed;
                            const threshold = getProofreadThreshold();
                            API.post('/api/proofread/auto', { threshold, chapter: '__ALL__' })
                                .catch(err => console.warn('Auto proofread trigger failed:', err));
                        }
                    }
                }
            } catch (e) {
                console.error('Audio queue poll error', e);
                if (audioQueuePollTimer) {
                    clearInterval(audioQueuePollTimer);
                    audioQueuePollTimer = null;
                }
            }
        }

        function ensureAudioQueuePolling() {
            if (audioQueuePollTimer) return;
            audioQueuePollTimer = setInterval(pollAudioQueueOnce, 2000);
            pollAudioQueueOnce();
        }

        function buildSilenceRowHtml(chunk) {
            const chunkRef = getChunkRef(chunk);
            const quotedChunkRef = JSON.stringify(chunkRef);
            const durVal = chunk.silence_duration_s != null ? chunk.silence_duration_s : 1.0;
            return `
                <tr data-id="${escapeHtml(chunkRef)}" data-chapter="${escapeHtml(getChunkChapterName(chunk) || '')}" class="chunk-row chunk-silence-row" style="background:rgba(0,0,0,0.04);">
                    <td class="text-center align-middle" style="white-space:nowrap;">
                        <button class="chunk-expand-btn" onclick='deleteChunk(${quotedChunkRef})' title="Delete silence"><i class="fas fa-trash" style="color:#dc3545;"></i></button>
                    </td>
                    <td colspan="3" class="align-middle ps-3">
                        <span class="text-muted me-2"><i class="fas fa-hourglass-half"></i> Silence</span>
                        <input type="number" class="form-control form-control-sm d-inline-block" style="width:80px;" value="${durVal}" min="0" step="0.1" oninput='validateSilenceDuration(this)' onchange='saveSilenceDuration(${quotedChunkRef}, this)' title="Duration in seconds">
                        <span class="text-muted small ms-1">seconds</span>
                    </td>
                    <td></td>
                    <td></td>
                </tr>
            `;
        }

        function buildChunkRowHtml(chunk) {
            if (chunk.type === 'silence') return buildSilenceRowHtml(chunk);
            const chunkRef = getChunkRef(chunk);
            const quotedChunkRef = JSON.stringify(chunkRef);
            const { statusColor, statusDetail } = getChunkStatusMeta(chunk);
            const audioFingerprint = getChunkAudioFingerprint(chunk);

            const audioPlayer = chunk.audio_path ?
                `<audio class="chunk-audio" data-id="${escapeHtml(chunkRef)}" data-audio-path="${escapeHtml(chunk.audio_path)}" data-audio-fingerprint="${escapeHtml(audioFingerprint)}" controls preload="none" src="${buildChunkAudioSrc(chunk, Date.now().toString())}" style="width: 200px; height: 30px;" onplay='stopOthers(${quotedChunkRef})'></audio>` :
                '<span class="text-muted small">No audio</span>';

            const actionArea = chunk.status === 'generating' ?
                `<div class="progress" style="width: 100px; height: 20px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>
                 </div>` :
                `<button class="btn btn-sm btn-primary" onclick='generateChunk(${quotedChunkRef})'><i class="fas fa-play"></i> Gen</button>`;

            return `
                <tr data-id="${escapeHtml(chunkRef)}" data-chapter="${escapeHtml(getChunkChapterName(chunk) || '')}" class="chunk-row">
                    <td class="text-center align-middle" style="white-space:nowrap;">
                        <button class="chunk-expand-btn" onclick="toggleChunkExpand(this)" title="Expand/collapse"><i class="fas fa-chevron-down"></i></button><button class="chunk-expand-btn" onclick='insertChunkAfter(${quotedChunkRef})' title="Insert line below"><i class="fas fa-plus"></i></button><button class="chunk-expand-btn" onclick='insertSilenceAfter(${quotedChunkRef})' title="Insert silence below"><i class="fas fa-asterisk"></i></button><button class="chunk-expand-btn" onclick='deleteChunk(${quotedChunkRef})' title="Delete line"><i class="fas fa-trash" style="color:#dc3545;"></i></button>
                    </td>
                    <td><input type="text" class="form-control form-control-sm" value="${escapeHtml(chunk.speaker)}" data-editor-field="speaker" oninput='scheduleEditorChunkSave(${quotedChunkRef})' onchange='flushEditorChunkSave(${quotedChunkRef})'></td>
                    <td><textarea class="form-control form-control-sm chunk-text" rows="2" data-editor-field="text" oninput='scheduleEditorChunkSave(${quotedChunkRef})' onchange='flushEditorChunkSave(${quotedChunkRef})'>${escapeHtml(chunk.text)}</textarea></td>
                    <td><textarea class="form-control form-control-sm chunk-instruct" rows="2" data-editor-field="instruct" oninput='scheduleEditorChunkSave(${quotedChunkRef})' onchange='flushEditorChunkSave(${quotedChunkRef})' title="Short TTS direction (3-8 words)">${escapeHtml(chunk.instruct || '')}</textarea></td>
                    <td class="chunk-status-cell"><span class="badge bg-${statusColor}">${chunk.status}</span>${statusDetail}</td>
                    <td>
                        <div class="d-flex align-items-center gap-2">
                            ${actionArea}
                            ${audioPlayer}
                        </div>
                    </td>
                </tr>
            `;
        }

        async function loadChunks(forceFullRedraw = false) {
            const tbody = document.getElementById('chunks-table-body');

            // Show loading only if empty
            if (tbody.children.length === 0 || (tbody.children.length === 1 && tbody.children[0].children.length === 1)) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center">Loading chunks...</td></tr>';
                forceFullRedraw = true;
            }

            try {
                const useChapterScope = !forceFullRedraw
                    && cachedChunks.length > 0
                    && selectedEditorChapter !== WHOLE_PROJECT_CHAPTER_ID;
                const chunks = await API.get(
                    useChapterScope
                        ? `/api/chunks/view?chapter=${encodeURIComponent(selectedEditorChapter)}`
                        : '/api/chunks/view'
                );
                const rawMerged = useChapterScope ? mergeChunkSnapshots(cachedChunks, chunks) : chunks;
                // Filter out any UIDs still being deleted (guards against polling race)
                const mergedChunks = pendingDeleteRefs.size > 0
                    ? rawMerged.filter(c => !pendingDeleteRefs.has(getChunkRef(c)))
                    : rawMerged;
                if (mergedChunks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center">No chunks found. Please generate script first.</td></tr>';
                    cachedChunks = [];
                    cachedVisibleChunkIds = [];
                    cachedProofreadVisibleChunkIds = [];
                    syncEditorChapterState([]);
                    syncProofreadChapterState([]);
                    renderProofreadTable([]);
                    refreshDictionaryCounts([]);
                    return;
                }

                if (!useChapterScope || forceFullRedraw || cachedChunks.length === 0) {
                    syncEditorChapterState(mergedChunks);
                    syncProofreadChapterState(mergedChunks);
                    refreshDictionaryCounts(mergedChunks);
                }

                const proofreadVisibleChunks = getProofreadVisibleChunks(mergedChunks);
                const proofreadRowIds = Array.from(
                    document.querySelectorAll('#proofread-table-body tr[data-proofread-id]')
                ).map(row => row.dataset.proofreadId || '');
                const canIncrementProofread = !forceFullRedraw &&
                    proofreadRowIds.length === proofreadVisibleChunks.length &&
                    proofreadRowIds.every((id, index) => id === getChunkRef(proofreadVisibleChunks[index]));

                if (canIncrementProofread) {
                    const threshold = getProofreadThreshold();
                    proofreadVisibleChunks.forEach(chunk => {
                        const chunkRef = getChunkRef(chunk);
                        const row = document.querySelector(`tr[data-proofread-id="${chunkRef}"]`);
                        const nextFingerprint = getProofreadFingerprint(chunk, threshold);
                        const currentFingerprint = row?.dataset?.proofreadFingerprint || '';
                        if (!row || currentFingerprint !== nextFingerprint) {
                            updateProofreadRow(chunk, threshold);
                        }
                    });
                    cachedProofreadVisibleChunkIds = proofreadVisibleChunks.map(chunk => getChunkRef(chunk));
                } else {
                    renderProofreadTable(mergedChunks);
                }
                renderProofreadTaskStatus(latestProofreadStatus || { running: false, progress: {}, logs: [] });
                const visibleChunks = getVisibleChunks(mergedChunks);
                const tableRowIds = Array.from(tbody.querySelectorAll('tr[data-id]')).map(row => row.dataset.id || '');

                renderEditorProgressBar(mergedChunks, latestAudioState);

                // Skip redraw if playing audio (unless forced)
                if (!forceFullRedraw && (isPlayingSequence || isAudioPlaying())) {
                    // Only update status badges and progress indicators
                    visibleChunks.forEach(chunk => updateChunkRow(chunk));
                    cachedChunks = mergedChunks;
                    cachedVisibleChunkIds = visibleChunks.map(chunk => getChunkRef(chunk));
                    syncTrackedChunkPollers(visibleChunks);

                    return;
                }

                // Check if we can do incremental update
                const canIncrement = !forceFullRedraw &&
                                    tableRowIds.length === visibleChunks.length &&
                                    tableRowIds.every((id, index) => id === getChunkRef(visibleChunks[index]));

                if (canIncrement) {
                    // Incremental update - only update changed rows
                    visibleChunks.forEach(chunk => {
                        const chunkRef = getChunkRef(chunk);
                        const cached = cachedChunks.find(candidate => getChunkRef(candidate) === chunkRef);
                        const cachedError = cached && cached.audio_validation ? cached.audio_validation.error : null;
                        const nextError = chunk.audio_validation ? chunk.audio_validation.error : null;
                        if (!cached || cached.status !== chunk.status || cached.audio_path !== chunk.audio_path || cachedError !== nextError) {
                            updateChunkRow(chunk);
                        }
                    });
                } else {
                    // Full redraw needed
                    tbody.innerHTML = visibleChunks.length === 0
                        ? '<tr><td colspan="6" class="text-center text-muted">No segments in this chapter yet.</td></tr>'
                        : visibleChunks.map(buildChunkRowHtml).join('');
                }

                cachedChunks = mergedChunks;
                cachedVisibleChunkIds = visibleChunks.map(chunk => getChunkRef(chunk));
                syncTrackedChunkPollers(visibleChunks);

            } catch (e) {
                console.error("Error loading chunks:", e);
            }
        }

        window.toggleChunkExpand = (btn) => {
            const row = btn.closest('tr');
            const expanding = !row.classList.contains('expanded');
            row.classList.toggle('expanded');

            row.querySelectorAll('.chunk-text, .chunk-instruct').forEach(ta => {
                if (expanding) {
                    // Auto-size to content
                    ta.style.height = 'auto';
                    ta.style.height = ta.scrollHeight + 'px';
                    ta.style.overflow = 'visible';
                } else {
                    // Collapse back to 2 rows
                    ta.style.height = '';
                    ta.style.overflow = '';
                }
            });
        };

        window.insertChunkAfter = async (id) => {
            try {
                const data = await API.post(`/api/chunks/${id}/insert`, {});
                const newChunk = data.chunk;
                const newRef = getChunkRef(newChunk);
                const targetRow = document.querySelector(`tr[data-id="${id}"]`);
                if (targetRow) {
                    targetRow.insertAdjacentHTML('afterend', buildChunkRowHtml(newChunk));
                } else {
                    // Fallback: target row not found, do a full reload
                    await loadChunks(true);
                    return;
                }
                const insertIdx = cachedChunks.findIndex(c => getChunkRef(c) === String(id));
                if (insertIdx >= 0) {
                    cachedChunks.splice(insertIdx + 1, 0, newChunk);
                } else {
                    cachedChunks.push(newChunk);
                }
                const visibleInsertIdx = cachedVisibleChunkIds.indexOf(String(id));
                if (visibleInsertIdx >= 0) {
                    cachedVisibleChunkIds.splice(visibleInsertIdx + 1, 0, newRef);
                } else {
                    cachedVisibleChunkIds.push(newRef);
                }
            } catch (e) {
                showToast('Failed to insert line: ' + e.message, 'error');
            }
        };

        window.insertSilenceAfter = async (id) => {
            try {
                const data = await API.post(`/api/chunks/${id}/insert_silence`, {});
                const newChunk = data.chunk;
                const newRef = getChunkRef(newChunk);
                const targetRow = document.querySelector(`tr[data-id="${id}"]`);
                if (targetRow) {
                    targetRow.insertAdjacentHTML('afterend', buildSilenceRowHtml(newChunk));
                } else {
                    await loadChunks(true);
                    return;
                }
                const insertIdx = cachedChunks.findIndex(c => getChunkRef(c) === String(id));
                if (insertIdx >= 0) {
                    cachedChunks.splice(insertIdx + 1, 0, newChunk);
                } else {
                    cachedChunks.push(newChunk);
                }
                const visibleInsertIdx = cachedVisibleChunkIds.indexOf(String(id));
                if (visibleInsertIdx >= 0) {
                    cachedVisibleChunkIds.splice(visibleInsertIdx + 1, 0, newRef);
                } else {
                    cachedVisibleChunkIds.push(newRef);
                }
            } catch (e) {
                showToast('Failed to insert silence: ' + e.message, 'error');
            }
        };

        function validateSilenceDuration(input) {
            const val = parseFloat(input.value);
            if (isNaN(val) || val < 0) {
                input.style.borderColor = 'red';
                input.style.boxShadow = '0 0 0 0.2rem rgba(220,53,69,.25)';
            } else {
                input.style.borderColor = '';
                input.style.boxShadow = '';
            }
        }

        window.saveSilenceDuration = async (id, input) => {
            const val = parseFloat(input.value);
            if (isNaN(val) || val < 0) return;
            try {
                await API.post(`/api/chunks/${id}`, { silence_duration_s: val });
                const idx = cachedChunks.findIndex(c => getChunkRef(c) === String(id));
                if (idx >= 0) cachedChunks[idx].silence_duration_s = val;
            } catch (e) {
                showToast('Failed to save silence duration: ' + e.message, 'error');
            }
        };

        window.decomposeLongSegments = async () => {
            if (!document.getElementById('legacy-mode-toggle').checked) {
                showToast('Decompose Long Segments is disabled in new mode. Use the new-mode pipeline to create the script.', 'warning');
                return;
            }
            const button = document.getElementById('btn-decompose-long-segments');
            const chapterOnly = isChapterOnlyEnabled();
            const chapter = chapterOnly && selectedEditorChapter !== WHOLE_PROJECT_CHAPTER_ID
                ? selectedEditorChapter
                : null;
            const scopeLabel = chapter ? `chapter "${chapter}"` : (chapterOnly ? 'the whole project' : 'the entire book');

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Decomposing...';
                }

                const result = await API.post('/api/chunks/decompose_long_segments', {
                    chapter,
                    max_words: 25,
                });
                await loadChunks(true);

                if (result.changed > 0) {
                    showToast(`Decomposed ${result.changed} long segment${result.changed === 1 ? '' : 's'} in ${scopeLabel}.`, 'success');
                } else {
                    showToast(`No unsynthesized segments over 25 words were found in ${scopeLabel}.`, 'warning');
                }
            } catch (e) {
                showToast('Failed to decompose long segments: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-cut me-1"></i>Decompose Long Segments';
                }
            }
        };

        window.mergeOrphans = async () => {
            if (!document.getElementById('legacy-mode-toggle').checked) {
                showToast('Merge Orphans is disabled in new mode. Use the new-mode pipeline to create the script.', 'warning');
                return;
            }
            const button = document.getElementById('btn-merge-orphans');
            const chapterOnly = isChapterOnlyEnabled();
            const chapter = chapterOnly && selectedEditorChapter !== WHOLE_PROJECT_CHAPTER_ID
                ? selectedEditorChapter
                : null;
            const scopeLabel = chapter ? `chapter "${chapter}"` : (chapterOnly ? 'the whole project' : 'the entire book');

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Merging...';
                }

                const result = await API.post('/api/chunks/merge_orphans', {
                    chapter,
                    min_words: 10,
                });
                await loadChunks(true);

                if (result.changed > 0) {
                    showToast(`Merged ${result.changed} orphan segment pair${result.changed === 1 ? '' : 's'} in ${scopeLabel}.`, 'success');
                } else {
                    showToast(`No orphan segments under 10 words could be merged in ${scopeLabel}.`, 'warning');
                }
            } catch (e) {
                showToast('Failed to merge orphan segments: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-compress-alt me-1"></i>Merge Orphans';
                }
            }
        };

        window.repairLegacyProject = async () => {
            const button = document.getElementById('btn-repair-legacy-project');
            const chunks = Array.isArray(cachedChunks) && cachedChunks.length > 0
                ? cachedChunks
                : await API.get('/api/chunks/view');

            if (!chunks.length) {
                showToast('No chunks are loaded to repair.', 'warning');
                return;
            }

            const confirmed = await showConfirm(
                'Repair the project using the editor\'s current full chunk order as the source of truth? This will rewrite the saved chunk structure to match what is currently loaded on screen.'
            );
            if (!confirmed) return;

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Repairing...';
                }

                const result = await API.post('/api/chunks/repair_legacy', { chunks });
                await loadChunks(true);
                showToast(`Repaired legacy chunk order for ${result.total} segment${result.total === 1 ? '' : 's'}.`, 'success');
            } catch (e) {
                showToast('Failed to repair legacy project: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-wrench me-1"></i>Repair Legacy Project';
                }
            }
        };

        window.invalidateStaleAudio = async () => {
            const button = document.getElementById('btn-invalidate-stale-audio');
            const confirmed = await showConfirm(
                'Invalidate only provably stale audio references? This clears chunks that share an audio file with another chunk, while keeping the defensible owner intact.'
            );
            if (!confirmed) return;

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Invalidating...';
                }

                const result = await API.post('/api/chunks/invalidate_stale_audio', {});
                await loadChunks(true);
                if (result.invalidated > 0) {
                    showToast(`Invalidated ${result.invalidated} stale audio reference${result.invalidated === 1 ? '' : 's'} across ${result.duplicate_groups} duplicate group${result.duplicate_groups === 1 ? '' : 's'}.`, 'success');
                } else {
                    showToast('No stale duplicate audio references were found.', 'warning');
                }
            } catch (e) {
                showToast('Failed to invalidate stale audio: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-eraser me-1"></i>Invalidate Stale Audio';
                }
            }
        };

        window.repairLostAudio = async () => {
            const button = document.getElementById('btn-repair-lost-audio');
            const confirmed = await showConfirm(
                'Repair lost audio links from the existing clip files on disk? This will first use safe filename matching, then use local ASR transcription to recover additional clips when the match is strong enough.'
            );
            if (!confirmed) return;

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Relinking...';
                }

                await API.post('/api/chunks/repair_lost_audio', { use_asr: true });
                await pollLogs('repair', 'script-logs');
                await loadChunks(true);
                showToast('Lost audio repair finished.', 'success', 6000);
            } catch (e) {
                showToast('Failed to repair lost audio: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-link me-1"></i>Lost Audio Repair';
                }
            }
        };

        window.repairRejectedAudio = async () => {
            const button = document.getElementById('btn-repair-rejected-audio');
            const confirmed = await showConfirm(
                'Recover audio only from the rejected folder using the current Proofread certainty threshold? This will not reset the active timeline.'
            );
            if (!confirmed) return;

            try {
                if (button) {
                    button.disabled = true;
                    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Recovering...';
                }

                await API.post('/api/chunks/repair_lost_audio', { use_asr: true, rejected_only: true });
                await pollLogs('repair', 'script-logs');
                await loadChunks(true);
                showToast('Rejected audio recovery finished.', 'success', 6000);
            } catch (e) {
                showToast('Failed to recover rejected audio: ' + e.message, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-recycle me-1"></i>Recover Rejected Audio';
                }
            }
        };

        async function repairLegacyProjectBeforeExport() {
            const chunks = Array.isArray(cachedChunks) && cachedChunks.length > 0
                ? cachedChunks
                : await API.get('/api/chunks/view');

            if (!chunks.length) {
                throw new Error('No chunks are loaded to export.');
            }

            await API.post('/api/chunks/repair_legacy', { chunks });
            cachedChunks = chunks;
        }

        async function autoPrepareSegmentsBeforeRender() {
            if (renderPrepComplete) {
                return;
            }

            const isLegacy = document.getElementById('legacy-mode-toggle')?.checked;
            const chapterOnly = isChapterOnlyEnabled();
            const chapter = chapterOnly && selectedEditorChapter !== WHOLE_PROJECT_CHAPTER_ID
                ? selectedEditorChapter
                : null;
            const scopeLabel = chapter ? `chapter "${chapter}"` : (chapterOnly ? 'the whole project' : 'the entire book');

            // Decompose long segments and merge orphans are legacy-only operations.
            // In non-legacy mode the paragraph pipeline already produces well-sized chunks.
            if (!isLegacy) {
                const persistResult = await API.post('/api/render_prep_state', { complete: true });
                renderPrepComplete = !!persistResult.render_prep_complete;
                return;
            }

            const mergeResult = await API.post('/api/chunks/merge_orphans', {
                chapter,
                min_words: 10,
            });
            const decomposeResult = await API.post('/api/chunks/decompose_long_segments', {
                chapter,
                max_words: 25,
            });

            if (mergeResult.changed > 0 || decomposeResult.changed > 0) {
                await loadChunks(true);
            }

            if (mergeResult.changed > 0) {
                showToast(`Merged ${mergeResult.changed} orphan segment pair${mergeResult.changed === 1 ? '' : 's'} before rendering ${scopeLabel}.`, 'success');
            }

            if (decomposeResult.changed > 0) {
                showToast(`Decomposed ${decomposeResult.changed} long segment${decomposeResult.changed === 1 ? '' : 's'} before rendering ${scopeLabel}.`, 'success');
            }

            const persistResult = await API.post('/api/render_prep_state', { complete: true });
            renderPrepComplete = !!persistResult.render_prep_complete;
        }

        let _lastDeleted = null;
        let _undoTimer = null;
        const pendingDeleteRefs = new Set();

        window.deleteChunk = async (id) => {
            const deletedRef = String(id);

            // Optimistically remove the row immediately so the UI feels instant.
            // pendingDeleteRefs guards the polling loop from re-adding the row
            // if a fetch completes before the DELETE API returns.
            pendingDeleteRefs.add(deletedRef);
            const editorRow = document.querySelector(`tr[data-id="${deletedRef}"]`);
            if (editorRow) editorRow.remove();
            const proofreadRow = document.querySelector(`tr[data-proofread-id="${deletedRef}"]`);
            if (proofreadRow) proofreadRow.remove();
            cachedChunks = cachedChunks.filter(c => getChunkRef(c) !== deletedRef);
            cachedVisibleChunkIds = cachedVisibleChunkIds.filter(ref => ref !== deletedRef);
            cachedProofreadVisibleChunkIds = cachedProofreadVisibleChunkIds.filter(ref => ref !== deletedRef);

            try {
                const res = await fetch(`/api/chunks/${id}`, { method: 'DELETE' });
                await API._handleError(res);
                const data = await res.json();

                // Store for undo
                _lastDeleted = { chunk: data.deleted, after_uid: data.restore_after_uid || null };
                clearTimeout(_undoTimer);

                // Show toast with undo action
                const toastId = 'toast-undo-' + Date.now();
                const container = document.getElementById('toast-container');
                container.insertAdjacentHTML('beforeend', `
                    <div id="${toastId}" class="toast align-items-center text-white bg-warning border-0" role="alert">
                        <div class="d-flex">
                            <div class="toast-body text-dark">
                                Line deleted (${data.deleted.speaker}: "${(data.deleted.text || '').substring(0, 40)}...")
                                <a href="#" class="ms-2 fw-bold text-dark" onclick="event.preventDefault(); undoDeleteChunk('${toastId}');">Undo</a>
                            </div>
                            <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast"></button>
                        </div>
                    </div>`);
                const el = document.getElementById(toastId);
                const toast = new bootstrap.Toast(el, { delay: 8000 });
                toast.show();
                el.addEventListener('hidden.bs.toast', () => { el.remove(); });

                // Clear undo data after timeout
                _undoTimer = setTimeout(() => { _lastDeleted = null; }, 8000);
            } catch (e) {
                // Restore the row if the delete failed
                pendingDeleteRefs.delete(deletedRef);
                showToast('Failed to delete line: ' + e.message, 'error');
                await loadChunks(true);
                return;
            }
            pendingDeleteRefs.delete(deletedRef);
        };

        window.undoDeleteChunk = async (toastId) => {
            if (!_lastDeleted) {
                showToast('Nothing to undo', 'warning');
                return;
            }

            try {
                await API.post('/api/chunks/restore', {
                    chunk: _lastDeleted.chunk,
                    after_uid: _lastDeleted.after_uid
                });

                // Dismiss the toast
                const el = document.getElementById(toastId);
                if (el) {
                    const toast = bootstrap.Toast.getInstance(el);
                    if (toast) toast.hide();
                }

                _lastDeleted = null;
                clearTimeout(_undoTimer);
                showToast('Line restored', 'success');
                await loadChunks(true);
            } catch (e) {
                showToast('Undo failed: ' + e.message, 'error');
            }
        };

        function updateDeleteChapterButtonVisibility() {
            const btn = document.getElementById('btn-delete-chapter');
            if (btn) {
                btn.style.display = (selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) ? 'none' : 'inline-block';
            }
        }

        window.deleteChapter = async () => {
            if (selectedEditorChapter === WHOLE_PROJECT_CHAPTER_ID) {
                showToast('Cannot delete the entire project. Select a specific chapter first.', 'warning');
                return;
            }
            const chapter = selectedEditorChapter;
            if (!chapter) {
                showToast('No chapter selected.', 'warning');
                return;
            }
            const chapterChunks = cachedChunks.filter(c => getChunkChapterName(c) === chapter);
            const chunkCount = chapterChunks.length;
            if (chunkCount === 0) {
                showToast('No clips found for this chapter.', 'warning');
                return;
            }
            if (!await showConfirm(`Delete chapter "${chapter}" and all ${chunkCount} clips? This cannot be undone.`)) return;

            const chapterChunkRefs = chapterChunks.map(c => getChunkRef(c));

            // Optimistically remove rows from DOM
            chapterChunkRefs.forEach(ref => {
                pendingDeleteRefs.add(ref);
                const editorRow = document.querySelector(`tr[data-id="${ref}"]`);
                if (editorRow) editorRow.remove();
                const proofreadRow = document.querySelector(`tr[data-proofread-id="${ref}"]`);
                if (proofreadRow) proofreadRow.remove();
            });

            // Filter from caches
            cachedChunks = cachedChunks.filter(c => getChunkChapterName(c) !== chapter);
            cachedVisibleChunkIds = cachedVisibleChunkIds.filter(ref => !chapterChunkRefs.includes(ref));
            if (typeof cachedProofreadVisibleChunkIds !== 'undefined') {
                cachedProofreadVisibleChunkIds = cachedProofreadVisibleChunkIds.filter(ref => !chapterChunkRefs.includes(ref));
            }

            try {
                const res = await fetch(`/api/chapters/${encodeURIComponent(chapter)}`, { method: 'DELETE' });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to delete chapter.', 'error');
                    await loadChunks(true);
                    return;
                }
                const data = await res.json();
                showToast(`Deleted chapter "${chapter}" (${data.deleted_count} clips).`, 'success');
            } catch (e) {
                showToast('Error deleting chapter: ' + e.message, 'error');
                await loadChunks(true);
                return;
            }

            chapterChunkRefs.forEach(ref => pendingDeleteRefs.delete(ref));

            // Clean up undo state if the last deleted chunk was in this chapter
            if (_lastDeleted && chapterChunkRefs.includes(String(_lastDeleted.chunk?.id))) {
                _lastDeleted = null;
                clearTimeout(_undoTimer);
            }

            // Re-sync chapter dropdown and button visibility
            syncEditorChapterState(cachedChunks);
            updateDeleteChapterButtonVisibility();
            await loadChunks(true);
        };

        window.stopOthers = (id) => {
            if (isPlayingSequence) return; // Sequence player handles its own logic
            document.querySelectorAll('audio').forEach(audio => {
                if (audio.dataset.id != id) {
                    audio.pause();
                }
            });
        };

        window.playSequence = async () => {
            isPlayingSequence = true;
            const btn = document.getElementById('btn-play-seq');
            btn.innerHTML = '<i class="fas fa-stop me-1"></i>Stop';
            btn.onclick = stopSequence;
            btn.classList.replace('btn-primary', 'btn-danger');

            const audios = Array.from(document.querySelectorAll('.chunk-audio'));
            if (audios.length === 0) {
                stopSequence();
                return;
            }

            let currentIndex = 0;

            const playNext = () => {
                if (!isPlayingSequence) return;

                // Find next valid audio
                while (currentIndex < audios.length) {
                    const audio = audios[currentIndex];
                    if (audio.getAttribute('src')) {
                        break;
                    }
                    currentIndex++;
                }

                if (currentIndex >= audios.length) {
                    stopSequence();
                    return;
                }

                const audio = audios[currentIndex];
                const tr = audio.closest('tr');

                // Visual feedback
                document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));
                tr.classList.add('table-primary');
                tr.scrollIntoView({ behavior: 'smooth', block: 'center' });

                const playPromise = audio.play();

                if (playPromise !== undefined) {
                    playPromise.catch(e => {
                        console.log("Play failed (empty or skipped):", e);
                        // If play fails, move next
                        currentIndex++;
                        playNext();
                    });
                }

                audio.onended = () => {
                    currentIndex++;
                    playNext();
                };

                audio.onerror = () => {
                     console.log("Audio error, skipping");
                     currentIndex++;
                     playNext();
                }
            };

            playNext();
        };

        window.stopSequence = () => {
            isPlayingSequence = false;
            document.querySelectorAll('audio').forEach(a => {
                a.pause();
                a.currentTime = 0;
                a.onended = null;
            });
            document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));

            const btn = document.getElementById('btn-play-seq');
            if (btn) {
                btn.innerHTML = '<i class="fas fa-play me-1"></i>Play Sequence';
                btn.onclick = playSequence;
                btn.classList.replace('btn-danger', 'btn-primary');
            }
        };

        const _editorChunkSaveTimers = new Map();
        const _editorChunkSavePromises = new Map();
        const EDITOR_CHUNK_SAVE_DEBOUNCE_MS = 350;
        let _dictionaryCountsRefreshTimer = null;

        function scheduleDictionaryCountsRefresh(chunks = cachedChunks) {
            if (_dictionaryCountsRefreshTimer) {
                clearTimeout(_dictionaryCountsRefreshTimer);
            }
            _dictionaryCountsRefreshTimer = setTimeout(() => {
                _dictionaryCountsRefreshTimer = null;
                refreshDictionaryCounts(chunks);
            }, 250);
        }

        function scheduleEditorChunkSave(id) {
            const key = String(id);
            const existingTimer = _editorChunkSaveTimers.get(key);
            if (existingTimer) {
                clearTimeout(existingTimer);
            }
            const timer = setTimeout(() => {
                _editorChunkSaveTimers.delete(key);
                if (_editorChunkSavePromises.has(key)) {
                    return;
                }
                const promise = saveRowEdits(key);
                _editorChunkSavePromises.set(key, promise);
                promise.then(() => {
                    _editorChunkSavePromises.delete(key);
                }).catch(err => {
                    _editorChunkSavePromises.delete(key);
                    console.error('Chunk save failed', err);
                });
            }, EDITOR_CHUNK_SAVE_DEBOUNCE_MS);
            _editorChunkSaveTimers.set(key, timer);
        }

        async function flushEditorChunkSave(id) {
            const key = String(id);
            const existingTimer = _editorChunkSaveTimers.get(key);
            if (existingTimer) {
                clearTimeout(existingTimer);
                _editorChunkSaveTimers.delete(key);
            }
            const inFlight = _editorChunkSavePromises.get(key);
            if (inFlight) {
                return inFlight;
            }
            const promise = saveRowEdits(key);
            _editorChunkSavePromises.set(key, promise);
            try {
                return await promise;
            } finally {
                _editorChunkSavePromises.delete(key);
            }
        }

        async function flushPendingEditorChunkSaves() {
            const pending = Array.from(new Set([
                ..._editorChunkSaveTimers.keys(),
                ..._editorChunkSavePromises.keys(),
            ]));
            if (!pending.length) return;
            for (const id of pending) {
                await flushEditorChunkSave(id);
            }
        }

        window.updateChunk = async (id, field, value) => {
            try {
                const data = {};
                data[field] = value;
                const updatedChunk = await API.post(`/api/chunks/${id}`, data);
                const cached = cachedChunks.find(chunk => getChunkRef(chunk) === String(id));
                if (cached) {
                    Object.assign(cached, updatedChunk || data);
                    updateChunkRow(cached);
                    scheduleDictionaryCountsRefresh(cachedChunks);
                }
            } catch (e) {
                console.error("Update failed", e);
                showToast("Failed to update chunk", 'error');
            }
        };

        // Save all pending edits from a row before generation
        async function saveRowEdits(id) {
            const tr = document.querySelector(`tr[data-id="${id}"]`);
            if (!tr) return;

            const inputs = tr.querySelectorAll('[data-editor-field]');
            const data = {};

            inputs.forEach(input => {
                const field = input.dataset.editorField;
                if (field) {
                    data[field] = input.value;
                }
            });

            // Save all fields at once
            if (Object.keys(data).length > 0) {
                console.log(`Saving chunk ${id} with data:`, data);
                const updatedChunk = await API.post(`/api/chunks/${id}`, data);
                const cached = cachedChunks.find(chunk => getChunkRef(chunk) === String(id));
                if (cached) {
                    Object.assign(cached, updatedChunk || data);
                    updateChunkRow(cached);
                    scheduleDictionaryCountsRefresh(cachedChunks);
                }
                console.log(`Chunk ${id} saved successfully`);
                return updatedChunk;
            }
            return null;
        }

        window.generateChunk = async (id) => {
            try {
                // First, save any pending edits in this row
                await saveRowEdits(id);

                // Skip empty lines
                const tr = document.querySelector(`tr[data-id="${id}"]`);
                if (tr) {
                    const textArea = tr.querySelector('.chunk-text');
                    if (textArea && !textArea.value.trim()) {
                        showToast('Cannot generate audio for an empty line', 'error');
                        return;
                    }
                }

                // Optimistic UI update
                if (tr) {
                    const statusBadge = tr.querySelector('.badge');
                    statusBadge.className = 'badge bg-warning';
                    statusBadge.innerText = 'generating';

                    // Replace button with progress bar
                    const container = tr.querySelector('.d-flex');
                    const btn = container.querySelector('button');
                    if (btn) {
                         const progressBar = document.createElement('div');
                         progressBar.className = 'progress';
                         progressBar.style.width = '100px';
                         progressBar.style.height = '20px';
                         progressBar.innerHTML = '<div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>';
                         container.replaceChild(progressBar, btn);
                    }
                }

                await API.post(`/api/chunks/${id}/generate`, {});
                markChunkGeneratingLocally(id);
                startTrackedChunkStatusPolling(id);
            } catch (e) {
                showToast("Failed to start generation: " + e.message, 'error');
                loadChunks(true); // Revert UI with full redraw
            }
        };

        window.cancelRender = async (skipApi = false) => {
            if (!skipApi) {
                try {
                    const result = await API.post('/api/cancel_audio', {});
                    if (result.status === 'cancelling' || result.status === 'cancelled') {
                        showToast('Audio queue cancellation requested.', 'warning');
                    }
                    ensureAudioQueuePolling();
                } catch (e) { console.error('Cancel error:', e); }
            }
            await refreshAudioQueueUI().catch(err => console.error('Audio queue refresh error', err));
            if (window.releaseNavTaskSpinner) {
                window.releaseNavTaskSpinner('editor');
            }
        };

        window.startRender = (regenerateAll = false) => {
            const mode = document.getElementById('tts-mode').value;
            if (mode === 'external') {
                renderAll(regenerateAll);
            } else {
                renderBatchFast(regenerateAll);
            }
        };

        window.renderAll = async (regenerateAll = false) => {
            try {
                if (!regenerateAll) {
                    await autoPrepareSegmentsBeforeRender();
                }
                const chunks = await API.get('/api/chunks/view');
                const targetChunks = getActionTargetChunks(chunks);
                const toProcess = (regenerateAll ? targetChunks : targetChunks.filter(c => c.status !== 'done'))
                    .filter(c => c.text && c.text.trim());

                if (toProcess.length === 0) {
                    showToast(`No non-empty segments to render in ${getActionScopeLabel()}.`, 'warning');
                    return;
                }

                if (regenerateAll && !await showConfirm(`Regenerate all ${toProcess.length} non-empty segments in ${getActionScopeLabel()}? This will replace existing audio.`)) {
                    return;
                }

                const indices = toProcess.map(c => c.id);

                if (regenerateAll) {
                    // Cancel any running job and immediately reset all target chunks to
                    // pending so the UI reflects the full reset before generation starts.
                    await API.post('/api/chunks/reset_to_pending', { indices });
                    await loadChunks(true);
                }

                // Call batch endpoint for parallel processing
                const response = await API.post('/api/generate_batch', {
                    indices,
                    label: regenerateAll ? `Regenerate ${getActionScopeLabel()}` : `Render pending in ${getActionScopeLabel()}`,
                    scope: getActionScopeLabel()
                });
                if (window.setNavTaskSpinner) {
                    window.setNavTaskSpinner('editor');
                }
                console.log(`Batch generation queued: ${response.total_chunks} chunks with ${response.workers} workers`);
                showToast(`Queued job #${response.job_id} for ${response.total_chunks} segment${response.total_chunks === 1 ? '' : 's'}.`, 'success');
                ensureAudioQueuePolling();

            } catch (e) {
                console.error("Render All error:", e);
                if (window.releaseNavTaskSpinner) {
                    window.releaseNavTaskSpinner('editor');
                }
                showToast("Error during batch rendering: " + e.message, 'error');
            }
        };

        window.renderBatchFast = async (regenerateAll = false) => {
            try {
                if (!regenerateAll) {
                    await autoPrepareSegmentsBeforeRender();
                }
                const chunks = await API.get('/api/chunks/view');
                const targetChunks = getActionTargetChunks(chunks);
                const toProcess = (regenerateAll ? targetChunks : targetChunks.filter(c => c.status !== 'done'))
                    .filter(c => c.text && c.text.trim());

                if (toProcess.length === 0) {
                    showToast(`No non-empty segments to render in ${getActionScopeLabel()}.`, 'warning');
                    return;
                }

                const indices = toProcess.map(c => c.id);

                if (regenerateAll) {
                    // Cancel any running job and immediately reset all target chunks to
                    // pending so the UI reflects the full reset before generation starts.
                    await API.post('/api/chunks/reset_to_pending', { indices });
                    await loadChunks(true);
                }

                const response = await API.post('/api/generate_batch_fast', {
                    indices,
                    label: regenerateAll ? `Batch regenerate ${getActionScopeLabel()}` : `Batch render pending in ${getActionScopeLabel()}`,
                    scope: getActionScopeLabel()
                });
                if (window.setNavTaskSpinner) {
                    window.setNavTaskSpinner('editor');
                }
                console.log(`Fast batch queued: ${response.total_chunks} chunks (batch_size=${response.batch_size}, seed=${response.batch_seed})`);
                showToast(`Queued fast job #${response.job_id} for ${response.total_chunks} segment${response.total_chunks === 1 ? '' : 's'}.`, 'success');
                ensureAudioQueuePolling();

            } catch (e) {
                console.error("Batch Fast error:", e);
                if (window.releaseNavTaskSpinner) {
                    window.releaseNavTaskSpinner('editor');
                }
                showToast("Error during batch rendering: " + e.message, 'error');
            }
        };

        async function populateExportChapterSelect() {
            const select = document.getElementById('export-chapter-select');
            if (!select) return;
            try {
                const chunks = await API.get('/api/chunks');
                const seen = new Set();
                const opts = [new Option('Full Project', '')];
                for (const chunk of chunks) {
                    const ch = (chunk.chapter || '').trim();
                    if (ch && !seen.has(ch)) {
                        seen.add(ch);
                        opts.push(new Option(ch, ch));
                    }
                }
                const current = select.value;
                select.replaceChildren(...opts);
                if ([...select.options].some(o => o.value === current)) select.value = current;
            } catch (e) { /* non-fatal */ }
        }
        window.populateExportChapterSelect = populateExportChapterSelect;

        document.getElementById('btn-merge').addEventListener('click', async () => {
             const chapterSelect = document.getElementById('export-chapter-select');
             const chapter = chapterSelect ? chapterSelect.value : '';
             const scopeLabel = chapter ? `chapter "${chapter}"` : 'the full project';
             if (!await showConfirm(`Merge ${scopeLabel} into an MP3?`)) return;

             try {
                 let exportConfig = null;
                 if (window.persistExportConfigFromUI) {
                     exportConfig = await window.persistExportConfigFromUI();
                 }
                 await repairLegacyProjectBeforeExport();
                 await API.post('/api/merge', { export: exportConfig || undefined, chapter: chapter || undefined });
                 markTaskActionRequested('audio', 'merge');
                 // Switch to Export tab and poll
                 document.querySelector('[data-tab="audio"]').click();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Merge failed: " + e.message, 'error');
             }
        });

        document.getElementById('btn-merge-optimized').addEventListener('click', async () => {
             const chapterSelect = document.getElementById('export-chapter-select');
             const chapter = chapterSelect ? chapterSelect.value : '';

             if (chapter) {
                 // Single chapter: optimized export not supported, route to merge
                 if (!await showConfirm(`Export chapter "${chapter}" as MP3? (Optimized export targets the full project only.)`)) return;
                 try {
                     let exportConfig = null;
                     if (window.persistExportConfigFromUI) {
                         exportConfig = await window.persistExportConfigFromUI();
                     }
                     await repairLegacyProjectBeforeExport();
                     await API.post('/api/merge', { export: exportConfig || undefined, chapter });
                     markTaskActionRequested('audio', 'merge');
                     document.querySelector('[data-tab="audio"]').click();
                     pollLogs('audio', 'audio-logs');
                 } catch (e) {
                     showToast("Export failed: " + e.message, 'error');
                 }
                 return;
             }

             if (!await showConfirm("Create an optimized export zip with audiobook parts capped at about 2 hours each?")) return;

             try {
                 let exportConfig = null;
                 if (window.persistExportConfigFromUI) {
                     exportConfig = await window.persistExportConfigFromUI();
                 }
                 await repairLegacyProjectBeforeExport();
                 await API.post('/api/merge_optimized', { export: exportConfig || undefined });
                 markTaskActionRequested('audio', 'optimized');
                 document.querySelector('[data-tab="audio"]').click();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Optimized export failed: " + e.message, 'error');
             }
        });

        document.getElementById('btn-clear-trim-cache').addEventListener('click', async () => {
             if (!await showConfirm("Clear cached trimmed clips? The next export will recompute trim for all clips.")) return;

             try {
                 const result = await API.post('/api/trim_cache/clear');
                 const removedFiles = Number(result?.removed_files || 0);
                 const removedBytes = Number(result?.removed_bytes || 0);
                 showToast(`Trim cache cleared (${removedFiles} file${removedFiles === 1 ? '' : 's'}, ${formatBytes(removedBytes)}).`, 'success');
             } catch (e) {
                 showToast("Failed to clear trim cache: " + e.message, 'error');
             }
        });

        document.getElementById('btn-trim-sanity-first-clip').addEventListener('click', async () => {
             try {
                 let exportConfig = null;
                 if (window.persistExportConfigFromUI) {
                     exportConfig = await window.persistExportConfigFromUI();
                 }
                 const result = await API.post('/api/trim_sanity/first_clip', { export: exportConfig || undefined });
                 const clipPath = result?.clip?.audio_path || 'unknown';
                 const trimmedFlag = result?.trim_info?.trimmed ? 'yes' : 'no';
                 showToast(`Sanity trim ready for ${clipPath} (trimmed: ${trimmedFlag}). Downloading...`, 'success');
                 const downloadUrl = `${result.download_url}?t=${Date.now()}`;
                 const a = document.createElement('a');
                 a.href = downloadUrl;
                 a.download = 'trim_sanity_first_clip.wav';
                 document.body.appendChild(a);
                 a.click();
                 a.remove();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Failed to run sanity trim: " + e.message, 'error');
             }
        });

        document.getElementById('btn-assemble-sanity-first5').addEventListener('click', async () => {
             try {
                 let exportConfig = null;
                 if (window.persistExportConfigFromUI) {
                     exportConfig = await window.persistExportConfigFromUI();
                 }
                 const result = await API.post('/api/assemble_sanity/first5', { export: exportConfig || undefined });
                 const clipCount = Number(result?.clip_count || 0);
                 showToast(`Assembled first ${clipCount} clips. Downloading...`, 'success');
                 const downloadUrl = `${result.download_url}?t=${Date.now()}`;
                 const a = document.createElement('a');
                 a.href = downloadUrl;
                 a.download = 'assemble_sanity_first5.wav';
                 document.body.appendChild(a);
                 a.click();
                 a.remove();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Failed to assemble first 5 clips: " + e.message, 'error');
             }
        });

        document.getElementById('btn-assemble-sanity-first5-normalized').addEventListener('click', async () => {
             try {
                 let exportConfig = null;
                 if (window.persistExportConfigFromUI) {
                     exportConfig = await window.persistExportConfigFromUI();
                 }
                 const result = await API.post('/api/assemble_sanity/first5_normalized', { export: exportConfig || undefined });
                 const clipCount = Number(result?.clip_count || 0);
                 showToast(`Assembled and normalized first ${clipCount} clips. Downloading...`, 'success');
                 const downloadUrl = `${result.download_url}?t=${Date.now()}`;
                 const a = document.createElement('a');
                 a.href = downloadUrl;
                 a.download = 'assemble_sanity_first5_normalized.wav';
                 document.body.appendChild(a);
                 a.click();
                 a.remove();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Failed to assemble+normalize first 5 clips: " + e.message, 'error');
             }
        });
