        // --- Voice Designer ---
        window._designedVoicesCache = [];
        window._cloneVoicesCache = [];
        window._currentPreviewFile = null;

        async function loadDesignedVoices() {
            try {
                const voices = await API.get('/api/voice_design/list');
                window._designedVoicesCache = voices;
                const container = document.getElementById('designed-voices-list');

                if (!voices.length) {
                    container.innerHTML = '<p class="text-muted mb-0">No designed voices yet. Generate and save a preview above.</p>';
                    return;
                }

                container.innerHTML = `
                    <table class="table table-sm table-hover mb-0">
                        <thead><tr><th>Name</th><th>Description</th><th style="width:120px">Actions</th></tr></thead>
                        <tbody>
                            ${voices.map(v => `
                                <tr>
                                    <td><strong>${v.name}</strong></td>
                                    <td class="text-muted" style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${v.description}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary me-1" onclick="playDesignedVoice('${v.filename}')" title="Play"><i class="fas fa-play"></i></button>
                                        <button class="btn btn-sm btn-outline-danger" onclick="deleteDesignedVoice('${v.id}')" title="Delete"><i class="fas fa-trash"></i></button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>`;
            } catch (e) {
                console.error('Failed to load designed voices:', e);
            }
        }

        window.generateDesignPreview = async () => {
            const description = document.getElementById('design-description').value.trim();
            const sampleText = document.getElementById('design-sample-text').value.trim();
            const statusEl = document.getElementById('design-status');
            const previewContainer = document.getElementById('design-preview-container');

            if (!description) { showToast('Please enter a voice description.', 'warning'); return; }
            if (!sampleText) { showToast('Please enter sample text.', 'warning'); return; }

            const btn = document.getElementById('btn-design-preview');
            btn.disabled = true;
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating preview (this may take a moment on first run)...';
            previewContainer.style.display = 'none';

            try {
                const result = await API.post('/api/voice_design/preview', {
                    description: description,
                    sample_text: sampleText
                });

                const audio = document.getElementById('design-preview-audio');
                audio.src = result.audio_url + '?t=' + Date.now();
                previewContainer.style.display = 'block';
                statusEl.innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Preview ready</span>';

                // Extract filename from URL for save
                window._currentPreviewFile = result.audio_url.split('/').pop().split('?')[0];
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>Failed: ${e.message}</span>`;
            } finally {
                btn.disabled = false;
            }
        };

        window.saveDesignedVoice = async () => {
            const name = document.getElementById('design-voice-name').value.trim();
            if (!name) { showToast('Please enter a name for the voice.', 'warning'); return; }
            if (!window._currentPreviewFile) { showToast('Generate a preview first.', 'warning'); return; }

            try {
                await API.post('/api/voice_design/save', {
                    name: name,
                    description: document.getElementById('design-description').value.trim(),
                    sample_text: document.getElementById('design-sample-text').value.trim(),
                    preview_file: window._currentPreviewFile
                });
                document.getElementById('design-voice-name').value = '';
                loadDesignedVoices();
            } catch (e) {
                showToast('Error saving voice: ' + e.message, 'error');
            }
        };

        window.playDesignedVoice = (filename) => {
            playSharedPreviewAudio(`/designed_voices/${filename}?t=${Date.now()}`).catch((e) => {
                if (isPreviewAbortError(e)) return;
                showToast(`Preview playback failed: ${e.message}`, 'error');
            });
        };

        window.deleteDesignedVoice = async (voiceId) => {
            if (!await showConfirm('Delete this designed voice?')) return;
            try {
                const res = await fetch(`/api/voice_design/${encodeURIComponent(voiceId)}`, {method: 'DELETE'});
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadDesignedVoices();
            } catch (e) {
                showToast('Error deleting voice: ' + e.message, 'error');
            }
        };

        window.onDesignedVoiceSelect = (select) => {
            const card = select.closest('.card-body');
            const refText = card.querySelector('.ref-text');
            const refAudio = card.querySelector('.ref-audio');
            const playBtn = card.querySelector('.clone-play-btn');
            const deleteBtn = card.querySelector('.clone-delete-btn');
            const val = select.value;

            if (val === '' || val === '__manual__') {
                refAudio.readOnly = false;
                if (val === '__manual__') {
                    refAudio.value = '';
                    refText.value = '';
                }
                if (playBtn) playBtn.style.display = 'none';
                if (deleteBtn) deleteBtn.style.display = 'none';
                refAudio.focus();
                return;
            }

            if (val.startsWith('clone:')) {
                const voiceId = val.substring(6);
                const voice = (window._cloneVoicesCache || []).find(v => v.id === voiceId);
                if (voice) {
                    refAudio.value = `clone_voices/${voice.filename}`;
                    refText.value = voice.sample_text || '';
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'inline-block';
                }
            } else if (val.startsWith('design:')) {
                const voiceId = val.substring(7);
                const voice = (window._designedVoicesCache || []).find(v => v.id === voiceId);
                if (voice) {
                    refAudio.value = `designed_voices/${voice.filename}`;
                    refText.value = voice.sample_text;
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'none';
                }
            } else {
                // Legacy: plain voice ID (backward compat with old designed voice values)
                const voice = (window._designedVoicesCache || []).find(v => v.id === val);
                if (voice) {
                    refAudio.value = `designed_voices/${voice.filename}`;
                    refText.value = voice.sample_text;
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'none';
                }
            }
            debouncedSaveVoices();
        };

