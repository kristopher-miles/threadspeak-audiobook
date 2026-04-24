        // --- Clone Voice Upload Handlers ---

        window.uploadCloneVoice = (btn) => {
            const card = btn.closest('.card-body');
            card.querySelector('.clone-voice-file-input').click();
        };

        window.handleCloneVoiceUpload = async (input) => {
            const file = input.files[0];
            if (!file) return;
            input.value = '';
            const card = input.closest('.card-body');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/clone_voices/upload', { method: 'POST', body: formData });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Upload failed', 'error'); return; }
                const result = await res.json();

                // Refresh cache and rebuild voice cards
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
                const uploadedVoice = (window._cloneVoicesCache || []).find(v => v.id === result.voice_id) || result;
                if (card && uploadedVoice?.voice_id) {
                    uploadedVoice.id = uploadedVoice.voice_id;
                }
                if (card && uploadedVoice?.id) {
                    const select = card.querySelector('.designed-voice-select');
                    const refAudio = card.querySelector('.ref-audio');
                    const refText = card.querySelector('.ref-text');
                    const playBtn = card.querySelector('.clone-play-btn');
                    const deleteBtn = card.querySelector('.clone-delete-btn');
                    if (select) {
                        let option = Array.from(select.options).find(o => o.value === `clone:${uploadedVoice.id}`);
                        if (!option) {
                            option = new Option(uploadedVoice.name || file.name.replace(/\.[^.]+$/, ''), `clone:${uploadedVoice.id}`);
                            select.add(option);
                        }
                        select.value = `clone:${uploadedVoice.id}`;
                    }
                    if (refAudio) {
                        refAudio.value = `clone_voices/${uploadedVoice.filename || result.filename}`;
                        refAudio.readOnly = true;
                    }
                    if (refText) {
                        refText.value = uploadedVoice.sample_text || result.sample_text || '';
                    }
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'inline-block';
                    if (typeof window.updateCloneActionButtonForCard === 'function') {
                        window.updateCloneActionButtonForCard(card);
                    }
                    await saveVoicesNow({ promptConfirmation: false, retryOnNetworkFailure: true });
                }
                await loadVoices();

                showToast(`Uploaded "${file.name}"`, 'success');
            } catch (e) {
                showToast('Upload failed: ' + e.message, 'error');
            }
        };

        window.playCloneVoice = (btn) => {
            const card = btn.closest('.card-body');
            const refAudio = card.querySelector('.ref-audio').value;
            if (refAudio) {
                playSharedPreviewAudio(`/${refAudio}?t=${Date.now()}`).catch((e) => {
                    if (isPreviewAbortError(e)) return;
                    showToast(`Preview playback failed: ${e.message}`, 'error');
                });
            }
        };

        window.deleteCloneVoice = async (btn) => {
            if (!await showConfirm('Delete this uploaded clone voice?')) return;
            const card = btn.closest('.card-body');
            const select = card.querySelector('.designed-voice-select');
            const val = select.value;
            if (!val.startsWith('clone:')) return;
            const voiceId = val.substring(6);

            try {
                const res = await fetch(`/api/clone_voices/${encodeURIComponent(voiceId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete', 'error'); return; }

                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
                await loadVoices();
                showToast('Clone voice deleted', 'success');
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        };
