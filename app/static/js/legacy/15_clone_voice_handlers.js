        // --- Clone Voice Upload Handlers ---

        window.uploadCloneVoice = (btn) => {
            const card = btn.closest('.card-body');
            card.querySelector('.clone-voice-file-input').click();
        };

        window.handleCloneVoiceUpload = async (input) => {
            const file = input.files[0];
            if (!file) return;
            input.value = '';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/clone_voices/upload', { method: 'POST', body: formData });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Upload failed', 'error'); return; }
                const result = await res.json();

                // Refresh cache and rebuild voice cards
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
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

