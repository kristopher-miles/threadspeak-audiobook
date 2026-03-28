        // --- New Mode stubs ---
        document.getElementById('btn-process-paragraphs-v2').addEventListener('click', async () => {
            await lockGenerationMode('process_paragraphs_v2');
            try {
                await API.post('/api/process_paragraphs', {});
                pollLogs('process_paragraphs', 'script-logs');
                showToast('Paragraph processing started.', 'success');
            } catch (e) {
                showToast(e.message || 'Failed to start paragraph processing.', 'error', 7000);
            }
        });
        document.getElementById('btn-assign-dialogue-v2').addEventListener('click', async () => {
            try {
                await API.post('/api/assign_dialogue', {});
                pollLogs('assign_dialogue', 'script-logs');
                showToast('Dialogue assignment started.', 'success');
            } catch (e) {
                showToast(e.message || 'Failed to start dialogue assignment.', 'error', 7000);
            }
        });
        document.getElementById('btn-extract-temperament-v2').addEventListener('click', async () => {
            try {
                await API.post('/api/extract_temperament', {});
                pollLogs('extract_temperament', 'script-logs');
                showToast('Temperament extraction started.', 'success');
            } catch (e) {
                showToast(e.message || 'Failed to start temperament extraction.', 'error', 7000);
            }
        });
        document.getElementById('btn-create-script-v2').addEventListener('click', async () => {
            await lockGenerationMode('create_script_v2');
            try {
                const info = await API.get('/api/script_info');
                if (info.entry_count > 0) {
                    if (!confirm(`An existing script with ${info.entry_count} lines was found.\n\nThis will permanently erase the current script and all voice assignments.\n\nContinue?`)) {
                        return;
                    }
                    await API.post('/api/reset_new_mode', {});
                    // Clear the editor display immediately so the old chunks don't linger
                    const tbody = document.getElementById('chunks-table-body');
                    if (tbody) tbody.innerHTML = '';
                }
            } catch (e) {
                showToast(e.message || 'Failed to reset existing script.', 'error', 7000);
                return;
            }
            try {
                await API.post('/api/create_script', {});
                pollLogs('create_script', 'script-logs');
                showToast('Script creation started.', 'success');
            } catch (e) {
                showToast(e.message || 'Failed to start script creation.', 'error', 7000);
            }
        });
        function updateNewModeWorkflowButtons(status) {
            const startBtn = document.getElementById('btn-process-script-v2');
            const pauseBtn = document.getElementById('btn-pause-processing-v2');
            if (!startBtn || !pauseBtn) return;
            const running = !!status?.running;
            const paused = !!status?.paused;
            const activeRun = running && !paused;
            startBtn.disabled = activeRun;
            startBtn.innerHTML = running && !paused
                ? '<i class="fas fa-spinner fa-spin me-1"></i>Processing...'
                : paused
                    ? '<i class="fas fa-play-circle me-1"></i>Resume Processing'
                    : '<i class="fas fa-play-circle me-1"></i>Process Script';
            pauseBtn.disabled = !activeRun;
            startBtn.style.display = activeRun ? 'none' : '';
            pauseBtn.style.display = activeRun ? '' : 'none';
        }

        document.getElementById('btn-process-script-v2').addEventListener('click', async () => {
            await lockGenerationMode('process_script_v2');
            try {
                const processVoices = document.getElementById('process-voices-toggle-v2')?.checked !== false;
                const generateAudio = document.getElementById('generate-audio-toggle-v2')?.checked === true;
                await API.post('/api/new_mode_workflow/start', { process_voices: processVoices, generate_audio: generateAudio });
                pollLogs('new_mode_workflow', 'script-logs');
                updateNewModeWorkflowButtons({ running: true, paused: false });
            } catch (e) {
                showToast(e.message || 'Failed to start processing.', 'error', 7000);
            }
        });
        document.getElementById('btn-pause-processing-v2').addEventListener('click', async () => {
            try {
                const result = await API.post('/api/new_mode_workflow/pause', {});
                if (result.status === 'pause_requested') {
                    showToast('Pause requested. Current stage will finish before stopping.', 'warning');
                }
            } catch (e) {
                showToast(e.message || 'Failed to request pause.', 'error', 7000);
            }
        });

        // Restore workflow button state on page load
        (async () => {
            try {
                const status = await API.get('/api/status/new_mode_workflow');
                updateNewModeWorkflowButtons(status);
                if (status.running) {
                    pollLogs('new_mode_workflow', 'script-logs');
                }
            } catch (e) {
                console.warn('Could not fetch new_mode_workflow status', e);
            }
        })();

