        // --- New Mode stubs ---

        const _stepIconMap = {
            process_paragraphs:  'icon-process-paragraphs',
            assign_dialogue:     'icon-assign-dialogue',
            extract_temperament: 'icon-extract-temperament',
            create_script:       'icon-create-script',
        };

        function setStepIcon(iconId, state) {
            const el = document.getElementById(iconId);
            if (!el || el.dataset.stepState === state) return;
            el.dataset.stepState = state;
            const icons = {
                not_started: '<i class="fa-regular fa-circle text-secondary"></i>',
                incomplete:  '<i class="fas fa-circle-exclamation text-warning"></i>',
                in_progress: '<span class="step-spinner text-primary"></span>',
                complete:    '<i class="fas fa-circle-check text-success"></i>',
            };
            el.innerHTML = icons[state] || icons.not_started;
        }

        function applyStepButtonStates(status) {
            const btnPP  = document.getElementById('btn-process-paragraphs-v2');
            const btnAD  = document.getElementById('btn-assign-dialogue-v2');
            const btnET  = document.getElementById('btn-extract-temperament-v2');
            const btnCS  = document.getElementById('btn-create-script-v2');
            if (btnPP) btnPP.disabled = !status.has_input_file;
            if (btnAD) btnAD.disabled = status.process_paragraphs !== 'complete';
            if (btnET) btnET.disabled = status.assign_dialogue !== 'complete';
            if (btnCS) btnCS.disabled = status.extract_temperament !== 'complete';
        }

        async function loadPipelineStepIcons() {
            try {
                const res = await fetch('/api/pipeline_step_status');
                if (!res.ok) return;
                const status = await res.json();
                setStepIcon('icon-process-paragraphs', status.process_paragraphs);
                setStepIcon('icon-assign-dialogue', status.assign_dialogue);
                setStepIcon('icon-extract-temperament', status.extract_temperament);
                setStepIcon('icon-create-script', status.create_script);
                applyStepButtonStates(status);
                if (window.updatePipelineTabLocks) {
                    const isLegacy = !!document.getElementById('legacy-mode-toggle')?.checked;
                    window.updatePipelineTabLocks(isLegacy, status.create_script === 'complete');
                }
            } catch (e) { /* silent */ }
        }

        document.getElementById('btn-process-paragraphs-v2').addEventListener('click', async () => {
            await lockGenerationMode('process_paragraphs_v2');
            try {
                setStepIcon('icon-process-paragraphs', 'in_progress');
                await API.post('/api/process_paragraphs', {});
                pollLogs('process_paragraphs', 'script-logs');
                showToast('Paragraph processing started.', 'success');
            } catch (e) {
                setStepIcon('icon-process-paragraphs', 'not_started');
                showToast(e.message || 'Failed to start paragraph processing.', 'error', 7000);
            }
        });
        document.getElementById('btn-assign-dialogue-v2').addEventListener('click', async () => {
            try {
                setStepIcon('icon-assign-dialogue', 'in_progress');
                await API.post('/api/assign_dialogue', {});
                pollLogs('assign_dialogue', 'script-logs');
                showToast('Dialogue assignment started.', 'success');
            } catch (e) {
                setStepIcon('icon-assign-dialogue', 'not_started');
                showToast(e.message || 'Failed to start dialogue assignment.', 'error', 7000);
            }
        });
        document.getElementById('btn-extract-temperament-v2').addEventListener('click', async () => {
            try {
                setStepIcon('icon-extract-temperament', 'in_progress');
                await API.post('/api/extract_temperament', {});
                pollLogs('extract_temperament', 'script-logs');
                showToast('Temperament extraction started.', 'success');
            } catch (e) {
                setStepIcon('icon-extract-temperament', 'not_started');
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
                setStepIcon('icon-create-script', 'in_progress');
                await API.post('/api/create_script', {});
                pollLogs('create_script', 'script-logs');
                showToast('Script creation started.', 'success');
            } catch (e) {
                setStepIcon('icon-create-script', 'not_started');
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

            // Update step icons based on workflow stage progress
            const completedStages = Array.isArray(status?.completed_stages) ? status.completed_stages : [];
            const currentStage = status?.current_stage || null;
            for (const [stage, iconId] of Object.entries(_stepIconMap)) {
                if (completedStages.includes(stage)) {
                    setStepIcon(iconId, 'complete');
                } else if (stage === currentStage && activeRun) {
                    setStepIcon(iconId, 'in_progress');
                }
            }
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

        // Restore workflow button state and step icons on page load
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
            loadPipelineStepIcons().catch(err => console.warn('Icon load failed', err));
        })();

