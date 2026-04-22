        // --- New Mode stubs ---

        const _stepIconMap = {
            process_paragraphs:  'icon-process-paragraphs',
            assign_dialogue:     'icon-assign-dialogue',
            extract_temperament: 'icon-extract-temperament',
            create_script:       'icon-create-script',
        };

        function setFullCastToggleLabel(isFullCast) {
            const label = document.getElementById('full-cast-toggle-label-v2');
            if (!label) return;
            label.textContent = isFullCast ? 'Full Cast' : 'Narrated';
        }

        function getFullCastEnabled() {
            return document.getElementById('full-cast-toggle-v2')?.checked !== false;
        }

        function syncFullCastToggleFromOptions(options) {
            if (!options || !Object.prototype.hasOwnProperty.call(options, 'full_cast')) {
                setFullCastToggleLabel(getFullCastEnabled());
                return;
            }
            const toggle = document.getElementById('full-cast-toggle-v2');
            if (toggle) {
                toggle.checked = options.full_cast !== false;
            }
            setFullCastToggleLabel(options.full_cast !== false);
        }

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
                await API.post('/api/assign_dialogue', { full_cast: getFullCastEnabled() });
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
            if (typeof window.flushSetupConfig === 'function') {
                await window.flushSetupConfig();
            }
            try {
                const info = await API.get('/api/script_info');
                const hasExistingScriptState = (info.entry_count || 0) > 0;
                const hasExistingVoiceState = !!info.has_voice_config;
                const hasExistingAudioState = !!info.has_voicelines;
                if (hasExistingScriptState || hasExistingVoiceState || hasExistingAudioState) {
                    const existingParts = [];
                    if (hasExistingScriptState) existingParts.push(`script with ${info.entry_count} lines`);
                    if (hasExistingVoiceState) existingParts.push(`voice assignments (${info.voice_count || 0})`);
                    if (hasExistingAudioState) existingParts.push('generated voice clips');
                    const summary = existingParts.length ? existingParts.join(', ') : 'existing generated content';
                    if (!confirm(`Existing ${summary} found.\n\nThis will permanently erase the current script and all generated voice clips.\n\nContinue?`)) {
                        return;
                    }
                    let preserveVoices = false;
                    if (hasExistingVoiceState) {
                        const deleteVoices = confirm(
                            'Delete saved voice assignments too?\n\n'
                            + 'Click OK to delete voice assignments.\n'
                            + 'Click Cancel to keep them and skip voice re-import.'
                        );
                        preserveVoices = !deleteVoices;
                    }
                    await API.post('/api/reset_new_mode', { preserve_voices: preserveVoices });
                    window.resetPipelineTabLocks?.();
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
            const completedStages = Array.isArray(status?.completed_stages) ? status.completed_stages : [];
            const scriptCreated = completedStages.includes('create_script');
            if (window.updatePipelineTabLocks) {
                const isLegacy = !!document.getElementById('legacy-mode-toggle')?.checked;
                window.updatePipelineTabLocks(isLegacy, scriptCreated);
            }
            if (scriptCreated) {
                window.primeVoicesForScriptWorkflow?.();
            } else {
                window.resetVoiceAliasWorkflowPrime?.();
            }
            syncFullCastToggleFromOptions(status?.options);
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
                if (typeof window.flushSetupConfig === 'function') {
                    await window.flushSetupConfig();
                }
                const processVoices = document.getElementById('process-voices-toggle-v2')?.checked !== false;
                const fullCast = getFullCastEnabled();
                const generateAudio = document.getElementById('generate-audio-toggle-v2')?.checked === true;
                await API.post('/api/new_mode_workflow/start', { process_voices: processVoices, generate_audio: generateAudio, full_cast: fullCast });
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
        document.getElementById('full-cast-toggle-v2')?.addEventListener('change', async (event) => {
            const fullCast = event?.target?.checked !== false;
            setFullCastToggleLabel(fullCast);
            try {
                await API.post('/api/new_mode_workflow/options', { full_cast: fullCast });
            } catch (e) {
                showToast(e.message || 'Failed to save script mode.', 'error', 7000);
            }
        });

        // Restore workflow button state and step icons on page load
        (async () => {
            setFullCastToggleLabel(getFullCastEnabled());
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
