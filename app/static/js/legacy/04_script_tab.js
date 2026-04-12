        // --- Script Tab ---
        document.getElementById('btn-reset-project-script').addEventListener('click', window.resetProject);
        document.getElementById('btn-reset-project-script-v2').addEventListener('click', window.resetProject);
        document.getElementById('file-upload').addEventListener('change', async () => {
            const fileInput = document.getElementById('file-upload');
            const statusEl = document.getElementById('upload-status');
            if (fileInput.files.length === 0) return;

            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Loading file...</span>';
            try {
                const res = await API.upload(fileInput.files[0]);
                const chapterSuffix = ['epub', 'docx'].includes(res.source_type) && Number.isInteger(res.chapter_count)
                    ? ` (${res.chapter_count} chapter${res.chapter_count === 1 ? '' : 's'} detected)`
                    : '';
                statusEl.innerHTML = `<span class="text-success"><i class="fas fa-check me-1"></i>Loaded: ${res.filename}${chapterSuffix}</span>`;
                document.getElementById('file-upload-section').style.display = 'none';
                document.getElementById('script-help-text').style.display = '';
                loadPipelineStepIcons().catch(() => {});
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>Failed to load file: ${e.message}</span>`;
                document.getElementById('file-upload-section').style.display = '';
            }
        });

        async function startGenerateScript() {
            const fileInput = document.getElementById('file-upload');
            const statusEl = document.getElementById('upload-status');

            // Check if a file has been loaded (status shows success) or if one was previously uploaded
            const hasLoadedFile = statusEl.innerHTML.includes('text-success');

            if (!hasLoadedFile && fileInput.files.length === 0) {
                statusEl.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle me-1"></i>Please select a text, EPUB, or DOCX file first using the file picker above.</span>';
                throw new Error('No input file selected');
            }
            if (typeof window.flushSetupConfig === 'function') {
                await window.flushSetupConfig();
            }

            const ingestionDecision = await resolveScriptIngestionDecision({ continueWorkflow: false });
            if (ingestionDecision.skipImport) {
                updateScriptTaskButtons({ running: false });
                showToast('Existing imported script preserved. Script generation was skipped.', 'info', 5000);
                return { status: 'skipped' };
            }

            updateScriptTaskButtons({ running: true });
            await API.post('/api/generate_script', {
                force_reimport: !!ingestionDecision.forceReimport,
                skip_import: false,
            });
            return pollLogs('script', 'script-logs');
        }

        document.getElementById('btn-gen-script').addEventListener('click', async () => {
            await lockGenerationMode('generate_annotated_script');
            const statusEl = document.getElementById('upload-status');
            try {
                await startGenerateScript();
            } catch (e) {
                updateScriptTaskButtons({ running: false });
                const detail = e.message || 'Unknown error';
                if (detail.includes('No input file')) {
                    statusEl.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle me-1"></i>No file loaded. Please select a text file first.</span>';
                } else {
                    statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${detail}</span>`;
                }
            }
        });

        async function startReviewScript() {
            await API.post('/api/review_script', {});
            return pollLogs('review', 'script-logs');
        }

        document.getElementById('btn-review-script').addEventListener('click', async () => {
            try {
                await startReviewScript();
            } catch (e) {
                showToast("Failed to start review: " + e.message, 'error');
            }
        });

        async function startScriptSanity() {
            await API.post('/api/script_sanity_check', {});
            return pollLogs('sanity', 'script-logs');
        }

        document.getElementById('btn-script-sanity').addEventListener('click', async () => {
            try {
                await startScriptSanity();
            } catch (e) {
                showToast("Failed to run sanity check: " + e.message, 'error');
            }
        });

        async function startReplaceMissingChunks() {
            await API.post('/api/replace_missing_chunks', {});
            return pollLogs('repair', 'script-logs');
        }

        let processingWorkflowPollTimer = null;
        const workflowStageTaskNames = {
            script: 'script',
            review: 'review',
            sanity: 'sanity',
            repair: 'repair',
            voices: 'voices',
            audio: 'audio',
        };

        async function resolveScriptIngestionDecision({ continueWorkflow }) {
            const preflight = await API.get('/api/script_ingestion/preflight');
            if (!preflight?.warn) {
                return { forceReimport: false, skipImport: false, skipScriptStage: false };
            }

            const confirmed = await showConfirm(
                `${preflight.message}\n\nDo you want to completely delete the existing project and re-import the document?`
            );
            if (confirmed) {
                return { forceReimport: true, skipImport: false, skipScriptStage: false };
            }

            return {
                forceReimport: false,
                skipImport: !continueWorkflow,
                skipScriptStage: !!continueWorkflow,
            };
        }

        function renderProcessingWorkflowStatus(status) {
            const el = document.getElementById('processing-workflow-status');
            const processBtn = document.getElementById('btn-process-script');
            const pauseBtn = document.getElementById('btn-pause-processing');
            if (!el || !processBtn || !pauseBtn || !status) return;

            const stageLabelMap = {
                script: 'Generating script',
                review: 'Reviewing script',
                sanity: 'Running sanity check',
                repair: 'Replacing missing chunks',
                voices: 'Processing voices',
                audio: 'Generating audio',
            };
            const stageLabel = status.current_stage ? (stageLabelMap[status.current_stage] || status.current_stage) : 'Idle';
            const options = status.options || {};
            const completedStages = Array.isArray(status.completed_stages) ? status.completed_stages : [];
            const optionSummary = [
                options.process_voices ? 'voices on' : 'voices off',
                options.generate_audio ? 'audio on' : 'audio off',
            ].join(', ');

            if (status.running) {
                processBtn.disabled = true;
                processBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing...';
                pauseBtn.disabled = false;
                processBtn.style.display = 'none';
                pauseBtn.style.display = '';
                el.textContent = `Running: ${stageLabel}. Completed: ${completedStages.join(', ') || 'none'}. ${optionSummary}.`;
            } else if (status.paused) {
                processBtn.disabled = false;
                processBtn.innerHTML = '<i class="fas fa-play-circle me-1"></i>Resume Process';
                pauseBtn.disabled = true;
                processBtn.style.display = '';
                pauseBtn.style.display = 'none';
                el.textContent = `Paused during ${stageLabel}. Completed: ${completedStages.join(', ') || 'none'}. ${optionSummary}.`;
            } else {
                processBtn.disabled = false;
                processBtn.innerHTML = '<i class="fas fa-play-circle me-1"></i>Process Script';
                pauseBtn.disabled = true;
                processBtn.style.display = '';
                pauseBtn.style.display = 'none';
                if (status.last_error) {
                    el.textContent = status.last_error;
                } else if (completedStages.length > 0) {
                    el.textContent = `Last run completed. Completed stages: ${completedStages.join(', ')}.`;
                } else {
                    el.textContent = '';
                }
            }
        }

        async function refreshProcessingWorkflowStatus() {
            const status = await API.get('/api/status/processing_workflow');
            renderProcessingWorkflowStatus(status);
            let newModeStatus = null;
            try {
                newModeStatus = await API.get('/api/status/new_mode_workflow');
            } catch (e) {
                console.warn('Could not fetch new_mode_workflow status while refreshing legacy workflow', e);
            }
            const newModeActive = !!newModeStatus?.running || !!newModeStatus?.paused;
            const legacyModeActive = !!document.getElementById('legacy-mode-toggle')?.checked;
            const canWriteLegacyScriptLogs = legacyModeActive && !newModeActive;
            const activeTaskName = workflowStageTaskNames[status.current_stage] || null;
            if (canWriteLegacyScriptLogs && status.running && activeTaskName) {
                try {
                    const activeTaskStatus = await API.get(`/api/status/${activeTaskName}`);
                    renderTaskLogs(activeTaskName, activeTaskStatus, 'script-logs');
                } catch (e) {
                    console.error(`Failed to load active workflow task status for ${activeTaskName}`, e);
                    if (Array.isArray(status.logs) && status.logs.length) {
                        document.getElementById('script-logs').innerText = status.logs.join('\n');
                    }
                }
            } else if (canWriteLegacyScriptLogs && Array.isArray(status.logs) && status.logs.length) {
                document.getElementById('script-logs').innerText = status.logs.join('\n');
            }
            const isActive = !!status.running || !!status.paused;
            if (status.running && window.setNavTaskSpinner) {
                window.setNavTaskSpinner('script');
            } else if (!status.running && window.releaseNavTaskSpinner) {
                window.releaseNavTaskSpinner('script');
            }
            if (isActive && !processingWorkflowPollTimer) {
                processingWorkflowPollTimer = setInterval(async () => {
                    try {
                        const [current, currentNewModeStatus] = await Promise.all([
                            API.get('/api/status/processing_workflow'),
                            API.get('/api/status/new_mode_workflow').catch(() => null),
                        ]);
                        const currentNewModeActive = !!currentNewModeStatus?.running || !!currentNewModeStatus?.paused;
                        const currentLegacyModeActive = !!document.getElementById('legacy-mode-toggle')?.checked;
                        const canWriteCurrentLegacyScriptLogs = currentLegacyModeActive && !currentNewModeActive;
                        renderProcessingWorkflowStatus(current);
                        if (current.running && window.setNavTaskSpinner) {
                            window.setNavTaskSpinner('script');
                        } else if (!current.running && window.releaseNavTaskSpinner) {
                            window.releaseNavTaskSpinner('script');
                        }
                        const currentTaskName = workflowStageTaskNames[current.current_stage] || null;
                        if (canWriteCurrentLegacyScriptLogs && current.running && currentTaskName) {
                            try {
                                const currentTaskStatus = await API.get(`/api/status/${currentTaskName}`);
                                renderTaskLogs(currentTaskName, currentTaskStatus, 'script-logs');
                            } catch (e) {
                                console.error(`Failed to poll active workflow task status for ${currentTaskName}`, e);
                                if (Array.isArray(current.logs) && current.logs.length) {
                                    document.getElementById('script-logs').innerText = current.logs.join('\n');
                                }
                            }
                        } else if (canWriteCurrentLegacyScriptLogs && Array.isArray(current.logs) && current.logs.length) {
                            document.getElementById('script-logs').innerText = current.logs.join('\n');
                        }
                        if (!current.running && !current.paused && processingWorkflowPollTimer) {
                            clearInterval(processingWorkflowPollTimer);
                            processingWorkflowPollTimer = null;
                        }
                    } catch (e) {
                        console.error('Processing workflow poll failed', e);
                        if (processingWorkflowPollTimer) {
                            clearInterval(processingWorkflowPollTimer);
                            processingWorkflowPollTimer = null;
                        }
                    }
                }, 1000);
            } else if (!isActive && processingWorkflowPollTimer) {
                clearInterval(processingWorkflowPollTimer);
                processingWorkflowPollTimer = null;
            }
            return status;
        }

        document.getElementById('btn-replace-missing-chunks').addEventListener('click', async () => {
            try {
                await startReplaceMissingChunks();
            } catch (e) {
                showToast("Failed to replace missing chunks: " + e.message, 'error');
            }
        });

        document.getElementById('btn-process-script').addEventListener('click', async () => {
            await lockGenerationMode('process_script_legacy');
            try {
                if (typeof window.flushSetupConfig === 'function') {
                    await window.flushSetupConfig();
                }
                const ingestionDecision = await resolveScriptIngestionDecision({ continueWorkflow: true });
                await API.post('/api/processing/start', {
                    process_voices: document.getElementById('process-voices-toggle').checked,
                    generate_audio: document.getElementById('generate-audio-toggle').checked,
                    force_reimport: !!ingestionDecision.forceReimport,
                    skip_script_stage: !!ingestionDecision.skipScriptStage,
                });
                await refreshProcessingWorkflowStatus();
                if (ingestionDecision.skipScriptStage) {
                    showToast('Script ingestion was skipped. Continuing with the remaining workflow stages.', 'info', 5000);
                } else if (ingestionDecision.forceReimport) {
                    showToast('Existing project cleared. Re-importing the source document.', 'warning', 5000);
                } else {
                    showToast('Processing workflow started.', 'success');
                }
            } catch (e) {
                showToast('Process Script failed: ' + e.message, 'error');
            }
        });

        document.getElementById('btn-pause-processing').addEventListener('click', async () => {
            try {
                const result = await API.post('/api/processing/pause', {});
                await refreshProcessingWorkflowStatus();
                if (result.status === 'pause_requested') {
                    showToast('Pause requested. Waiting for the current stage to stop safely.', 'warning');
                }
            } catch (e) {
                showToast('Pause request failed: ' + e.message, 'error');
            }
        });
