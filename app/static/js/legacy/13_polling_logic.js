        // --- Polling Logic ---
        const taskLogTargets = {
            script: 'script-logs',
            review: 'script-logs',
            sanity: 'script-logs',
            repair: 'script-logs',
            process_paragraphs: 'script-logs',
            assign_dialogue: 'script-logs',
            extract_temperament: 'script-logs',
            create_script: 'script-logs',
            new_mode_workflow: 'script-logs',
            proofread: 'proofread-logs',
            audio: 'audio-logs',
            lora_training: 'lora-train-logs',
        };
        const taskNavTabs = {
            script: 'script',
            review: 'script',
            sanity: 'script',
            repair: 'script',
            process_paragraphs: 'script',
            assign_dialogue: 'script',
            extract_temperament: 'script',
            create_script: 'script',
            new_mode_workflow: 'script',
            voices: 'voices',
            proofread: 'proofread',
            audio: 'audio',
        };
        const activeTaskPolls = new Map();
        const handledTaskCompletions = new Set();
        const taskCompletionStorage = (() => {
            try {
                const probeKey = '__threadspeak_task_completion_probe__';
                sessionStorage.setItem(probeKey, '1');
                sessionStorage.removeItem(probeKey);
                return sessionStorage;
            } catch (_) {
                return null;
            }
        })();

        function resolveTaskNavTab(taskName, status, fallbackNavTab) {
            if (taskName !== 'audio') return fallbackNavTab;
            const hasEditorAudioWork = Boolean(status?.current_job) || (status?.queue || []).length > 0;
            if (hasEditorAudioWork) return 'editor';
            if (status?.merge_running || status?.merge_progress?.running) return 'audio';
            return fallbackNavTab;
        }

        function buildTaskCompletionKey(kind, taskName, action) {
            return `threadspeak:${kind}:${taskName}:${action}`;
        }

        function markTaskActionRequested(taskName, action) {
            if (!taskCompletionStorage) return;
            taskCompletionStorage.setItem(buildTaskCompletionKey('requested', taskName, action), '1');
        }

        function clearTaskActionRequested(taskName, action) {
            if (!taskCompletionStorage) return;
            taskCompletionStorage.removeItem(buildTaskCompletionKey('requested', taskName, action));
        }

        function wasTaskActionRequested(taskName, action) {
            if (!taskCompletionStorage) return false;
            return taskCompletionStorage.getItem(buildTaskCompletionKey('requested', taskName, action)) === '1';
        }

        function hasHandledTaskCompletion(taskName, marker) {
            if (!taskCompletionStorage) return false;
            return taskCompletionStorage.getItem(buildTaskCompletionKey('handled', taskName, marker)) === '1';
        }

        function markHandledTaskCompletion(taskName, marker) {
            if (!taskCompletionStorage) return;
            taskCompletionStorage.setItem(buildTaskCompletionKey('handled', taskName, marker), '1');
        }

        function shouldHandleTaskCompletion(taskName, marker) {
            if (!marker) return false;
            const key = `${taskName}:${marker}`;
            if (handledTaskCompletions.has(key) || hasHandledTaskCompletion(taskName, marker)) {
                return false;
            }
            handledTaskCompletions.add(key);
            markHandledTaskCompletion(taskName, marker);
            return true;
        }

        function updateScriptTaskButtons(status) {
            const running = !!status?.running;
            const generateBtn = document.getElementById('btn-gen-script');
            const reviewBtn = document.getElementById('btn-review-script');
            const sanityBtn = document.getElementById('btn-script-sanity');
            const replaceBtn = document.getElementById('btn-replace-missing-chunks');
            if (generateBtn) {
                generateBtn.disabled = running;
                generateBtn.innerHTML = running
                    ? '<i class="fas fa-spinner fa-spin me-2"></i>Generating Script...'
                    : '<i class="fas fa-magic me-2"></i>Generate Annotated Script';
            }
            if (reviewBtn) reviewBtn.disabled = running;
            if (sanityBtn) sanityBtn.disabled = running;
            if (replaceBtn) replaceBtn.disabled = running;
        }

        function renderTaskLogs(taskName, status, elementId = taskLogTargets[taskName]) {
            const el = document.getElementById(elementId);
            if (!el || !status) return;
            const logs = Array.isArray(status.logs) ? status.logs : [];
            if (taskName === 'audio') {
                renderAudioMergeProgress(status);
            } else if (taskName === 'proofread') {
                renderProofreadTaskStatus(status);
                const proofreadTabVisible = document.getElementById('proofread-tab')?.style.display !== 'none';
                if (status.running && proofreadTabVisible) {
                    if (window.loadProofreadData) {
                        window.loadProofreadData({ includeChapters: false }).catch(err => console.error('Failed to refresh proofread rows', err));
                    }
                }
            } else if (taskName === 'repair') {
                if (status.running) {
                    loadChunks(false).catch(err => console.error('Failed to refresh chunks during repair', err));
                }
            }
            if (taskName === 'script') {
                updateScriptTaskButtons(status);
            }
            if (taskName === 'new_mode_workflow') {
                updateNewModeWorkflowButtons(status);
            }
            if ('value' in el && el.tagName === 'TEXTAREA') {
                el.value = logs.join('\n');
                el.scrollTop = el.scrollHeight;
                return;
            }
            el.innerText = logs.join('\n');
            el.scrollTop = el.scrollHeight;
        }

        function finalizeTaskStatus(taskName, status, options = {}) {
            const activeRunTransition = Boolean(options.activeRunTransition);
            if (taskName === 'audio' && status.logs.some(l => l.includes("Optimized export complete"))) {
                const marker = `optimized:${status.merge_progress?.updated_at || status.logs[status.logs.length - 1] || ''}`;
                if (wasTaskActionRequested(taskName, 'optimized') && shouldHandleTaskCompletion(taskName, marker)) {
                    const a = document.createElement('a');
                    a.href = `/api/optimized_export?t=${new Date().getTime()}`;
                    a.download = 'optimized_audiobook.zip';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
                clearTaskActionRequested(taskName, 'optimized');
            } else if (taskName === 'audio' && status.logs.some(l => l.includes("Merge complete"))) {
                const marker = `merge:${status.merge_progress?.updated_at || status.logs[status.logs.length - 1] || ''}`;
                if (wasTaskActionRequested(taskName, 'merge') && shouldHandleTaskCompletion(taskName, marker)) {
                    const audio = document.getElementById('main-audio');
                    audio.src = `/api/audiobook?t=${new Date().getTime()}`;
                    document.getElementById('audio-player-container').style.display = 'block';
                    document.getElementById('download-link').href = audio.src;
                }
                clearTaskActionRequested(taskName, 'merge');
            }
            if ((taskName === 'script' || taskName === 'review') && status.logs.some(l => l.includes("completed successfully"))) {
                const tbody = document.getElementById('chunks-table-body');
                if (tbody) tbody.innerHTML = '';
                if (document.getElementById('editor-tab').style.display !== 'none') {
                    loadChunks();
                }
            } else if (taskName === 'proofread') {
                if (activeRunTransition && window.loadProofreadData) {
                    window.loadProofreadData({ includeChapters: false }).catch(err => console.error('Failed to refresh proofread after completion', err));
                }
            } else if (taskName === 'repair') {
                loadChunks(false).catch(err => console.error('Failed to refresh chunks after repair', err));
            } else if (taskName === 'create_script') {
                loadChunks(true).catch(err => console.error('Failed to refresh chunks after script creation', err));
            }
            // Refresh step icons and button states after any pipeline step finishes
            if (_stepIconMap && _stepIconMap[taskName]) {
                loadPipelineStepIcons().catch(() => {});
            }
            if (taskName === 'new_mode_workflow') {
                updateNewModeWorkflowButtons(status);
                if (!status.paused && !status.last_error && status.completed_at) {
                    loadChunks(true).catch(err => console.error('Failed to refresh chunks after workflow', err));
                }
            }
        }

        async function pollLogs(taskName, elementId, navTab) {
            const resolvedNavTab = navTab === false ? null : (navTab || taskNavTabs[taskName] || null);
            if (resolvedNavTab && window.setNavTaskSpinner) {
                window.setNavTaskSpinner(resolvedNavTab);
            }

            const existing = activeTaskPolls.get(taskName);
            if (existing && existing.elementId === elementId) {
                return existing.promise;
            }

            const promise = new Promise((resolve, reject) => {
                let interval = null;
                const pollOnce = async () => {
                    try {
                        const status = await API.get(`/api/status/${taskName}`);
                        renderTaskLogs(taskName, status, elementId);

                        if (!status.running) {
                            if (interval) clearInterval(interval);
                            activeTaskPolls.delete(taskName);
                            finalizeTaskStatus(taskName, status, { activeRunTransition: true });
                            if (resolvedNavTab && window.releaseNavTaskSpinner) {
                                window.releaseNavTaskSpinner(resolvedNavTab);
                            }
                            resolve(status);
                        }
                    } catch (e) {
                        console.error("Poll error", e);
                        if (interval) clearInterval(interval);
                        activeTaskPolls.delete(taskName);
                        if (resolvedNavTab && window.releaseNavTaskSpinner) {
                            window.releaseNavTaskSpinner(resolvedNavTab);
                        }
                        reject(e);
                    }
                };

                interval = setInterval(pollOnce, 1000);
                pollOnce();
            });

            activeTaskPolls.set(taskName, {
                elementId,
                navTab: resolvedNavTab,
                promise,
            });

            return promise;
        }

        async function reconnectTaskLogs() {
            let newModeWorkflowActive = false;
            try {
                const workflowStatus = await API.get('/api/status/new_mode_workflow');
                newModeWorkflowActive = !!workflowStatus?.running || !!workflowStatus?.paused;
            } catch (e) {
                console.error('Failed to prefetch new_mode_workflow status', e);
            }

            for (const [taskName, elementId] of Object.entries(taskLogTargets)) {
                const el = document.getElementById(elementId);
                if (!el) continue;
                const legacyModeActive = !!document.getElementById('legacy-mode-toggle')?.checked;
                const isNewModeStageTask = (
                    taskName === 'process_paragraphs'
                    || taskName === 'assign_dialogue'
                    || taskName === 'extract_temperament'
                    || taskName === 'create_script'
                );
                const isLegacyScriptPaneTask = (
                    taskName === 'script'
                    || taskName === 'review'
                    || taskName === 'sanity'
                    || taskName === 'repair'
                );
                if (newModeWorkflowActive && (isNewModeStageTask || isLegacyScriptPaneTask)) {
                    // Avoid multiple pollers writing to the same script log pane
                    // while the new-mode workflow is orchestrating these stages.
                    continue;
                }
                if (!legacyModeActive && isLegacyScriptPaneTask) {
                    // Legacy script-stage logs should only be restored when legacy mode is active.
                    continue;
                }
                try {
                    const status = await API.get(`/api/status/${taskName}`);
                    renderTaskLogs(taskName, status, elementId);
                    if (status.running) {
                        const hasNavTask = Boolean(window.getNavTaskSpinnerTab && window.getNavTaskSpinnerTab());
                        const reconnectNavTab = hasNavTask ? false : resolveTaskNavTab(taskName, status, undefined);
                        pollLogs(taskName, elementId, reconnectNavTab).catch(err => {
                            console.error(`Polling failed for ${taskName}`, err);
                        });
                    } else {
                        finalizeTaskStatus(taskName, status, { activeRunTransition: false });
                    }
                } catch (e) {
                    console.error(`Reconnect failed for ${taskName}`, e);
                }
            }
        }

        window.addEventListener('pageshow', () => {
            reconnectTaskLogs().catch(err => console.error('pageshow reconnect failed', err));
            refreshProcessingWorkflowStatus().catch(err => console.error('pageshow processing reconnect failed', err));
            loadPipelineStepIcons().catch(err => console.error('pageshow icon load failed', err));
        });
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                reconnectTaskLogs().catch(err => console.error('visibility reconnect failed', err));
                refreshProcessingWorkflowStatus().catch(err => console.error('visibility processing reconnect failed', err));
                loadPipelineStepIcons().catch(err => console.error('visibility icon load failed', err));
            }
        });

        // ── Projects ──────────────────────────────────────

        async function loadSavedScripts() {
            try {
                const res = await fetch('/api/scripts');
                const scripts = await res.json();
                const container = document.getElementById('saved-scripts-list');

                if (!scripts.length) {
                    container.innerHTML = '<p class="text-muted mb-0">No saved projects yet.</p>';
                    return;
                }

                container.innerHTML = scripts.map(s => {
                    const date = new Date(s.created * 1000).toLocaleDateString('en-US', {
                        month: 'short', day: 'numeric', year: 'numeric'
                    });
                    const badge = s.kind === 'script'
                        ? '<span class="badge bg-secondary me-2" title="Legacy script snapshot">Legacy</span>'
                        : s.has_audio
                        ? '<span class="badge bg-warning text-dark me-2" title="Includes generated audio">Audio</span>'
                        : s.has_voice_config
                        ? '<span class="badge bg-info me-2" title="Includes voice configuration">Voices</span>'
                        : '<span class="badge bg-primary me-2" title="Full saved project">Project</span>';
                    return `
                        <div class="d-flex align-items-center py-2 border-bottom">
                            <div class="d-flex align-items-center flex-grow-1 overflow-hidden">
                                ${badge}<strong class="text-truncate">${s.name}</strong>
                            </div>
                            <div class="d-flex align-items-center gap-2 ms-3 flex-shrink-0">
                                <small class="text-muted">${date}</small>
                                <button class="btn btn-sm btn-outline-success" onclick="loadScript('${s.name}')"><i class="fas fa-upload me-1"></i>Load</button>
                                <button class="btn btn-sm btn-outline-danger" onclick="deleteScript('${s.name}')"><i class="fas fa-trash"></i></button>
                            </div>
                        </div>`;
                }).join('');
            } catch (e) {
                console.error('Failed to load saved scripts:', e);
            }
        }

        async function saveScript() {
            const nameInput = document.getElementById('save-script-name');
            const name = nameInput.value.trim();
            try {
                const res = await fetch('/api/scripts/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name})
                });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to save project.', 'error');
                    return;
                }
                nameInput.value = '';
                loadSavedScripts();
            } catch (e) {
                showToast('Error saving project: ' + e.message, 'error');
            }
        }

        window.downloadProjectArchive = async () => {
            try {
                const response = await fetch('/api/project_archive');
                if (!response.ok) {
                    const raw = await response.text();
                    let detail = raw;
                    try {
                        const parsed = JSON.parse(raw);
                        detail = parsed.detail || raw;
                    } catch (_) {}
                    showToast(detail || 'Failed to save project archive.', 'error');
                    return;
                }
                const blob = await response.blob();
                const disposition = response.headers.get('Content-Disposition') || '';
                const match = disposition.match(/filename="?([^"]+)"?/i);
                const filename = match ? match[1] : 'threadspeak_project.zip';
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            } catch (e) {
                showToast('Error saving project archive: ' + e.message, 'error');
            }
        };

        window.triggerProjectArchiveLoad = () => {
            const input = document.getElementById('project-archive-input');
            if (!input) return;
            input.value = '';
            input.click();
        };

        window.handleProjectArchiveLoad = async (input) => {
            const file = input?.files?.[0];
            if (!file) return;
            if (!file.name.toLowerCase().endsWith('.zip')) {
                showToast('Project archive must be a .zip file.', 'warning');
                input.value = '';
                return;
            }
            if (!await showConfirm(`Load project archive "${file.name}"? This will replace the current project state and audio segments.`)) {
                input.value = '';
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/project_archive/load', {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    const raw = await response.text();
                    let detail = raw;
                    try {
                        const parsed = JSON.parse(raw);
                        detail = parsed.detail || raw;
                    } catch (_) {}
                    showToast(detail || 'Failed to load project archive.', 'error');
                    return;
                }
                await response.json();
                await loadConfig();
                await loadSavedScripts();
                await loadVoices();
                await loadChunks(true);
                await refreshProcessingWorkflowStatus();
                await refreshAudioQueueUI().catch(() => null);
                showToast('Project archive loaded.', 'success');
            } catch (e) {
                showToast('Error loading project archive: ' + e.message, 'error');
            } finally {
                input.value = '';
            }
        };

        async function loadScript(name) {
            if (!await showConfirm(`Load project "${name}"? This will replace your current project state.`)) return;
            try {
                const res = await fetch('/api/scripts/load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name})
                });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to load project.', 'error');
                    return;
                }
                await loadConfig();
                loadSavedScripts();
                loadChunks(true);
                loadVoices();
                refreshProcessingWorkflowStatus();
                refreshAudioQueueUI().catch(() => null);
            } catch (e) {
                showToast('Error loading project: ' + e.message, 'error');
            }
        }

        async function deleteScript(name) {
            if (!await showConfirm(`Delete saved project "${name}"? This cannot be undone.`)) return;
            try {
                const res = await fetch(`/api/scripts/${encodeURIComponent(name)}`, {method: 'DELETE'});
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to delete project.', 'error');
                    return;
                }
                loadSavedScripts();
            } catch (e) {
                showToast('Error deleting project: ' + e.message, 'error');
            }
        }
