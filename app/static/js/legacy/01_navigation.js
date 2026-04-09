        // --- Navigation ---

        const SCRIPT_GATED_TABS = ['voices', 'editor', 'proofread', 'audio'];

        window.updatePipelineTabLocks = function(isLegacy, scriptReady) {
            SCRIPT_GATED_TABS.forEach(tab => {
                const link = document.querySelector(`.nav-link[data-tab="${tab}"]`);
                if (!link) return;
                if (isLegacy || scriptReady) {
                    link.classList.remove('nav-locked');
                } else {
                    link.classList.add('nav-locked');
                }
            });
        };

        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', async (e) => {
                const target = e.currentTarget;
                if (target.classList.contains('nav-locked')) return;
                const currentTab = document.querySelector('.nav-link.active')?.dataset.tab || null;
                const nextTab = target.dataset.tab;

                if (currentTab === 'editor' && nextTab !== 'editor') {
                    await flushPendingEditorChunkSaves().catch(err => {
                        console.error('Failed to flush editor saves:', err);
                    });
                }
                if (currentTab === 'audio' && nextTab !== 'audio' && window.persistExportConfigFromUI) {
                    await window.persistExportConfigFromUI().catch(err => {
                        console.error('Failed to flush export settings before tab switch:', err);
                    });
                }

                // Remove active class from all links
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                // Add active to clicked
                target.classList.add('active');

                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                // Show target tab
                const targetId = nextTab + '-tab';
                document.getElementById(targetId).style.display = 'block';

                // Trigger tab specific loads
                if (nextTab === 'editor') {
                    syncEditorChunksOnNavigation()
                        .then((result) => {
                            const needsFullRefresh = !Array.isArray(cachedChunks) || cachedChunks.length === 0 || !!result?.synced;
                            loadChunks(needsFullRefresh);
                        })
                        .catch(err => console.error('Editor sync error', err));
                    refreshAudioQueueUI().catch(err => console.error('Audio queue refresh error', err));
                    ensureAudioQueuePolling();
                    syncNarratorSelectionsFromBackend().catch(() => {});
                } else if (nextTab === 'voices') {
                    loadVoices();
                } else if (nextTab === 'dictionary') {
                    loadDictionary();
                } else if (nextTab === 'saved-scripts') {
                    loadSavedScripts();
                } else if (nextTab === 'designer') {
                    loadDesignedVoices();
                } else if (nextTab === 'training') {
                    loadLoraDatasets();
                    loadLoraModels();
                } else if (nextTab === 'dataset-builder') {
                    dsbLoadProjects(dsbCurrentProject);
                } else if (nextTab === 'audio') {
                    if (window.populateExportChapterSelect) window.populateExportChapterSelect();
                }

                reconnectTaskLogs().catch(err => console.error('Task log reconnect error', err));
            });
        });
