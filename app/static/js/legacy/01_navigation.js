        // --- Navigation ---
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', async (e) => {
                const target = e.currentTarget;
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
                        .catch(err => console.error('Editor sync error', err))
                        .finally(() => {
                            loadChunks();
                        });
                    refreshAudioQueueUI().catch(err => console.error('Audio queue refresh error', err));
                    ensureAudioQueuePolling();
                } else if (nextTab === 'voices') {
                    loadVoices();
                } else if (nextTab === 'dictionary') {
                    loadDictionary();
                } else if (nextTab === 'designer') {
                    loadDesignedVoices();
                } else if (nextTab === 'training') {
                    loadLoraDatasets();
                    loadLoraModels();
                } else if (nextTab === 'dataset-builder') {
                    dsbLoadProjects(dsbCurrentProject);
                }

                reconnectTaskLogs().catch(err => console.error('Task log reconnect error', err));
            });
        });
