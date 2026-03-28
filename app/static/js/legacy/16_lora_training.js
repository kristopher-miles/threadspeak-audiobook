        // --- LoRA Training ---
        window._loraModelsCache = [];

        async function loadLoraDatasets() {
            try {
                const datasets = await API.get('/api/lora/datasets');
                const listEl = document.getElementById('lora-datasets-list');
                const selectEl = document.getElementById('lora-dataset-select');

                // Update dropdown
                const currentVal = selectEl.value;
                selectEl.innerHTML = '<option value="">-- Select dataset --</option>' +
                    datasets.map(d => `<option value="${d.dataset_id}">${d.dataset_id} (${d.sample_count} samples)</option>`).join('');
                if (currentVal) selectEl.value = currentVal;

                // Update list
                if (!datasets.length) {
                    listEl.innerHTML = '<span class="text-muted">No datasets uploaded yet.</span>';
                    return;
                }
                listEl.innerHTML = datasets.map(d => `
                    <div class="d-flex justify-content-between align-items-center py-1">
                        <span><strong>${d.dataset_id}</strong> <small class="text-muted">(${d.sample_count} samples)</small></span>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteLoraDataset('${d.dataset_id}')"><i class="fas fa-trash"></i></button>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load LoRA datasets:', e);
            }
        }

        window.uploadLoraDataset = async () => {
            const fileInput = document.getElementById('lora-dataset-file');
            if (!fileInput.files.length) { showToast('Select a ZIP file first.', 'warning'); return; }

            const file = fileInput.files[0];
            if (!file.name.endsWith('.zip')) { showToast('File must be a .zip archive.', 'warning'); return; }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/lora/upload_dataset', { method: 'POST', body: formData });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Upload failed.', 'error');
                    return;
                }
                const result = await res.json();
                showToast(`Dataset "${result.dataset_id}" uploaded (${result.sample_count} samples).`, 'success');
                fileInput.value = '';
                loadLoraDatasets();
            } catch (e) {
                showToast('Upload error: ' + e.message, 'error');
            }
        };

        window.deleteLoraDataset = async (datasetId) => {
            if (!await showConfirm(`Delete dataset "${datasetId}"?`)) return;
            try {
                const res = await fetch(`/api/lora/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadLoraDatasets();
            } catch (e) {
                showToast('Error deleting dataset: ' + e.message, 'error');
            }
        };


        window.startLoraTraining = async () => {
            const name = document.getElementById('lora-adapter-name').value.trim();
            const datasetId = document.getElementById('lora-dataset-select').value;
            if (!name) { showToast('Enter an adapter name.', 'warning'); return; }
            if (!datasetId) { showToast('Select a dataset.', 'warning'); return; }

            const request = {
                name: name,
                dataset_id: datasetId,
                epochs: parseInt(document.getElementById('lora-epochs').value) || 5,
                lr: parseFloat(document.getElementById('lora-lr').value) || 5e-6,
                batch_size: parseInt(document.getElementById('lora-batch-size').value) || 1,
                lora_r: parseInt(document.getElementById('lora-rank').value) || 32,
                lora_alpha: parseInt(document.getElementById('lora-alpha').value) || 128,
                gradient_accumulation_steps: parseInt(document.getElementById('lora-grad-accum').value) || 8
            };

            const btn = document.getElementById('btn-lora-train');
            btn.disabled = true;
            document.getElementById('lora-train-status').innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Starting...';

            try {
                const result = await API.post('/api/lora/train', request);
                document.getElementById('lora-progress-section').style.display = 'block';
                document.getElementById('lora-train-status').innerHTML = '<span class="text-info">Training in progress...</span>';
                pollLoraTraining(request.epochs);
            } catch (e) {
                showToast('Failed to start training: ' + e.message, 'error');
                btn.disabled = false;
                document.getElementById('lora-train-status').innerHTML = '';
            }
        };

        function pollLoraTraining(totalEpochs) {
            const logsEl = document.getElementById('lora-train-logs');
            const progressBar = document.getElementById('lora-progress-bar');
            const epochDisplay = document.getElementById('lora-epoch-display');
            const lossDisplay = document.getElementById('lora-loss-display');

            const interval = setInterval(async () => {
                try {
                    const status = await API.get('/api/status/lora_training');
                    logsEl.innerText = status.logs.join('\n');
                    logsEl.scrollTop = logsEl.scrollHeight;

                    // Parse latest metrics from log lines
                    for (let i = status.logs.length - 1; i >= 0; i--) {
                        const line = status.logs[i];
                        const epochMatch = line.match(/\[EPOCH\]\s*(\d+)\/(\d+)\s+avg_loss=([\d.]+)/);
                        if (epochMatch) {
                            const epoch = parseInt(epochMatch[1]);
                            const maxEpoch = parseInt(epochMatch[2]);
                            const loss = epochMatch[3];
                            const pct = Math.round((epoch / maxEpoch) * 100);
                            epochDisplay.innerText = `${epoch}/${maxEpoch}`;
                            lossDisplay.innerText = loss;
                            progressBar.style.width = `${pct}%`;
                            progressBar.innerText = `${pct}%`;
                            break;
                        }
                        const trainMatch = line.match(/\[TRAIN\]\s*epoch=(\d+)\/(\d+)\s+step=\d+\/\d+\s+loss=([\d.]+)/);
                        if (trainMatch) {
                            const epoch = parseInt(trainMatch[1]);
                            const maxEpoch = parseInt(trainMatch[2]);
                            const loss = trainMatch[3];
                            const pct = Math.round(((epoch - 1) / maxEpoch) * 100);
                            epochDisplay.innerText = `${epoch}/${maxEpoch}`;
                            lossDisplay.innerText = loss;
                            progressBar.style.width = `${pct}%`;
                            progressBar.innerText = `${pct}%`;
                            break;
                        }
                    }

                    if (!status.running) {
                        clearInterval(interval);
                        const btn = document.getElementById('btn-lora-train');
                        btn.disabled = false;

                        const isDone = status.logs.some(l => l.includes('[DONE]'));
                        const isError = status.logs.some(l => l.includes('[ERROR]'));

                        if (isDone) {
                            document.getElementById('lora-train-status').innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Training complete!</span>';
                            progressBar.style.width = '100%';
                            progressBar.innerText = '100%';
                            progressBar.classList.remove('progress-bar-animated');
                            progressBar.classList.replace('bg-info', 'bg-success');
                            loadLoraModels();
                        } else if (isError) {
                            document.getElementById('lora-train-status').innerHTML = '<span class="text-danger"><i class="fas fa-times me-1"></i>Training failed</span>';
                            progressBar.classList.remove('progress-bar-animated');
                            progressBar.classList.replace('bg-info', 'bg-danger');
                        } else {
                            document.getElementById('lora-train-status').innerHTML = '<span class="text-warning">Training stopped</span>';
                        }
                    }
                } catch (e) {
                    console.error('LoRA poll error:', e);
                    clearInterval(interval);
                }
            }, 2000);
        }

        async function loadLoraModels() {
            try {
                const models = await API.get('/api/lora/models');
                window._loraModelsCache = models;
                const container = document.getElementById('lora-models-list');
                const testForm = document.getElementById('lora-test-form');

                if (!models.length) {
                    container.innerHTML = '<p class="text-muted mb-0">No adapters available.</p>';
                    testForm.style.display = 'none';
                    return;
                }

                container.innerHTML = `
                    <table class="table table-sm table-hover mb-0">
                        <thead><tr><th>Name</th><th>Dataset</th><th>Epochs</th><th>Final Loss</th><th>Samples</th><th style="width:240px">Actions</th></tr></thead>
                        <tbody>
                            ${models.map(m => `
                                <tr${m.builtin ? ' class="table-light"' : ''}>
                                    <td><strong>${m.name}</strong>${m.builtin ? ` <span class="badge bg-secondary">built-in</span>${m.downloaded === false ? ' <span class="badge bg-warning text-dark">not downloaded</span>' : ''}` : ''}</td>
                                    <td>${m.dataset_id || (m.builtin ? '--' : '--')}</td>
                                    <td>${m.epochs || '--'}</td>
                                    <td>${m.final_loss != null ? m.final_loss.toFixed(4) : '--'}</td>
                                    <td>${m.sample_count || '--'}</td>
                                    <td>
                                        ${m.builtin && m.downloaded === false ? `
                                            <button class="btn btn-sm btn-outline-warning" id="lora-dl-btn-${m.id}" onclick="downloadBuiltinAdapter('${m.id}')" title="Download from HuggingFace"><i class="fas fa-download me-1"></i>Download</button>
                                        ` : `
                                            <button class="btn btn-sm ${m.preview_audio_url ? 'btn-outline-success' : 'btn-outline-secondary'} me-1" id="lora-preview-btn-${m.id}" onclick="playLoraPreview('${m.id}')" title="${m.preview_audio_url ? 'Play preview' : 'Generate and play preview (first time may take a moment)'}"><i class="fas fa-volume-up"></i></button>
                                            <button class="btn btn-sm btn-outline-primary me-1" onclick="testLoraModel('${m.id}')" title="Generate test line with custom text"><i class="fas fa-flask me-1"></i>Test</button>
                                            ${m.builtin ? '' : `<button class="btn btn-sm btn-outline-danger" onclick="deleteLoraModel('${m.id}')" title="Delete"><i class="fas fa-trash"></i></button>`}
                                        `}
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>`;

                // Populate test dropdown
                const dropdown = document.getElementById('lora-test-adapter');
                const prevVal = dropdown.value;
                dropdown.innerHTML = models.filter(m => m.downloaded !== false).map(m =>
                    `<option value="${m.id}">${m.name}</option>`
                ).join('');
                if (prevVal && models.some(m => m.id === prevVal)) dropdown.value = prevVal;
                testForm.style.display = '';
            } catch (e) {
                console.error('Failed to load LoRA models:', e);
            }
        }

        window.playLoraPreview = async (adapterId) => {
            const btn = document.getElementById(`lora-preview-btn-${adapterId}`);
            const origHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                const result = await API.post(`/api/lora/preview/${encodeURIComponent(adapterId)}`, {});
                await playSharedPreviewAudio(`${result.audio_url}?t=${Date.now()}`);
                // Update button now that preview is cached
                btn.title = 'Play preview';
                btn.classList.replace('btn-outline-secondary', 'btn-outline-success');
            } catch (e) {
                showToast('Preview failed: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = origHtml;
            }
        };

        window.testLoraModel = (adapterId) => {
            document.getElementById('lora-test-adapter').value = adapterId;
            document.getElementById('lora-test-form').style.display = '';
            document.getElementById('lora-test-text').focus();
        };

        window.runLoraTest = async () => {
            const adapterId = document.getElementById('lora-test-adapter').value;
            const text = document.getElementById('lora-test-text').value.trim();
            const instruct = document.getElementById('lora-test-instruct').value.trim();
            if (!adapterId) { showToast('Select an adapter.', 'warning'); return; }
            if (!text) { showToast('Enter text to synthesize.', 'warning'); return; }

            const statusEl = document.getElementById('lora-test-status');
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating...';

            try {
                const result = await API.post('/api/lora/test', {
                    adapter_id: adapterId,
                    text: text,
                    instruct: instruct
                });

                statusEl.innerHTML = '';
                const audioDiv = document.getElementById('lora-test-audio');
                audioDiv.innerHTML = `<audio controls autoplay src="${result.audio_url}?t=${Date.now()}"></audio>`;
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger">Failed: ${e.message}</span>`;
            }
        };

        window.deleteLoraModel = async (adapterId) => {
            if (!await showConfirm('Delete this trained adapter? This cannot be undone.')) return;
            try {
                const res = await fetch(`/api/lora/models/${encodeURIComponent(adapterId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadLoraModels();
            } catch (e) {
                showToast('Error deleting adapter: ' + e.message, 'error');
            }
        };

        window.downloadBuiltinAdapter = async (adapterId) => {
            const btn = document.getElementById(`lora-dl-btn-${adapterId}`);
            const origHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Downloading...';

            try {
                await API.post(`/api/lora/download/${encodeURIComponent(adapterId)}`, {});
                showToast('Adapter downloaded successfully.', 'success');
                loadLoraModels();
            } catch (e) {
                showToast('Download failed: ' + e.message, 'error');
                btn.disabled = false;
                btn.innerHTML = origHtml;
            }
        };

        // ── Dataset Builder ──────────────────────────────────────

        // In-memory row data: [{emotion, text, status, audio_url}, ...]
        let dsbRows = [];
        let dsbPolling = null;
        let dsbBatchRunning = false;
        let dsbSaveMetaTimer = null;
        let dsbSaveRowsTimer = null;
        let dsbCurrentProject = '';

        // Clean up legacy localStorage
        localStorage.removeItem('alexandria-dsb-form');

        async function dsbLoadProjects(selectName) {
            try {
                const projects = await API.get('/api/dataset_builder/list');
                const select = document.getElementById('dsb-project-select');
                select.innerHTML = '<option value="">-- Select project --</option>' +
                    projects.map(p => `<option value="${p.name}">${p.name} (${p.done_count}/${p.sample_count})</option>`).join('');
                if (selectName) {
                    select.value = selectName;
                    dsbOnProjectChange();
                }
            } catch (e) { console.error('Failed to load projects:', e); }
        }

        window.dsbOnProjectChange = async () => {
            const name = document.getElementById('dsb-project-select').value;
            const formArea = document.getElementById('dsb-form-area');
            const deleteBtn = document.getElementById('dsb-btn-delete-project');
            if (!name) {
                dsbCurrentProject = '';
                formArea.style.display = 'none';
                deleteBtn.style.display = 'none';
                dsbRows = [];
                dsbRenderTable();
                return;
            }
            dsbCurrentProject = name;
            formArea.style.display = '';
            deleteBtn.style.display = '';
            await dsbLoadProject(name);
        };

        async function dsbLoadProject(name) {
            try {
                const result = await API.get(`/api/dataset_builder/status/${encodeURIComponent(name)}`);
                document.getElementById('dsb-description').value = result.description || '';
                document.getElementById('dsb-global-seed').value = result.global_seed || '';
                dsbRows = (result.samples || []).map(s => ({
                    emotion: s.emotion || s.description || '',
                    text: s.text || '',
                    seed: s.seed ?? '',
                    status: s.status || 'pending',
                    audio_url: s.audio_url || null,
                }));
                if (dsbRows.length === 0) dsbAddRow();
                dsbRenderTable();
                // Resume polling if batch is running
                if (result.running) {
                    dsbBatchRunning = true;
                    dsbStartPolling(name);
                    document.getElementById('dsb-btn-gen-all').style.display = 'none';
                    document.getElementById('dsb-btn-regen-all').style.display = 'none';
                    document.getElementById('dsb-btn-cancel').style.display = '';
                }
            } catch (e) {
                dsbRows = [];
                dsbAddRow();
            }
        }

        window.dsbCreateProject = async () => {
            const name = prompt('Dataset name:');
            if (!name || !name.trim()) return;
            try {
                const result = await API.post('/api/dataset_builder/create', { name: name.trim() });
                await dsbLoadProjects(result.name);
            } catch (e) {
                showToast('Failed to create project: ' + e.message, 'error');
            }
        };

        window.dsbDeleteProject = async () => {
            if (!dsbCurrentProject) return;
            if (!await showConfirm(`Delete project "${dsbCurrentProject}" and all its samples?`)) return;
            try {
                await fetch(`/api/dataset_builder/${encodeURIComponent(dsbCurrentProject)}`, { method: 'DELETE' });
                dsbCurrentProject = '';
                document.getElementById('dsb-form-area').style.display = 'none';
                document.getElementById('dsb-btn-delete-project').style.display = 'none';
                dsbRows = [];
                dsbRenderTable();
                await dsbLoadProjects();
            } catch (e) {
                showToast('Delete failed: ' + e.message, 'error');
            }
        };

        function dsbSaveForm() {
            if (!dsbCurrentProject) return;
            clearTimeout(dsbSaveMetaTimer);
            dsbSaveMetaTimer = setTimeout(async () => {
                try {
                    await API.post('/api/dataset_builder/update_meta', {
                        name: dsbCurrentProject,
                        description: document.getElementById('dsb-description').value,
                        global_seed: document.getElementById('dsb-global-seed').value,
                    });
                } catch (e) { console.error('Failed to save meta:', e); }
            }, 500);
        }

        function dsbSaveRows() {
            if (!dsbCurrentProject) return;
            clearTimeout(dsbSaveRowsTimer);
            dsbSaveRowsTimer = setTimeout(async () => {
                try {
                    await API.post('/api/dataset_builder/update_rows', {
                        name: dsbCurrentProject,
                        rows: dsbRows.map(r => ({ emotion: r.emotion || '', text: (r.text || '').trim(), seed: r.seed ?? '' })),
                    });
                } catch (e) { console.error('Failed to save rows:', e); }
            }, 500);
        }

        function dsbAddRow(emotion = '', text = '', seed = '') {
            dsbRows.push({ emotion, text, seed, status: 'pending', audio_url: null });
            dsbRenderTable();
            dsbSaveRows();
            // Focus the new emotion field
            setTimeout(() => {
                const rows = document.querySelectorAll('#dsb-table-body tr');
                const last = rows[rows.length - 1];
                if (last) last.querySelector('input')?.focus();
            }, 50);
        }

        function dsbRemoveRow(index) {
            dsbRows.splice(index, 1);
            dsbRenderTable();
            dsbSaveRows();
            dsbUpdateRefDropdown();
        }

        function dsbBuildRowHtml(row, i) {
            const statusColor = row.status === 'done' ? 'success' :
                                row.status === 'generating' ? 'warning' :
                                row.status === 'error' ? 'danger' : 'secondary';
            const statusLabel = row.status || 'pending';

            let actionHtml = '';
            if (row.status === 'generating') {
                actionHtml = '<div class="progress" style="width:80px;height:20px;"><div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" style="width:100%"></div></div>';
            } else {
                const genLabel = row.status === 'done' ? '<i class="fas fa-redo"></i>' : '<i class="fas fa-play"></i>';
                actionHtml = `<button class="btn btn-sm btn-primary" onclick="dsbGenSample(${i})" title="${row.status === 'done' ? 'Regenerate' : 'Generate'}">${genLabel}</button>`;
            }

            let audioHtml = '';
            if (row.status === 'done' && row.audio_url) {
                audioHtml = `<audio controls src="${row.audio_url}" style="width:180px;height:28px;" onplay="dsbStopOthers(${i})"></audio>`;
            }

            return `<tr data-dsb-idx="${i}" data-dsb-status="${row.status || 'pending'}" data-dsb-audio="${row.audio_url || ''}" class="${row.status === 'generating' ? 'table-info' : ''}">
                <td class="text-center align-middle">${i + 1}</td>
                <td><input type="text" class="form-control form-control-sm" value="${(row.emotion || '').replace(/"/g, '&quot;')}" onchange="dsbUpdateRow(${i}, 'emotion', this.value)" placeholder="e.g. Savagely sarcastic"></td>
                <td><textarea class="form-control form-control-sm" rows="2" onchange="dsbUpdateRow(${i}, 'text', this.value)" placeholder="Sample text...">${(row.text || '').replace(/</g, '&lt;')}</textarea></td>
                <td><input type="number" class="form-control form-control-sm" value="${row.seed ?? ''}" onchange="dsbUpdateRow(${i}, 'seed', this.value)" placeholder="-" style="width:65px;" min="-1"></td>
                <td class="text-center align-middle"><span class="badge bg-${statusColor}">${statusLabel}</span></td>
                <td class="align-middle">
                    <div class="d-flex align-items-center gap-1">
                        ${actionHtml}
                        ${audioHtml}
                        <button class="btn btn-sm btn-outline-danger ms-auto" onclick="dsbRemoveRow(${i})" title="Delete row"><i class="fas fa-trash"></i></button>
                    </div>
                </td>
            </tr>`;
        }

        function dsbRenderTable(changedIndices) {
            const tbody = document.getElementById('dsb-table-body');

            // Full rebuild if no specific indices or row count changed
            if (!changedIndices || tbody.children.length !== dsbRows.length) {
                tbody.innerHTML = dsbRows.map((row, i) => dsbBuildRowHtml(row, i)).join('');
                dsbUpdateProgress();
                return;
            }

            // Targeted update: only re-render changed rows
            for (const i of changedIndices) {
                const existing = tbody.children[i];
                if (!existing) continue;
                const row = dsbRows[i];
                const oldStatus = existing.getAttribute('data-dsb-status');
                const oldAudio = existing.getAttribute('data-dsb-audio');
                if (oldStatus === (row.status || 'pending') && oldAudio === (row.audio_url || '')) continue;
                const temp = document.createElement('tbody');
                temp.innerHTML = dsbBuildRowHtml(row, i);
                existing.replaceWith(temp.firstElementChild);
            }
            dsbUpdateProgress();
        }

        window.dsbUpdateRow = (index, field, value) => {
            if (dsbRows[index]) {
                dsbRows[index][field] = value;
                dsbSaveRows();
            }
        };

        window.dsbStopOthers = (index) => {
            document.querySelectorAll('#dsb-table-body audio').forEach(audio => {
                const row = audio.closest('tr');
                if (row && parseInt(row.getAttribute('data-dsb-idx')) !== index) audio.pause();
            });
        };

        let dsbLastDoneCount = -1;

        function dsbUpdateProgress() {
            const done = dsbRows.filter(r => r.status === 'done').length;
            const total = dsbRows.length;
            const pct = total > 0 ? Math.round((done / total) * 100) : 0;
            const wrap = document.getElementById('dsb-progress-wrap');
            const bar = document.getElementById('dsb-progress-bar');
            if (done > 0 || dsbBatchRunning) {
                wrap.style.display = '';
                bar.style.width = pct + '%';
                bar.innerText = `${pct}% (${done}/${total})`;
            } else {
                wrap.style.display = 'none';
            }
            // Only rebuild dropdown when done count actually changes
            if (done !== dsbLastDoneCount) {
                dsbLastDoneCount = done;
                dsbUpdateRefDropdown();
            }
        }

        function dsbUpdateRefDropdown() {
            const select = document.getElementById('dsb-ref-select');
            const doneSamples = dsbRows.map((r, i) => ({ index: i, row: r })).filter(x => x.row.status === 'done');
            select.innerHTML = doneSamples.length === 0
                ? '<option value="0">No completed samples yet</option>'
                : doneSamples.map(x => `<option value="${x.index}">${x.index + 1}. ${(x.row.emotion || 'neutral').substring(0, 30)} - "${(x.row.text || '').substring(0, 40)}..."</option>`).join('');
        }

        // Single sample generation
        window.dsbGenSample = async (index) => {
            const name = dsbCurrentProject;
            const rootDesc = document.getElementById('dsb-description').value.trim();
            if (!name) { showToast('Select or create a project first.', 'warning'); return; }
            if (!rootDesc) { showToast('Enter a root voice description first.', 'warning'); return; }

            const row = dsbRows[index];
            if (!row || !row.text.trim()) { showToast('This row has no text.', 'warning'); return; }

            const emotion = row.emotion.trim();
            const description = emotion ? `${rootDesc}, ${emotion}` : rootDesc;

            // Resolve seed: per-line > global > random
            const globalSeed = parseInt(document.getElementById('dsb-global-seed').value);
            const lineSeed = row.seed !== '' ? parseInt(row.seed) : NaN;
            const seed = !isNaN(lineSeed) && lineSeed >= 0 ? lineSeed : (!isNaN(globalSeed) && globalSeed >= 0 ? globalSeed : -1);

            // Optimistic UI
            dsbRows[index].status = 'generating';
            dsbRenderTable([index]);

            try {
                const result = await API.post('/api/dataset_builder/generate_sample', {
                    description,
                    text: row.text.trim(),
                    dataset_name: name,
                    sample_index: index,
                    seed,
                });
                dsbRows[index].status = 'done';
                dsbRows[index].audio_url = result.audio_url;
            } catch (e) {
                dsbRows[index].status = 'error';
                console.error('Sample generation failed:', e);
            }
            dsbRenderTable([index]);
        };

        // Batch generation
        window.dsbGenerateAll = async (regenAll = false) => {
            const name = dsbCurrentProject;
            const rootDesc = document.getElementById('dsb-description').value.trim();
            if (!name) { showToast('Select or create a project first.', 'warning'); return; }
            if (!rootDesc) { showToast('Enter a root voice description first.', 'warning'); return; }

            const samples = dsbRows.filter(r => r.text.trim());
            if (samples.length === 0) { showToast('Add at least one sample with text.', 'warning'); return; }

            const indices = regenAll
                ? dsbRows.map((_, i) => i).filter(i => dsbRows[i].text.trim())
                : dsbRows.map((r, i) => i).filter(i => dsbRows[i].text.trim() && dsbRows[i].status !== 'done');

            if (indices.length === 0) { showToast('All samples are already generated.', 'warning'); return; }
            if (regenAll && !await showConfirm(`Regenerate all ${indices.length} samples?`)) return;

            // Mark as generating
            indices.forEach(i => { dsbRows[i].status = 'generating'; });
            dsbRenderTable();
            dsbBatchRunning = true;
            document.getElementById('dsb-btn-gen-all').style.display = 'none';
            document.getElementById('dsb-btn-regen-all').style.display = 'none';
            document.getElementById('dsb-btn-cancel').style.display = '';
            document.getElementById('dsb-logs').style.display = '';

            const globalSeed = parseInt(document.getElementById('dsb-global-seed').value);
            const perSeeds = dsbRows.map(r => r.seed !== '' && r.seed !== undefined ? parseInt(r.seed) : -1);

            try {
                await API.post('/api/dataset_builder/generate_batch', {
                    name,
                    description: rootDesc,
                    samples: dsbRows.map(r => ({ emotion: r.emotion || '', text: r.text || '' })),
                    indices,
                    global_seed: !isNaN(globalSeed) && globalSeed >= 0 ? globalSeed : -1,
                    seeds: perSeeds,
                });

                // Start polling
                dsbStartPolling(name);
            } catch (e) {
                showToast('Batch generation failed: ' + e.message, 'error');
                dsbStopBatch();
            }
        };

        function dsbStartPolling(name) {
            if (dsbPolling) clearInterval(dsbPolling);
            dsbPolling = setInterval(() => dsbPollStatus(name), 2000);
        }

        async function dsbPollStatus(name, silent = false) {
            try {
                const result = await API.get(`/api/dataset_builder/status/${encodeURIComponent(name)}`);
                const serverSamples = result.samples || [];

                // Merge server state into local rows, creating missing rows
                const changed = [];
                let added = false;
                serverSamples.forEach((s, i) => {
                    if (i < dsbRows.length) {
                        const oldStatus = dsbRows[i].status;
                        const oldAudio = dsbRows[i].audio_url;
                        if (s.status) dsbRows[i].status = s.status;
                        if (s.audio_url) dsbRows[i].audio_url = s.audio_url;
                        if (dsbRows[i].status !== oldStatus || dsbRows[i].audio_url !== oldAudio) changed.push(i);
                    } else {
                        dsbRows.push({
                            emotion: s.description || '',
                            text: s.text || '',
                            seed: s.seed ?? '',
                            status: s.status || 'pending',
                            audio_url: s.audio_url || null
                        });
                        added = true;
                    }
                });

                if (added) {
                    dsbRenderTable();
                } else if (changed.length > 0) {
                    dsbRenderTable(changed);
                }

                // Update logs
                if (result.logs && result.logs.length > 0) {
                    const logsEl = document.getElementById('dsb-logs');
                    logsEl.style.display = '';
                    logsEl.innerText = result.logs.join('\n');
                    logsEl.scrollTop = logsEl.scrollHeight;
                }

                // Resume polling if server is still running (e.g. after page reload)
                if (result.running && !dsbBatchRunning) {
                    dsbBatchRunning = true;
                    dsbStartPolling(name);
                    document.getElementById('dsb-btn-gen-all').style.display = 'none';
                    document.getElementById('dsb-btn-regen-all').style.display = 'none';
                    document.getElementById('dsb-btn-cancel').style.display = '';
                }

                // Check if batch is done
                if (!result.running && dsbBatchRunning) {
                    dsbStopBatch();
                }

                // If not running and this was a one-time check, stop polling
                if (!result.running && silent && dsbPolling) {
                    clearInterval(dsbPolling);
                    dsbPolling = null;
                }
            } catch (e) {
                if (!silent) console.error('Poll error:', e);
                // Status endpoint may 404 if no state.json yet — ignore silently
                if (silent && dsbPolling) {
                    clearInterval(dsbPolling);
                    dsbPolling = null;
                }
            }
        }

        function dsbStopBatch() {
            dsbBatchRunning = false;
            if (dsbPolling) { clearInterval(dsbPolling); dsbPolling = null; }
            document.getElementById('dsb-btn-gen-all').style.display = '';
            document.getElementById('dsb-btn-regen-all').style.display = '';
            document.getElementById('dsb-btn-cancel').style.display = 'none';
            dsbRenderTable();
        }

        window.dsbCancel = async () => {
            try {
                await API.post('/api/dataset_builder/cancel', {});
            } catch (e) { console.error('Cancel error:', e); }
        };

        // Import / Export
        window.dsbImport = (event) => {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    if (!Array.isArray(data)) throw new Error('Expected JSON array');
                    dsbRows = data.map(item => ({
                        emotion: item.emotion || item.instruct || '',
                        text: item.text || '',
                        seed: item.seed ?? '',
                        status: 'pending',
                        audio_url: null,
                    }));
                    dsbRenderTable();
                    dsbSaveRows();
                } catch (err) {
                    showToast('Import failed: ' + err.message, 'error');
                }
            };
            reader.readAsText(file);
            event.target.value = '';  // reset file input
        };

        window.dsbExport = () => {
            const data = dsbRows.map(r => {
                const entry = { emotion: r.emotion, text: r.text };
                if (r.seed !== '' && r.seed !== undefined) entry.seed = parseInt(r.seed);
                return entry;
            });
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const name = dsbCurrentProject || 'dataset';
            a.download = `${name}_script.json`;
            a.click();
            URL.revokeObjectURL(url);
        };

        // Save as training dataset
        window.dsbSave = async () => {
            const name = dsbCurrentProject;
            if (!name) { showToast('Select or create a project first.', 'warning'); return; }

            const doneSamples = dsbRows.filter(r => r.status === 'done');
            if (doneSamples.length === 0) { showToast('No completed samples to save. Generate some first.', 'warning'); return; }

            const refIdx = parseInt(document.getElementById('dsb-ref-select').value) || 0;

            if (!await showConfirm(`Save "${name}" as training dataset with ${doneSamples.length} samples?`)) return;

            const statusEl = document.getElementById('dsb-save-status');
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';

            try {
                const result = await API.post('/api/dataset_builder/save', {
                    name,
                    ref_index: refIdx,
                });
                statusEl.innerHTML = `<span class="text-success"><i class="fas fa-check me-1"></i>Saved! ${result.sample_count} samples.</span>`;
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger">Save failed: ${e.message}</span>`;
            }
        };

        // Persist on input changes
        document.getElementById('dsb-description')?.addEventListener('input', dsbSaveForm);
        document.getElementById('dsb-global-seed')?.addEventListener('input', dsbSaveForm);

        // Init
        applyTheme(getStoredTheme());
        bindAudioProcessingDependency('process-voices-toggle', 'generate-audio-toggle');
        bindAudioProcessingDependency('process-voices-toggle-v2', 'generate-audio-toggle-v2');
        initPromptTextareaAutosize();
        loadConfig();
        loadVoices();
        loadSavedScripts();
        loadDesignedVoices();
        dsbLoadProjects();
        reconnectTaskLogs().catch(err => console.error('initial reconnect failed', err));
        refreshProcessingWorkflowStatus().catch(err => console.error('initial processing reconnect failed', err));
