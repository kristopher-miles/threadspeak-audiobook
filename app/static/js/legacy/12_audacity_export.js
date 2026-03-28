        // --- Audacity Export ---
        window.exportAudacity = async () => {
            const statusEl = document.getElementById('audacity-status');
            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Exporting...</span>';

            try {
                await API.post('/api/export_audacity', {});

                const poll = setInterval(async () => {
                    try {
                        const status = await API.get('/api/status/audacity_export');
                        if (!status.running) {
                            clearInterval(poll);
                            if (status.logs.some(l => l.includes("complete"))) {
                                statusEl.innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Done!</span>';
                                // Auto-download the zip
                                const a = document.createElement('a');
                                a.href = `/api/export_audacity?t=${Date.now()}`;
                                a.download = 'audacity_export.zip';
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                setTimeout(() => { statusEl.innerHTML = ''; }, 5000);
                            } else {
                                const lastLog = status.logs[status.logs.length - 1] || 'Unknown error';
                                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${lastLog}</span>`;
                            }
                        }
                    } catch (e) {
                        clearInterval(poll);
                        statusEl.innerHTML = `<span class="text-danger">Poll error: ${e.message}</span>`;
                    }
                }, 1000);
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${e.message}</span>`;
            }
        };

        // Handle M4B cover image upload
        document.getElementById('m4b-cover-input').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const statusEl = document.getElementById('m4b-cover-status');
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try {
                const resp = await fetch('/api/m4b_cover', { method: 'POST', body: formData });
                if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
                statusEl.textContent = 'Uploaded';
                statusEl.className = 'small text-success';
            } catch (err) {
                statusEl.textContent = err.message;
                statusEl.className = 'small text-danger';
            }
        });

        window.exportM4B = async () => {
            const statusEl = document.getElementById('m4b-status');
            const perChunk = document.getElementById('m4b-per-chunk').checked;
            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Exporting M4B...</span>';

            try {
                await API.post('/api/merge_m4b', {
                    per_chunk_chapters: perChunk,
                    title: document.getElementById('m4b-title').value,
                    author: document.getElementById('m4b-author').value,
                    narrator: document.getElementById('m4b-narrator').value,
                    year: document.getElementById('m4b-year').value,
                    description: document.getElementById('m4b-description').value
                });

                const poll = setInterval(async () => {
                    try {
                        const status = await API.get('/api/status/m4b_export');
                        if (!status.running) {
                            clearInterval(poll);
                            if (status.logs.some(l => l.includes("complete"))) {
                                statusEl.innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Done!</span>';
                                const a = document.createElement('a');
                                a.href = `/api/audiobook_m4b?t=${Date.now()}`;
                                a.download = 'audiobook.m4b';
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                setTimeout(() => { statusEl.innerHTML = ''; }, 5000);
                            } else {
                                const lastLog = status.logs[status.logs.length - 1] || 'Unknown error';
                                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${lastLog}</span>`;
                            }
                        }
                    } catch (e) {
                        clearInterval(poll);
                        statusEl.innerHTML = `<span class="text-danger">Poll error: ${e.message}</span>`;
                    }
                }, 1000);
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${e.message}</span>`;
            }
        };

