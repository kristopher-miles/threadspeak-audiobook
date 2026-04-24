// Persistent model-download progress toast.
(function () {
    let modelDownloadSource = null;
    let modelDownloadPollTimer = null;
    let modelDownloadClearTimer = null;
    let latestModelDownloadSnapshot = { downloads: [] };

    function escapeModelDownloadHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatModelDownloadBytes(value) {
        const bytes = Number(value || 0);
        if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = bytes;
        let unitIndex = 0;
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex += 1;
        }
        const precision = size >= 10 || unitIndex === 0 ? 0 : 1;
        return `${size.toFixed(precision)} ${units[unitIndex]}`;
    }

    function formatModelDownloadSpeed(value) {
        const speed = Number(value || 0);
        if (!Number.isFinite(speed) || speed <= 0) return '--';
        return `${formatModelDownloadBytes(speed)}/s`;
    }

    function formatModelDownloadEta(value) {
        const seconds = Number(value);
        if (!Number.isFinite(seconds) || seconds < 0) return '--';
        if (seconds < 1) return 'now';
        if (seconds < 60) return `${Math.ceil(seconds)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainder = Math.ceil(seconds % 60);
        if (minutes < 60) return `${minutes}m ${remainder}s`;
        const hours = Math.floor(minutes / 60);
        return `${hours}h ${minutes % 60}m`;
    }

    function ensureModelDownloadToastElement() {
        let el = document.getElementById('model-download-toast');
        if (el) return el;
        el = document.createElement('div');
        el.id = 'model-download-toast';
        el.className = 'model-download-toast';
        el.style.display = 'none';
        document.body.appendChild(el);
        return el;
    }

    function summarizeModelDownload(downloads) {
        const failed = downloads.find(item => item.status === 'failed');
        const active = downloads.find(item => item.status === 'active');
        return failed || active || downloads[0] || null;
    }

    function modelDownloadPercent(item) {
        const total = Number(item?.total_bytes || 0);
        const downloaded = Number(item?.downloaded_bytes || 0);
        if (!total) return item?.status === 'completed' ? 100 : 0;
        return Math.max(0, Math.min(100, Math.round((downloaded / total) * 100)));
    }

    function renderModelDownloadFileRows(item) {
        const files = Array.isArray(item?.files) ? item.files : [];
        const visible = files.length > 1 || item?.status === 'failed'
            ? files.filter(row => row.status !== 'completed' || item.status === 'failed')
            : [];
        if (!visible.length) return '';
        return `
            <div class="model-download-files">
                ${visible.map(row => {
                    const total = Number(row.total_bytes || 0);
                    const downloaded = Number(row.downloaded_bytes || 0);
                    const percent = total ? Math.max(0, Math.min(100, Math.round((downloaded / total) * 100))) : 0;
                    return `
                        <div class="model-download-file-row">
                            <div class="model-download-file-name">${escapeModelDownloadHtml(row.filename || 'model file')}</div>
                            <div class="model-download-file-meta">
                                <span>${formatModelDownloadBytes(downloaded)}${total ? ` / ${formatModelDownloadBytes(total)}` : ''}</span>
                                <span>${row.status === 'failed' ? 'failed' : `${percent}%`}</span>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    function renderModelDownloadToast(snapshot) {
        latestModelDownloadSnapshot = snapshot || { downloads: [] };
        const el = ensureModelDownloadToastElement();
        const downloads = (Array.isArray(latestModelDownloadSnapshot.downloads) ? latestModelDownloadSnapshot.downloads : [])
            .filter(item => ['active', 'failed', 'completed'].includes(String(item?.status || '')));

        if (!downloads.length) {
            el.style.display = 'none';
            el.innerHTML = '';
            return;
        }

        const item = summarizeModelDownload(downloads);
        if (!item) {
            el.style.display = 'none';
            el.innerHTML = '';
            return;
        }

        const failed = downloads.some(row => row.status === 'failed');
        const activeCount = downloads.filter(row => row.status === 'active').length;
        const percent = modelDownloadPercent(item);
        const title = failed ? 'Model download failed' : item.status === 'completed' ? 'Model download complete' : 'Downloading model weights';
        const subtitle = activeCount > 1 ? `${item.display_name || item.repo_id || 'Model'} and ${activeCount - 1} more` : (item.display_name || item.repo_id || 'Model');
        const retryHtml = failed && item.retryable
            ? `<button type="button" class="btn btn-sm btn-outline-light model-download-retry" onclick="retryModelDownload('${escapeModelDownloadHtml(item.id)}')">Retry</button>`
            : '';

        el.className = `model-download-toast ${failed ? 'is-failed' : ''}`;
        el.innerHTML = `
            <div class="model-download-header">
                <div>
                    <div class="model-download-title">${escapeModelDownloadHtml(title)}</div>
                    <div class="model-download-model">${escapeModelDownloadHtml(subtitle)}</div>
                </div>
                ${retryHtml}
            </div>
            <div class="model-download-progress" role="progressbar" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100">
                <div class="model-download-progress-bar" style="width:${percent}%"></div>
            </div>
            <div class="model-download-stats">
                <span>${formatModelDownloadBytes(item.downloaded_bytes)}${item.total_bytes ? ` / ${formatModelDownloadBytes(item.total_bytes)}` : ''}</span>
                <span>${formatModelDownloadSpeed(item.speed_bps)}</span>
                <span>ETA ${formatModelDownloadEta(item.eta_seconds)}</span>
            </div>
            ${failed && item.error ? `<div class="model-download-error">${escapeModelDownloadHtml(item.error)}</div>` : ''}
            ${renderModelDownloadFileRows(item)}
        `;
        el.style.display = 'block';

        if (!failed && downloads.every(row => row.status === 'completed')) {
            if (modelDownloadClearTimer) clearTimeout(modelDownloadClearTimer);
            modelDownloadClearTimer = setTimeout(() => {
                if ((latestModelDownloadSnapshot.downloads || []).every(row => row.status === 'completed')) {
                    renderModelDownloadToast({ downloads: [] });
                }
            }, 3500);
        }
    }

    async function retryModelDownload(downloadId) {
        const id = String(downloadId || '').trim();
        if (!id) return;
        try {
            const response = await fetch(`/api/model_downloads/retry/${encodeURIComponent(id)}`, { method: 'POST' });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || `Retry failed (${response.status})`);
            }
            const payload = await response.json();
            renderModelDownloadToast({ downloads: [payload] });
        } catch (error) {
            if (typeof showToast === 'function') {
                showToast(`Download retry failed: ${error.message}`, 'error', 7000);
            }
        }
    }

    async function pollModelDownloadStatus() {
        try {
            const response = await fetch('/api/model_downloads/status');
            if (!response.ok) return;
            renderModelDownloadToast(await response.json());
        } catch (_) {
            // Ignore transient status failures; the next SSE reconnect/poll will refresh.
        }
    }

    function ensureModelDownloadPolling() {
        if (modelDownloadPollTimer) return;
        modelDownloadPollTimer = setInterval(pollModelDownloadStatus, 5000);
        pollModelDownloadStatus();
    }

    function connectModelDownloadEvents() {
        if (!window.EventSource) {
            ensureModelDownloadPolling();
            return;
        }
        if (modelDownloadSource) {
            modelDownloadSource.close();
            modelDownloadSource = null;
        }
        const source = new EventSource('/api/model_downloads/events');
        modelDownloadSource = source;
        source.addEventListener('snapshot', event => {
            try {
                renderModelDownloadToast(JSON.parse(event.data || '{"downloads":[]}'));
            } catch (error) {
                console.warn('Failed to render model download snapshot', error);
            }
        });
        source.addEventListener('open', () => {
            if (modelDownloadPollTimer) {
                clearInterval(modelDownloadPollTimer);
                modelDownloadPollTimer = null;
            }
        });
        source.onerror = () => {
            ensureModelDownloadPolling();
        };
    }

    window.formatModelDownloadBytes = formatModelDownloadBytes;
    window.renderModelDownloadToast = renderModelDownloadToast;
    window.retryModelDownload = retryModelDownload;
    window.connectModelDownloadEvents = connectModelDownloadEvents;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', connectModelDownloadEvents, { once: true });
    } else {
        connectModelDownloadEvents();
    }
})();
