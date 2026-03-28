        // --- Dictionary Tab ---
        let dictionaryEntries = [];
        let dictionaryCounts = [];

        function normalizeDictionaryRows(entries) {
            if (!Array.isArray(entries)) return [];
            return entries.map(entry => ({
                source: (entry?.source || '').trim(),
                alias: (entry?.alias || '').trim(),
            }));
        }

        function getSavedDictionaryEntries() {
            return normalizeDictionaryRows(dictionaryEntries).filter(entry => entry.source && entry.alias);
        }

        function getActiveDictionaryRows(entries = dictionaryEntries) {
            return normalizeDictionaryRows(entries)
                .map((entry, rowIndex) => ({ ...entry, rowIndex }))
                .filter(entry => entry.source && entry.alias);
        }

        function ensureTrailingDictionaryRow(entries) {
            const rows = normalizeDictionaryRows(entries);
            const last = rows[rows.length - 1];
            if (!last || last.source || last.alias) {
                rows.push({ source: '', alias: '' });
            }
            return rows;
        }

        function escapeRegex(text) {
            return (text || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }

        function dictionaryPatternFor(source) {
            const escaped = escapeRegex(source);
            return /\w/.test(source)
                ? new RegExp(`(?<!\\w)${escaped}(?!\\w)`, 'gi')
                : new RegExp(escaped, 'gi');
        }

        function applyAliasCase(alias, matchedText) {
            if (!matchedText) return alias;
            if (matchedText === matchedText.toUpperCase()) return alias.toUpperCase();
            if (matchedText === matchedText.toLowerCase()) return alias.toLowerCase();
            const words = matchedText.match(/[A-Za-z]+/g) || [];
            if (words.length > 0 && words.every(word => word[0] === word[0].toUpperCase() && word.slice(1) === word.slice(1).toLowerCase())) {
                return alias.replace(/\b\w+/g, word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
            }
            if (matchedText.charAt(0) === matchedText.charAt(0).toUpperCase()) {
                return alias.charAt(0).toUpperCase() + alias.slice(1);
            }
            return alias;
        }

        function applyDictionaryEntriesToText(text, entries) {
            let currentText = text || '';
            const counts = [];
            const placeholders = new Map();
            let placeholderIndex = 0;
            getActiveDictionaryRows(entries).forEach(entry => {
                    const pattern = dictionaryPatternFor(entry.source);
                    let count = 0;
                    currentText = currentText.replace(pattern, match => {
                        count += 1;
                        const token = `\uE000${placeholderIndex}\uE001`;
                        placeholderIndex += 1;
                        placeholders.set(token, applyAliasCase(entry.alias, match));
                        return token;
                    });
                    counts.push(count);
                });
            placeholders.forEach((replacement, token) => {
                currentText = currentText.split(token).join(replacement);
            });
            return { text: currentText, counts };
        }

        async function getDictionaryChunks() {
            if (cachedChunks.length > 0) return cachedChunks;
            try {
                const chunks = await API.get('/api/chunks');
                if (Array.isArray(chunks)) return chunks;
            } catch (e) {
                return [];
            }
            return [];
        }

        async function refreshDictionaryCounts(chunksOverride = null) {
            const chunks = Array.isArray(chunksOverride) ? chunksOverride : await getDictionaryChunks();
            const texts = chunks.map(chunk => chunk.text || '');
            const activeEntries = getActiveDictionaryRows();
            dictionaryCounts = new Array(dictionaryEntries.length).fill(0);

            texts.forEach(text => {
                const { counts } = applyDictionaryEntriesToText(text, activeEntries);
                counts.forEach((count, index) => {
                    const rowIndex = activeEntries[index]?.rowIndex;
                    if (rowIndex != null) {
                        dictionaryCounts[rowIndex] = (dictionaryCounts[rowIndex] || 0) + count;
                    }
                });
            });

            document.querySelectorAll('#dictionary-table-body .dictionary-count-cell').forEach(cell => {
                const index = Number(cell.dataset.index || -1);
                cell.textContent = Number.isInteger(index) && index >= 0 ? String(dictionaryCounts[index] || 0) : '0';
            });
        }

        function renderDictionaryTable() {
            const tbody = document.getElementById('dictionary-table-body');
            if (!tbody) return;

            dictionaryEntries = ensureTrailingDictionaryRow(dictionaryEntries);
            tbody.innerHTML = dictionaryEntries.map((entry, index) => `
                <tr>
                    <td><input type="text" class="form-control form-control-sm" value="${escapeHtml(entry.source || '')}" oninput="updateDictionaryEntry(${index}, 'source', this.value)"></td>
                    <td><input type="text" class="form-control form-control-sm" value="${escapeHtml(entry.alias || '')}" oninput="updateDictionaryEntry(${index}, 'alias', this.value)"></td>
                    <td class="text-muted small dictionary-count-cell" data-index="${index}">${dictionaryCounts[index] || 0}</td>
                </tr>
            `).join('');
        }

        window.updateDictionaryEntry = async (index, field, value) => {
            if (!dictionaryEntries[index]) {
                dictionaryEntries[index] = { source: '', alias: '' };
            }
            dictionaryEntries[index][field] = value;
            const beforeLength = dictionaryEntries.length;
            dictionaryEntries = ensureTrailingDictionaryRow(dictionaryEntries);
            if (dictionaryEntries.length !== beforeLength) {
                renderDictionaryTable();
            }
            await refreshDictionaryCounts();
            const statusEl = document.getElementById('dictionary-save-status');
            if (statusEl) statusEl.textContent = 'Unsaved changes';
        };

        function parseDelimitedLine(line, delimiter) {
            const cells = [];
            let current = '';
            let inQuotes = false;

            for (let i = 0; i < line.length; i += 1) {
                const char = line[i];
                if (char === '"') {
                    if (inQuotes && line[i + 1] === '"') {
                        current += '"';
                        i += 1;
                    } else {
                        inQuotes = !inQuotes;
                    }
                } else if (char === delimiter && !inQuotes) {
                    cells.push(current);
                    current = '';
                } else {
                    current += char;
                }
            }
            cells.push(current);
            return cells.map(cell => cell.trim());
        }

        function parseDictionaryFile(text) {
            const lines = (text || '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
            if (lines.length === 0) return [];
            const delimiter = lines.some(line => line.includes('\t')) ? '\t' : ',';
            let startIndex = 0;
            const firstRow = parseDelimitedLine(lines[0], delimiter).map(cell => cell.toLowerCase());
            if (firstRow.length >= 2 && firstRow[0] === 'source' && firstRow[1] === 'alias') {
                startIndex = 1;
            }

            const entries = [];
            for (let i = startIndex; i < lines.length; i += 1) {
                const [source = '', alias = ''] = parseDelimitedLine(lines[i], delimiter);
                if (source.trim() && alias.trim()) {
                    entries.push({ source: source.trim(), alias: alias.trim() });
                }
            }
            return entries;
        }

        window.triggerDictionaryImport = () => {
            const input = document.getElementById('dictionary-import-input');
            if (input) input.click();
        };

        window.handleDictionaryImport = async (input) => {
            const [file] = input.files || [];
            if (!file) return;
            const text = await file.text();
            const imported = parseDictionaryFile(text);
            dictionaryEntries = ensureTrailingDictionaryRow(imported);
            dictionaryCounts = new Array(dictionaryEntries.length).fill(0);
            renderDictionaryTable();
            await refreshDictionaryCounts();
            const statusEl = document.getElementById('dictionary-save-status');
            if (statusEl) statusEl.textContent = `Imported ${imported.length} entr${imported.length === 1 ? 'y' : 'ies'}; save to persist`;
            input.value = '';
        };

        window.exportDictionaryCsv = () => {
            const entries = getSavedDictionaryEntries();
            const lines = ['source,alias', ...entries.map(entry => {
                const source = `"${entry.source.replace(/"/g, '""')}"`;
                const alias = `"${entry.alias.replace(/"/g, '""')}"`;
                return `${source},${alias}`;
            })];
            const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'dictionary.csv';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        };

        window.saveDictionary = async () => {
            const statusEl = document.getElementById('dictionary-save-status');
            try {
                const payload = { entries: getSavedDictionaryEntries() };
                const result = await API.post('/api/dictionary', payload);
                dictionaryEntries = ensureTrailingDictionaryRow(result.entries || []);
                dictionaryCounts = new Array(dictionaryEntries.length).fill(0);
                renderDictionaryTable();
                await refreshDictionaryCounts();
                if (statusEl) statusEl.textContent = 'Saved';
            } catch (e) {
                if (statusEl) statusEl.textContent = 'Save failed';
                showToast(`Dictionary save failed: ${e.message}`, 'error');
            }
        };

        async function loadDictionary() {
            try {
                const result = await API.get('/api/dictionary');
                dictionaryEntries = ensureTrailingDictionaryRow(result.entries || []);
                dictionaryCounts = new Array(dictionaryEntries.length).fill(0);
                renderDictionaryTable();
                await refreshDictionaryCounts();
                const statusEl = document.getElementById('dictionary-save-status');
                if (statusEl) statusEl.textContent = '';
            } catch (e) {
                showToast(`Failed to load dictionary: ${e.message}`, 'error');
            }
        }

