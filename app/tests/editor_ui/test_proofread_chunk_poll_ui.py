"""Editor tab legacy JS harness tests split by behavior area."""

from ._editor_tab_node_harness import EditorTabChunkPollTests


class ProofreadChunkPollUiTests(EditorTabChunkPollTests):
    def test_regenerate_proofread_chunk_uses_targeted_tracker(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const proofreadUpdates = [];
                let loadChunksCalls = 0;
                const chunkStates = [
                    { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'done', audio_path: 'voicelines/proofread.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url) => {
                    if (url === '/api/chunks/3/regenerate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/3') {
                        return chunkStates.shift() || { id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'done', audio_path: 'voicelines/proofread.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 3, speaker: 'Narrator', text: 'Proofread row update', instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {{ loadChunksCalls += 1; }});
                context.__editorTabTestHooks.setUpdateProofreadRow((chunk) => {{
                    proofreadUpdates.push(chunk.status);
                    return true;
                }});

                await context.window.regenerateProofreadChunk('3');
                await flushTicks();

                assert.deepStrictEqual(proofreadUpdates.slice(-2), ['generating', 'done']);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
                assert.strictEqual(loadChunksCalls, 0, 'proofread regeneration should not trigger broad chunk reloads');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_regenerate_proofread_chunk_sends_neutral_narrator_toggle(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                let regeneratePayload = null;
                const neutralToggle = context.document.getElementById('editor-neutral-narrator');
                neutralToggle.checked = true;

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/12/regenerate') {
                        regeneratePayload = payload;
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/12') {
                        return {
                            id: 12,
                            speaker: 'NARRATOR',
                            text: 'Proofread neutral narrator line',
                            instruct: 'Furious with anger, shouting.',
                            status: 'done',
                            audio_path: 'voicelines/proofread-neutral.wav',
                            audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                        };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{
                    id: 12,
                    speaker: 'NARRATOR',
                    text: 'Proofread neutral narrator line',
                    instruct: 'Furious with anger, shouting.',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                }]);

                await context.window.regenerateProofreadChunk('12');
                await flushTicks();

                assert.strictEqual(regeneratePayload.neutral_narrator, true);
                assert.deepStrictEqual(Object.keys(regeneratePayload), ['neutral_narrator']);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_regenerate_proofread_chunk_keeps_generating_state_during_initial_pending_race(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const proofreadUpdates = [];
                const chunkStates = [
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'pending', audio_path: null, audio_validation: null },
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'done', audio_path: 'voicelines/proofread-race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url) => {
                    if (url === '/api/chunks/5/regenerate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/5') {
                        return chunkStates.shift() || { id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'done', audio_path: 'voicelines/proofread-race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 5, speaker: 'Narrator', text: 'Proofread pending race', instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setUpdateProofreadRow((chunk) => {
                    proofreadUpdates.push(chunk.status);
                    return true;
                });

                const regenerationPromise = context.window.regenerateProofreadChunk('5');
                await flushTicks(1);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'generating');

                await regenerationPromise;
                await flushTicks();

                assert.deepStrictEqual(proofreadUpdates.slice(-2), ['generating', 'done']);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_proofread_chapter_options_use_summaries_not_chunk_rows(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.API.get = async (url) => {
                    if (url.startsWith('/api/proofread/view?')) {
                        return {
                            chunks: [
                                { id: 1, uid: 'chunk-1', speaker: 'Narrator', chapter: 'Chapter 1', text: 'one', status: 'done', audio_path: null, audio_validation: null },
                            ],
                            chapters: [{ chapter: 'Chapter 1', chunk_count: 2 }, { chapter: 'Chapter 2', chunk_count: 1 }, { chapter: 'Chapter 3', chunk_count: 3 }],
                            pagination: { page: 1, page_size: 2000, total: 1, has_next: false },
                            stats: {
                                chapter: { total: 1, passed: 0, failed: 0, auto_failed: 0, skipped: 1 },
                                project: { total: 6, passed: 0, failed: 0, auto_failed: 0, skipped: 6 },
                            },
                        };
                    }
                    if (url === '/api/status/proofread') {
                        return { running: false, progress: {}, logs: [] };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                await context.window.loadProofreadData({ includeChapters: true });
                await flushTicks();

                const select = context.document.getElementById('proofread-chapter-select');
                const values = Array.from(select.innerHTML.matchAll(/value=\"([^\"]+)\"/g), match => match[1]);
                const uniqueValues = [...new Set(values)];
                assert.ok(values.includes('__whole_project__'), 'proofread chapter selector should include whole project option');
                assert.ok(uniqueValues.includes('Chapter 1'), 'Chapter 1 should be listed from summaries');
                assert.ok(uniqueValues.includes('Chapter 2'), 'Chapter 2 should be listed from summaries');
                assert.ok(uniqueValues.includes('Chapter 3'), 'Chapter 3 should be listed from summaries');
                assert.strictEqual(values.length, uniqueValues.length, 'proofread options should not duplicate chapter entries');
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_proofread_chapter_switch_uses_full_project_cache_after_chapter_scoped_editor_reload(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.__editorTabTestHooks.setCachedChunks([
                    { id: 1, uid: 'chunk-1', speaker: 'Narrator', chapter: 'Chapter 1', text: 'one', instruct: '', status: 'done', audio_path: null, audio_validation: null },
                    { id: 2, uid: 'chunk-2', speaker: 'Narrator', chapter: 'Chapter 2', text: 'two', instruct: '', status: 'done', audio_path: null, audio_validation: null },
                ]);

                context.API.get = async (url) => {
                    if (url.startsWith('/api/proofread/view?')) {
                        if (url.includes('chapter=Chapter+2') || url.includes('chapter=Chapter%202')) {
                            return {
                                chunks: [
                                    { id: 2, uid: 'chunk-2', speaker: 'Narrator', chapter: 'Chapter 2', text: 'two', status: 'done', audio_path: null, audio_validation: null },
                                ],
                                chapters: [{ chapter: 'Chapter 1', chunk_count: 1 }, { chapter: 'Chapter 2', chunk_count: 1 }],
                                pagination: { page: 1, page_size: 2000, total: 1, has_next: false },
                                stats: {
                                    chapter: { total: 1, passed: 0, failed: 0, auto_failed: 0, skipped: 1 },
                                    project: { total: 2, passed: 0, failed: 0, auto_failed: 0, skipped: 2 },
                                },
                            };
                        }
                        return {
                            chunks: [
                                { id: 1, uid: 'chunk-1', speaker: 'Narrator', chapter: 'Chapter 1', text: 'one (updated)', status: 'done', audio_path: null, audio_validation: null },
                            ],
                            chapters: [{ chapter: 'Chapter 1', chunk_count: 1 }, { chapter: 'Chapter 2', chunk_count: 1 }],
                            pagination: { page: 1, page_size: 2000, total: 1, has_next: false },
                            stats: {
                                chapter: { total: 1, passed: 0, failed: 0, auto_failed: 0, skipped: 1 },
                                project: { total: 2, passed: 0, failed: 0, auto_failed: 0, skipped: 2 },
                            },
                        };
                    }
                    if (url === '/api/status/proofread') {
                        return { running: false, progress: {}, logs: [] };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                await context.window.loadProofreadData({ chapterId: 'Chapter 1', includeChapters: true });
                await flushTicks();
                await context.window.changeProofreadChapter('Chapter 2');
                await flushTicks();

                const cached = context.__editorTabTestHooks.getCachedChunks();
                assert.ok(cached.some(chunk => String(chunk.chapter) === 'Chapter 2'), 'expected full-project cache to retain other chapters');

                const proofreadBody = context.document.getElementById('proofread-table-body');
                assert.ok(String(proofreadBody.innerHTML || '').includes('two'), 'proofread table should render Chapter 2 rows');
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_jump_to_first_proofread_failure_uses_dedicated_endpoint(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const requested = [];
                context.API.get = async (url) => {
                    requested.push(url);
                    if (url === '/api/proofread/next_failure?') {
                        return { found: true, uid: 'chunk-2', chapter: 'Chapter 2', ordinal: 2 };
                    }
                    if (url.startsWith('/api/proofread/view?')) {
                        return {
                            chunks: [
                                { id: 2, uid: 'chunk-2', speaker: 'Narrator', chapter: 'Chapter 2', text: 'failure row', status: 'done', audio_path: null, audio_validation: null, proofread: { checked: true, passed: false } },
                            ],
                            chapters: [{ chapter: 'Chapter 1', chunk_count: 1 }, { chapter: 'Chapter 2', chunk_count: 1 }],
                            pagination: { page: 1, page_size: 2000, total: 1, has_next: false },
                            stats: {
                                chapter: { total: 1, passed: 0, failed: 1, auto_failed: 0, skipped: 1 },
                                project: { total: 2, passed: 0, failed: 1, auto_failed: 0, skipped: 2 },
                            },
                        };
                    }
                    if (url === '/api/status/proofread') {
                        return { running: false, progress: {}, logs: [] };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                await context.window.jumpToFirstProofreadFailure();
                await flushTicks();

                assert.ok(requested.includes('/api/proofread/next_failure?'));
                assert.ok(
                    requested.some((url) => (
                        url.startsWith('/api/proofread/view?')
                        && (url.includes('chapter=Chapter+2') || url.includes('chapter=Chapter%202'))
                    ))
                );
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )
