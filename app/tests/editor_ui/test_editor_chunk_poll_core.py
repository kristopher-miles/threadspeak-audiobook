"""Editor tab legacy JS harness tests split by behavior area."""

from ._editor_tab_node_harness import EditorTabChunkPollTests


class EditorChunkPollCoreTests(EditorTabChunkPollTests):
    def test_generate_chunk_polls_until_done_without_broad_reload(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 1,
                    speaker: 'Narrator',
                    text: 'A finished clip should update in place.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('1', row);

                let loadChunksCalls = 0;
                const savePayloads = [];
                const chunkStates = [
                    { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/clip.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/1') {
                        savePayloads.push(payload);
                        return { id: 1, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/1/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/1') {
                        return chunkStates.shift() || { id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/clip.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 1, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {{ loadChunksCalls += 1; }});

                await context.window.generateChunk('1');
                await flushTicks();

                assert.strictEqual(savePayloads.length, 1);
                assert.ok(row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
                assert.strictEqual(loadChunksCalls, 0, 'single-clip tracking should not trigger broad chunk reloads');
                assert.ok(row.__generateSlot.button, 'generate button should be restored');
                assert.strictEqual(row.__generateSlot.progress, null, 'progress bar should be removed');
                assert.ok(row.__audioContainer.audio, 'audio player should be inserted after completion');
                assert.strictEqual(row.__audioContainer.audio !== null, true);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_render_all_surfaces_missing_voice_and_focuses_voice_row(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const focusedVoices = [];
                context.window.focusVoiceCard = async (voiceName) => {
                    focusedVoices.push(voiceName);
                    return true;
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/view') {
                        return [{
                            id: 0,
                            uid: 'aerial-0',
                            speaker: 'Aerial',
                            text: 'Aerial line with enough words to render correctly.',
                            status: 'pending',
                            chapter: 'Chapter 1',
                        }];
                    }
                    return [];
                };
                context.API.post = async (url) => {
                    if (url === '/api/generate_batch') {
                        const error = new Error('Cannot render because "Aerial" has no voice selected.');
                        error.detail = {
                            code: 'voice_config_required',
                            message: 'Cannot render because "Aerial" has no voice selected.',
                            speaker: 'Aerial',
                            voice_speaker: 'Aerial',
                        };
                        throw error;
                    }
                    if (url === '/api/chunks/sync_from_script_if_stale') {
                        return { synced: false };
                    }
                    return { status: 'ok' };
                };

                vm.createContext(context);
                vm.runInContext(source, context);

                context.document.getElementById('editor-chapter-only').checked = false;
                await context.window.renderAll(false);
                await flushTicks();

                assert.ok(
                    context.__toasts.some((entry) => entry.message.includes('Aerial') && entry.level === 'warning'),
                    'expected a warning toast for the missing voice'
                );
                assert.deepStrictEqual(focusedVoices, ['Aerial']);
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_render_editor_progress_bar_uses_whole_book_audio_coverage_summary(self):
        self._run_node_test(
            """
            (() => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.renderEditorProgressBar([], {
                    running: true,
                    current_job: { processed_clips: 2, total_chunks: 9 },
                    audio_coverage: {
                        total_clips: 10,
                        valid_clips: 7,
                        invalid_clips: 3,
                        percentage: 70,
                    },
                });

                const progressBar = context.document.getElementById('full-progress-bar');
                assert.strictEqual(progressBar.style.width, '70%');
                assert.strictEqual(progressBar.innerText, '70% (7/10)');
                assert.ok(progressBar.classList.contains('bg-info'));
                assert.ok(!progressBar.classList.contains('bg-warning'));
                assert.ok(String(progressBar.title || '').includes('7 of 10 clips'));
            })();
            """
        )

    def test_generate_chunk_prefers_targeted_poll_results_over_stale_broad_state(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 6,
                    speaker: 'Narrator',
                    text: 'The focused chunk poll should win even if broad state is stale.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('6', row);

                let loadChunksCalls = 0;
                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/6') {
                        return { id: 6, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/6/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/6') {
                        return { id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/overlay.mp3', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                context.__editorTabTestHooks.setLoadChunks(async () => {
                    loadChunksCalls += 1;
                    context.__editorTabTestHooks.setCachedChunks([{ id: 6, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);
                });

                await context.window.generateChunk('6');
                await flushTicks();

                assert.strictEqual(loadChunksCalls, 0, 'focused tracking should not fall back to broad reloads');
                assert.ok(row.classList.contains('status-done'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].audio_path, 'voicelines/overlay.mp3');
                assert.ok(row.__audioContainer.audio, 'audio player should be rendered from the targeted poll result');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_queued_chunk_updates_apply_after_stale_reload_finishes(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const initialChunk = {
                    id: 8,
                    speaker: 'Narrator',
                    text: 'Burst updates should survive a stale reload.',
                    instruct: '',
                    chapter: 'Chapter 1',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                };
                const row = context.__createChunkRow(initialChunk);
                context.__rows.set('8', row);

                let resolveChunksView = null;
                const staleChunksView = new Promise((resolve) => {
                    resolveChunksView = resolve;
                });

                context.API.get = async (url) => {
                    if (url === '/api/chunks/chapters') {
                        return { chapters: [{ chapter: 'Chapter 1', chunk_count: 1, narrator_label: '' }] };
                    }
                    if (url === '/api/chunks/view' || url === '/api/chunks/view?chapter=Chapter%201') {
                        return staleChunksView;
                    }
                    if (url === '/api/voices') {
                        return [];
                    }
                    if (url === '/api/narrator_candidates?chapter=Chapter%201') {
                        return { chapter: 'Chapter 1', voices: ['NARRATOR'] };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([initialChunk]);
                const tbody = context.document.getElementById('chunks-table-body');
                tbody.children = [row];
                tbody.querySelectorAll = (selector) => {
                    if (selector === 'tr[data-id]') return [row];
                    return [];
                };

                const loadPromise = context.loadChunks(false);
                await flushTicks(1);

                context.__editorTabTestHooks.enqueueTrackedChunkUpdate({
                    ...initialChunk,
                    status: 'generating',
                });
                context.__editorTabTestHooks.enqueueTrackedChunkUpdate({
                    ...initialChunk,
                    status: 'done',
                    audio_path: 'voicelines/final-8.wav',
                    audio_validation: { file_size_bytes: 42, actual_duration_sec: 2.5 },
                });

                resolveChunksView([initialChunk]);
                await loadPromise;
                await flushTicks();

                const finalChunk = context.__editorTabTestHooks.getCachedChunks()[0];
                assert.strictEqual(finalChunk.status, 'done');
                assert.strictEqual(finalChunk.audio_path, 'voicelines/final-8.wav');
                assert.ok(row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.ok(row.__audioContainer.audio, 'audio player should be restored after queued completion');
                assert.ok(String(row.__audioContainer.audio.src || '').includes('voicelines/final-8.wav'));
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_generate_chunk_polls_until_error_and_restores_button(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 2,
                    speaker: 'Narrator',
                    text: 'A failed clip should recover the button.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('2', row);

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/2') {
                        return { id: 2, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/2/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/2') {
                        return { id: 2, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'error', audio_path: null, audio_validation: { error: 'tts failed' } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 2, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);

                await context.window.generateChunk('2');
                await flushTicks();

                assert.ok(!row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.ok(row.__generateSlot.button, 'button should return after error');
                assert.strictEqual(row.__generateSlot.progress, null);
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'error');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_generate_chunk_keeps_generating_state_during_initial_pending_race(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 4,
                    speaker: 'Narrator',
                    text: 'Pending should not wipe the optimistic generating state.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });
                context.__rows.set('4', row);

                let getCalls = 0;
                const chunkStates = [
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null },
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'generating', audio_path: null, audio_validation: null },
                    { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                ];

                context.API.post = async (url, payload) => {
                    if (url === '/api/chunks/4') {
                        return { id: 4, speaker: payload.speaker, text: payload.text, instruct: payload.instruct, status: 'pending', audio_path: null, audio_validation: null };
                    }
                    if (url === '/api/chunks/4/generate') {
                        return { status: 'started' };
                    }
                    throw new Error(`Unexpected POST ${url}`);
                };
                context.API.get = async (url) => {
                    if (url === '/api/chunks/4') {
                        getCalls += 1;
                        return chunkStates.shift() || { id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'done', audio_path: 'voicelines/race.wav', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } };
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([{ id: 4, speaker: 'Narrator', text: row.__textArea.value, instruct: '', status: 'pending', audio_path: null, audio_validation: null }]);

                const generationPromise = context.window.generateChunk('4');
                await flushTicks(1);
                assert.ok(row.classList.contains('status-generating'), 'optimistic generating state should survive the initial pending poll response');
                assert.ok(row.__generateSlot.progress, 'progress bar should remain visible while backend catches up');

                await generationPromise;
                await flushTicks();

                assert.ok(getCalls >= 3, 'poller should continue past the stale pending response');
                assert.ok(row.classList.contains('status-done'));
                assert.strictEqual(context.__editorTabTestHooks.getCachedChunks()[0].status, 'done');
                assert.strictEqual(context.__editorTabTestHooks.getTrackedChunkPollCount(), 0);
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_update_chunk_row_removes_stale_audio_after_invalidation(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const row = context.__createChunkRow({
                    id: 7,
                    speaker: 'Narrator',
                    text: 'Invalidated narrator audio should disappear.',
                    instruct: '',
                    status: 'done',
                    audio_path: 'voicelines/stale.mp3',
                    audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                });
                row.__audioContainer.audio = createAudioNode('<audio src="/voicelines/stale.mp3?t=old" data-audio-path="voicelines/stale.mp3" data-audio-fingerprint="old"></audio>');
                row.__audioContainer.audio.__container = row.__audioContainer;
                context.__rows.set('7', row);

                vm.createContext(context);
                vm.runInContext(source, context);

                const changed = context.updateChunkRow({
                    id: 7,
                    speaker: 'Narrator',
                    text: 'Invalidated narrator audio should disappear.',
                    instruct: '',
                    status: 'pending',
                    audio_path: null,
                    audio_validation: null,
                });

                assert.strictEqual(changed, true);
                assert.ok(!row.classList.contains('status-done'));
                assert.ok(!row.classList.contains('status-generating'));
                assert.strictEqual(row.__audioContainer.audio, null);
                assert.ok(row.__generateSlot.button, 'generate button should be visible again');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_handle_chunk_audio_error_retries_once_with_fresh_src(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const audio = createAudioNode('<audio src="/voicelines/clip.mp3?t=old" data-audio-path="voicelines/clip.mp3" data-audio-fingerprint="fp"></audio>');
                const originalSrc = audio.src;

                context.handleChunkAudioError(audio);
                const retriedSrc = audio.src;
                context.handleChunkAudioError(audio);

                assert.notStrictEqual(retriedSrc, originalSrc, 'retry should replace the src with a fresh cache token');
                assert.strictEqual(audio.loadCalls, 1, 'audio reload should only happen once');
                assert.strictEqual(audio.dataset.audioRetryCount, '1');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_play_sequence_waits_for_silence_rows_between_audio_clips(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const timers = [];
                context.setTimeout = (fn, ms) => {
                    timers.push({ fn, ms });
                    return `timer-${timers.length}`;
                };
                context.clearTimeout = () => {};

                const button = context.document.getElementById('btn-play-seq');
                button.classList.add('btn-primary');

                function makeRow(id, classNames = []) {
                    const row = createGenericElement();
                    row.dataset = { id: String(id) };
                    classNames.forEach(name => row.classList.add(name));
                    row.scrollIntoView = () => {};
                    return row;
                }

                function makeAudio(row, src) {
                    const audio = createGenericElement();
                    audio.dataset = { id: row.dataset.id };
                    audio.src = src;
                    audio.getAttribute = (name) => name === 'src' ? audio.src : null;
                    audio.closest = (selector) => selector === 'tr' ? row : null;
                    audio.playCalls = 0;
                    audio.play = () => {
                        audio.playCalls += 1;
                        return Promise.resolve();
                    };
                    return audio;
                }

                const firstRow = makeRow(1);
                const silenceRow = makeRow(2, ['chunk-silence-row']);
                const secondRow = makeRow(3);
                const firstAudio = makeAudio(firstRow, '/voicelines/one.mp3');
                const secondAudio = makeAudio(secondRow, '/voicelines/two.mp3');
                const silenceInput = { value: '1.7' };

                firstRow.querySelector = (selector) => selector === 'audio.chunk-audio' ? firstAudio : null;
                silenceRow.querySelector = (selector) => selector === 'input' || selector === 'input[type="number"]' ? silenceInput : null;
                secondRow.querySelector = (selector) => selector === 'audio.chunk-audio' ? secondAudio : null;

                const rows = [firstRow, silenceRow, secondRow];
                context.document.querySelectorAll = (selector) => {
                    if (selector === '#chunks-table-body tr[data-id]') return rows;
                    if (selector === '.chunk-audio') return [firstAudio, secondAudio];
                    if (selector === 'audio') return [firstAudio, secondAudio];
                    if (selector === 'tr') return rows;
                    return [];
                };

                await context.playSequence();

                assert.strictEqual(firstAudio.playCalls, 1, 'first clip should start immediately');
                assert.strictEqual(secondAudio.playCalls, 0, 'second clip should wait behind the silence row');

                firstAudio.onended();

                assert.strictEqual(timers.length, 1, 'sequence should schedule a silence wait after the first clip');
                assert.strictEqual(timers[0].ms, 1700);
                assert.ok(silenceRow.classList.contains('table-primary'), 'silence row should receive playback highlight');
                assert.strictEqual(secondAudio.playCalls, 0, 'second clip should not play before the silence timer fires');

                timers[0].fn();

                assert.strictEqual(secondAudio.playCalls, 1, 'second clip should play after the silence duration elapses');
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_play_sequence_uses_export_gaps_between_adjacent_audio_clips(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.__editorTabTestHooks.setCachedChunks([
                    { id: 1, speaker: 'A', paragraph_id: 'p1' },
                    { id: 2, speaker: 'A', paragraph_id: 'p1' },
                    { id: 3, speaker: 'B', paragraph_id: 'p1' },
                    { id: 4, speaker: 'B', paragraph_id: 'p2' },
                ]);
                context.API.get = async (url) => {
                    assert.strictEqual(url, '/api/config');
                    return {
                        export: {
                            silence_same_speaker_ms: 111,
                            silence_between_speakers_ms: 222,
                            silence_paragraph_ms: 333,
                        },
                    };
                };

                const timers = [];
                context.setTimeout = (fn, ms) => {
                    timers.push({ fn, ms });
                    return `timer-${timers.length}`;
                };
                context.clearTimeout = () => {};

                const button = context.document.getElementById('btn-play-seq');
                button.classList.add('btn-primary');

                function makeRow(id) {
                    const row = createGenericElement();
                    row.dataset = { id: String(id) };
                    row.scrollIntoView = () => {};
                    return row;
                }

                function makeAudio(row, src) {
                    const audio = createGenericElement();
                    audio.dataset = { id: row.dataset.id };
                    audio.src = src;
                    audio.getAttribute = (name) => name === 'src' ? audio.src : null;
                    audio.closest = (selector) => selector === 'tr' ? row : null;
                    audio.playCalls = 0;
                    audio.play = () => {
                        audio.playCalls += 1;
                        return Promise.resolve();
                    };
                    return audio;
                }

                const rows = [makeRow(1), makeRow(2), makeRow(3), makeRow(4)];
                const audios = rows.map((row) => makeAudio(row, `/voicelines/${row.dataset.id}.mp3`));
                rows.forEach((row, index) => {
                    row.querySelector = (selector) => selector === 'audio.chunk-audio' ? audios[index] : null;
                });

                context.document.querySelectorAll = (selector) => {
                    if (selector === '#chunks-table-body tr[data-id]') return rows;
                    if (selector === '.chunk-audio') return audios;
                    if (selector === 'audio') return audios;
                    if (selector === 'tr') return rows;
                    return [];
                };

                await context.playSequence();

                assert.strictEqual(audios[0].playCalls, 1);
                assert.strictEqual(audios[1].playCalls, 0);

                audios[0].onended();
                assert.strictEqual(timers.length, 1);
                assert.strictEqual(timers[0].ms, 111, 'same-speaker clips in the same paragraph use same-speaker export gap');
                timers.shift().fn();
                assert.strictEqual(audios[1].playCalls, 1);

                audios[1].onended();
                assert.strictEqual(timers.length, 1);
                assert.strictEqual(timers[0].ms, 222, 'speaker changes in the same paragraph use speaker-change export gap');
                timers.shift().fn();
                assert.strictEqual(audios[2].playCalls, 1);

                audios[2].onended();
                assert.strictEqual(timers.length, 1);
                assert.strictEqual(timers[0].ms, 333, 'paragraph changes take precedence over speaker matching');
                timers.shift().fn();
                assert.strictEqual(audios[3].playCalls, 1);
            })().catch((error) => {
                console.error(error);
                process.exit(1);
            });
            """
        )

    def test_audio_queue_poll_preserves_existing_audio_element_while_generation_runs(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                const initialChunk = {
                    id: 8,
                    speaker: 'Narrator',
                    text: 'Existing audio should remain usable during active generation.',
                    instruct: '',
                    status: 'done',
                    audio_path: 'voicelines/existing.mp3',
                    audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                };
                const row = context.__createChunkRow(initialChunk);
                row.__audioContainer.audio = createAudioNode('<audio src="/voicelines/existing.mp3?t=stable" data-audio-path="voicelines/existing.mp3" data-audio-fingerprint="voicelines/existing.mp3|10|1|done"></audio>');
                row.__audioContainer.audio.__container = row.__audioContainer;
                const preservedAudio = row.__audioContainer.audio;
                context.__rows.set('8', row);

                context.API.get = async (url) => {
                    if (url === '/api/status/audio') {
                        return {
                            running: true,
                            queue: [],
                            current_job: { total_chunks: 2 },
                            metrics: { processed_clips: 1 },
                        };
                    }
                    if (url === '/api/chunks/view') {
                        return [initialChunk];
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                vm.createContext(context);
                vm.runInContext(source, context);
                context.__editorTabTestHooks.setCachedChunks([initialChunk]);

                await context.pollAudioQueueOnce();
                await flushTicks();

                assert.strictEqual(row.__audioContainer.audio, preservedAudio, 'running generation should not replace an unchanged audio element');
                assert.strictEqual(row.__audioContainer.audio.src, '/voicelines/existing.mp3?t=stable');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_load_chunks_full_redraw_preserves_existing_editor_audio_element(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const initialChunk = {
                    id: 9,
                    speaker: 'Narrator',
                    text: 'Existing audio should survive a full table redraw.',
                    instruct: '',
                    status: 'done',
                    audio_path: 'voicelines/preserved.mp3',
                    audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 },
                };

                const row = context.__createChunkRow(initialChunk);
                row.__audioContainer.audio = createAudioNode('<audio src="/voicelines/preserved.mp3?t=voicelines%2Fpreserved.mp3%7C10%7C1%7Cdone" data-audio-path="voicelines/preserved.mp3" data-audio-fingerprint="voicelines/preserved.mp3|10|1|done"></audio>');
                row.__audioContainer.audio.__container = row.__audioContainer;
                const preservedAudio = row.__audioContainer.audio;

                const staleRow = context.__createChunkRow({ id: 99, speaker: 'Narrator', text: 'stale', instruct: '', status: 'pending' });
                context.__rows.set('9', row);
                context.__rows.set('99', staleRow);
                const tbody = context.document.getElementById('chunks-table-body');
                tbody.children = [row, staleRow];

                context.API.get = async (url) => {
                    assert.strictEqual(url, '/api/chunks/view');
                    return [initialChunk];
                };

                context.__editorTabTestHooks.setCachedChunks([initialChunk]);

                await context.loadChunks(false);
                await flushTicks();

                const redrawnRow = context.__rows.get('9');
                assert.ok(redrawnRow, 'expected redrawn row to exist');
                assert.strictEqual(redrawnRow.__audioContainer.audio, preservedAudio, 'full redraw should reuse the existing audio element');
                assert.strictEqual(redrawnRow.__audioContainer.audio.src, '/voicelines/preserved.mp3?t=voicelines%2Fpreserved.mp3%7C10%7C1%7Cdone');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_load_chunks_full_redraw_uses_chapter_scope_when_cached_project_is_available(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                context.__editorTabTestHooks.setSelectedEditorChapter('Chapter 1');
                context.__editorTabTestHooks.setCachedChunks([
                    { id: 1, uid: 'chunk-1', speaker: 'Narrator', chapter: 'Chapter 1', text: 'one', instruct: '', status: 'done', audio_path: 'voicelines/one.mp3', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                    { id: 2, uid: 'chunk-2', speaker: 'Narrator', chapter: 'Chapter 2', text: 'two', instruct: '', status: 'pending', audio_path: null, audio_validation: null },
                ]);

                const requestedUrls = [];
                context.API.get = async (url) => {
                    requestedUrls.push(url);
                    if (url === '/api/chunks/chapters') {
                        return { chapters: [{ chapter: 'Chapter 1', count: 1 }, { chapter: 'Chapter 2', count: 1 }] };
                    }
                    if (url === '/api/chunks/view?chapter=Chapter%201') {
                        return [
                            { id: 1, uid: 'chunk-1', speaker: 'Narrator', chapter: 'Chapter 1', text: 'one', instruct: '', status: 'done', audio_path: 'voicelines/one.mp3', audio_validation: { file_size_bytes: 10, actual_duration_sec: 1.0 } },
                        ];
                    }
                    if (url === '/api/voices') {
                        return {};
                    }
                    throw new Error(`Unexpected GET ${url}`);
                };

                await context.loadChunks(true);
                await flushTicks();

                assert.ok(requestedUrls.includes('/api/chunks/chapters'));
                assert.ok(requestedUrls.includes('/api/chunks/view?chapter=Chapter%201'));
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )

    def test_cancel_render_refreshes_chunks_and_hides_cancel_button_when_audio_is_idle(self):
        self._run_node_test(
            """
            (async () => {
                const context = createContext();
                vm.createContext(context);
                vm.runInContext(source, context);

                const cancelBtn = context.document.getElementById('btn-cancel-render');
                cancelBtn.style.display = 'inline-block';
                let loadChunksCalls = 0;
                context.__editorTabTestHooks.setLoadChunks(async () => {
                    loadChunksCalls += 1;
                });

                context.API.post = async (url) => {
                    assert.strictEqual(url, '/api/cancel_audio');
                    return { status: 'cancelling' };
                };
                context.API.get = async (url) => {
                    assert.strictEqual(url, '/api/status/audio');
                    return {
                        running: false,
                        queue: [],
                        current_job: null,
                        metrics: {},
                    };
                };

                await context.cancelRender();
                await flushTicks();

                assert.strictEqual(loadChunksCalls, 1, 'cancel should immediately reload chunk state');
                assert.strictEqual(cancelBtn.style.display, 'none', 'cancel button should hide once audio work is idle');
            })().catch((error) => {{
                console.error(error);
                process.exit(1);
            }});
            """
        )
