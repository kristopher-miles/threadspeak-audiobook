"""ProjectManager behavior tests grouped by domain."""

import json
import os
import tempfile
import time
import threading
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from types import SimpleNamespace
from pydub import AudioSegment

import project as project_module
import project_core.mixins.chunk_store as chunk_store_module
from project import ProjectManager

class MergeAudioTests(unittest.TestCase):
    # Policy: do not stub MP3 concat in merge/export integration tests.
    # These tests must exercise the real encoder path so they fail if libmp3lame is unavailable.
    NEVER_STUB_MP3_CONCAT_NOTE = (
        "Do not stub ProjectManager._export_concat_mp3 in merge/optimized export tests. "
        "These tests intentionally depend on real MP3 encoder availability."
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)
        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)
        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.manager.shutdown_script_store(flush=True)
        self.temp_dir.cleanup()

    def _write_wav(self, relative_path, duration_seconds):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def _write_tone_with_silence(self, relative_path, lead_s=0.25, tone_s=0.3, tail_s=0.25, hz=440):
        full_path = os.path.join(self.root_dir, relative_path)
        sample_rate = 24000
        lead = np.zeros(int(sample_rate * lead_s), dtype=np.float32)
        tail = np.zeros(int(sample_rate * tail_s), dtype=np.float32)
        t = np.arange(int(sample_rate * tone_s), dtype=np.float32) / sample_rate
        tone = (0.35 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        samples = np.concatenate([lead, tone, tail]).astype(np.float32)
        sf.write(full_path, samples, sample_rate)
        return full_path

    def _assert_real_mp3_concat_path(self):
        bound = self.manager._export_concat_mp3
        self.assertTrue(hasattr(bound, "__func__"), self.NEVER_STUB_MP3_CONCAT_NOTE)
        self.assertIs(bound.__func__, ProjectManager._export_concat_mp3, self.NEVER_STUB_MP3_CONCAT_NOTE)

    def test_merge_audio_reports_progress_and_creates_mp3(self):
        self._assert_real_mp3_concat_path()
        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ])

        progress = []
        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda path, export_config=None: (True, path)
        try:
            success, output_filename = self.manager.merge_audio(progress_callback=progress.append)
        finally:
            self.manager._normalize_audio_file = original_normalize

        self.assertTrue(
            success,
            "Merge audio test requires real MP3 concat success. "
            + self.NEVER_STUB_MP3_CONCAT_NOTE,
        )
        self.assertEqual(output_filename, "cloned_audiobook.mp3")
        self.assertTrue(os.path.exists(os.path.join(self.root_dir, output_filename)))
        self.assertGreater(os.path.getsize(os.path.join(self.root_dir, output_filename)), 0)
        stages = [item.get("stage") for item in progress]
        self.assertIn("preparing", stages)
        self.assertIn("assembling", stages)
        self.assertIn("exporting", stages)
        self.assertIn("normalizing", stages)
        self.assertEqual(stages[-1], "complete")

    def test_merge_audio_fails_when_normalization_fails(self):
        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ])
        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda *args, **kwargs: (False, "simulated loudnorm failure")
        try:
            success, message = self.manager.merge_audio()
        finally:
            self.manager._normalize_audio_file = original_normalize

        self.assertFalse(success)
        self.assertIn("Audio normalization failed", message)

    def test_merge_audio_reuses_cached_workspace_artifacts(self):
        self._assert_real_mp3_concat_path()
        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ])

        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda path, export_config=None: (True, path)
        try:
            first_success, _ = self.manager.merge_audio()
        finally:
            self.manager._normalize_audio_file = original_normalize
        self.assertTrue(first_success)

        with patch.object(self.manager, "_export_concat_mp3", side_effect=AssertionError("concat should be reused")):
            with patch.object(self.manager, "_normalize_audio_file", side_effect=AssertionError("normalize should be reused")):
                second_success, second_output = self.manager.merge_audio()
        self.assertTrue(second_success)
        self.assertEqual(second_output, "cloned_audiobook.mp3")

        wip_root = os.path.join(self.root_dir, "_wip")
        self.assertTrue(os.path.isdir(wip_root))
        fingerprints = [name for name in os.listdir(wip_root) if os.path.isdir(os.path.join(wip_root, name))]
        self.assertGreaterEqual(len(fingerprints), 1)
        manifest_path = os.path.join(wip_root, fingerprints[0], "stage_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual((manifest.get("stages") or {}).get("normalize", {}).get("status"), "complete")

    def test_optimized_export_creates_ordered_zip_parts(self):
        self._assert_real_mp3_concat_path()
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "My Great Book.txt")}, f)

        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip3.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 2,
                "speaker": "Narrator",
                "text": "Three.",
                "instruct": "",
                "chapter": "Chapter 3",
                "status": "done",
                "audio_path": "voicelines/clip3.wav",
            },
        ])

        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda path, export_config=None: (True, path)
        try:
            success, output_filename = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)
        finally:
            self.manager._normalize_audio_file = original_normalize

        self.assertTrue(
            success,
            "Optimized export ordering test requires real MP3 concat success. "
            + self.NEVER_STUB_MP3_CONCAT_NOTE,
        )
        self.assertEqual(output_filename, "optimized_audiobook.zip")
        zip_path = os.path.join(self.root_dir, output_filename)
        self.assertTrue(os.path.exists(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            stems = [os.path.splitext(name)[0] for name in names]
            self.assertGreaterEqual(len(stems), 2)
            expected_stems = [f"my-great-book-{index:02d}" for index in range(1, len(stems) + 1)]
            self.assertEqual(stems, expected_stems)
            self.assertTrue(all(name.endswith((".mp3", ".wav")) for name in names))

    def test_optimized_export_reuses_cached_workspace_artifacts(self):
        self._assert_real_mp3_concat_path()
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "Cache Book.txt")}, f)

        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip3.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 2,
                "speaker": "Narrator",
                "text": "Three.",
                "instruct": "",
                "chapter": "Chapter 3",
                "status": "done",
                "audio_path": "voicelines/clip3.wav",
            },
        ])

        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda path, export_config=None: (True, path)
        try:
            first_success, _ = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)
        finally:
            self.manager._normalize_audio_file = original_normalize
        self.assertTrue(first_success)

        with patch.object(self.manager, "_export_concat_mp3", side_effect=AssertionError("concat should be reused")):
            with patch.object(self.manager, "_normalize_audio_file", side_effect=AssertionError("normalize should be reused")):
                second_success, second_output = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)
        self.assertTrue(second_success)
        self.assertEqual(second_output, "optimized_audiobook.zip")

    def test_optimized_export_surfaces_mp3_failure_without_wav_fallback(self):
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "Fallback Book.txt")}, f)

        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip3.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 2,
                "speaker": "Narrator",
                "text": "Three.",
                "instruct": "",
                "chapter": "Chapter 3",
                "status": "done",
                "audio_path": "voicelines/clip3.wav",
            },
        ])

        calls = []
        original_run = self.manager._run_ffmpeg_concat

        def fake_run_ffmpeg_concat(concat_path, output_path, codec_args, progress_tick=None):
            calls.append((os.path.basename(output_path), tuple(codec_args)))
            if tuple(codec_args) == ("-c:a", "libmp3lame", "-q:a", "2"):
                return False, "simulated mp3 failure"
            raise AssertionError("WAV fallback should not be attempted when MP3 concat fails")

        self.manager._run_ffmpeg_concat = fake_run_ffmpeg_concat
        original_normalize = self.manager._normalize_audio_file
        self.manager._normalize_audio_file = lambda path, export_config=None: (True, path)
        try:
            success, output_filename = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)
        finally:
            self.manager._run_ffmpeg_concat = original_run
            self.manager._normalize_audio_file = original_normalize

        self.assertFalse(success)
        self.assertIn("MP3 concat export failed", output_filename)
        zip_path = os.path.join(self.root_dir, "optimized_audiobook.zip")
        self.assertFalse(os.path.exists(zip_path))
        self.assertTrue(any(codec_args == ("-c:a", "libmp3lame", "-q:a", "2") for _, codec_args in calls))
        self.assertFalse(any(codec_args == ("-c:a", "pcm_s16le") for _, codec_args in calls))

    def test_optimized_export_normalizes_each_part(self):
        self._assert_real_mp3_concat_path()
        with open(os.path.join(self.root_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"input_file_path": os.path.join(self.root_dir, "Normalize Parts Book.txt")}, f)

        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip2.wav", duration_seconds=0.5)
        self._write_wav("voicelines/clip3.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Two.",
                "instruct": "",
                "chapter": "Chapter 2",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 2,
                "speaker": "Narrator",
                "text": "Three.",
                "instruct": "",
                "chapter": "Chapter 3",
                "status": "done",
                "audio_path": "voicelines/clip3.wav",
            },
        ])
        normalize_calls = []
        original_normalize = self.manager._normalize_audio_file

        def fake_normalize(path, export_config=None):
            normalize_calls.append(os.path.basename(path))
            return True, path

        self.manager._normalize_audio_file = fake_normalize
        try:
            success, output_filename = self.manager.export_optimized_mp3_zip(max_part_seconds=1.4)
        finally:
            self.manager._normalize_audio_file = original_normalize

        self.assertTrue(
            success,
            "Optimized export normalization test requires real MP3 concat success. "
            + self.NEVER_STUB_MP3_CONCAT_NOTE,
        )
        self.assertEqual(output_filename, "optimized_audiobook.zip")
        self.assertGreaterEqual(len(normalize_calls), 2)
        self.assertTrue(all(name.startswith("normalize-parts-book-") for name in normalize_calls))

    def test_merge_m4b_normalizes_temp_audio_before_encode(self):
        self._write_wav("voicelines/clip1.wav", duration_seconds=0.5)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "Chapter one starts here.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
        ])

        normalize_calls = []
        original_normalize = self.manager._normalize_audio_file
        original_run = project_module.subprocess.run

        def fake_normalize(path, export_config=None):
            normalize_calls.append(path)
            return True, path

        class DummyResult:
            returncode = 0
            stderr = ""

        self.manager._normalize_audio_file = fake_normalize
        project_module.subprocess.run = lambda *args, **kwargs: DummyResult()
        try:
            success, output_filename = self.manager.merge_m4b()
        finally:
            self.manager._normalize_audio_file = original_normalize
            project_module.subprocess.run = original_run

        self.assertTrue(success)
        self.assertEqual(output_filename, "audiobook.m4b")
        self.assertEqual(len(normalize_calls), 1)
        self.assertTrue(normalize_calls[0].endswith("temp_m4b_combined.wav"))

    def test_trim_cache_persists_and_is_reused(self):
        original_path = self._write_tone_with_silence("voicelines/trim_me.wav")
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "Trim me.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/trim_me.wav",
            },
        ])

        export_config = SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-45.0,
            trim_min_silence_len_ms=80,
            trim_keep_padding_ms=20,
        )

        timeline_first = self.manager._collect_merge_timeline(export_config=export_config)
        self.assertEqual(len(timeline_first), 1)
        trimmed_path = timeline_first[0]["full_path"]
        self.assertIn(f"voicelines{os.sep}.trim_cache{os.sep}", trimmed_path)
        self.assertTrue(os.path.exists(trimmed_path))

        original_ms = len(AudioSegment.from_file(original_path))
        trimmed_ms = len(AudioSegment.from_file(trimmed_path))
        self.assertLess(trimmed_ms, original_ms)

        mtime_first = os.path.getmtime(trimmed_path)
        timeline_second = self.manager._collect_merge_timeline(export_config=export_config)
        self.assertEqual(trimmed_path, timeline_second[0]["full_path"])
        self.assertEqual(mtime_first, os.path.getmtime(trimmed_path))

    def test_trimmed_cache_only_cuts_boundaries_without_altering_samples(self):
        self._write_tone_with_silence("voicelines/declick.wav", lead_s=0.12, tone_s=0.25, tail_s=0.12, hz=550)
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "Trim edge fade check.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/declick.wav",
            },
        ])

        export_config = SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-45.0,
            trim_min_silence_len_ms=50,
            trim_keep_padding_ms=0,
        )
        timeline = self.manager._collect_merge_timeline(export_config=export_config)
        trimmed = AudioSegment.from_file(timeline[0]["full_path"])
        source = AudioSegment.from_file(os.path.join(self.root_dir, "voicelines/declick.wav"))
        expected, _, _, changed = self.manager._trim_audio_segment_boundaries(
            source,
            self.manager._resolve_trim_config(export_config),
        )
        self.assertTrue(changed)

        # Round-trip expected through MP3 to match the lossy encoding used in the trim cache
        import io as _io
        mp3_buf = _io.BytesIO()
        expected.export(mp3_buf, format="mp3", bitrate="128k")
        mp3_buf.seek(0)
        expected_mp3 = AudioSegment.from_file(mp3_buf, format="mp3")
        expected_samples = np.array(expected_mp3.get_array_of_samples(), dtype=np.int64)
        actual_samples = np.array(trimmed.get_array_of_samples(), dtype=np.int64)
        self.assertEqual(expected_samples.shape, actual_samples.shape)
        self.assertEqual(int(np.max(np.abs(expected_samples - actual_samples))), 0)

    def test_trim_disabled_uses_original_audio_paths(self):
        original_path = self._write_tone_with_silence("voicelines/no_trim.wav")
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "No trim.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/no_trim.wav",
            },
        ])

        export_config = SimpleNamespace(trim_clip_silence_enabled=False)
        timeline = self.manager._collect_merge_timeline(export_config=export_config)
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0]["full_path"], original_path)

    def test_trim_disabled_via_config_ignores_existing_trim_cache(self):
        original_path = self._write_tone_with_silence("voicelines/no_trim_from_config.wav")
        self.manager.save_chunks([
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "No trim from config.",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/no_trim_from_config.wav",
            },
        ])

        enabled = SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-45.0,
            trim_min_silence_len_ms=80,
            trim_keep_padding_ms=20,
        )
        timeline_enabled = self.manager._collect_merge_timeline(export_config=enabled)
        self.assertIn(f"voicelines{os.sep}.trim_cache{os.sep}", timeline_enabled[0]["full_path"])

        with open(os.path.join(self.root_dir, "app", "config.json"), "w", encoding="utf-8") as f:
            json.dump({"export": {"trim_clip_silence_enabled": "false"}}, f)

        timeline_disabled = self.manager._collect_merge_timeline(export_config=None)
        self.assertEqual(timeline_disabled[0]["full_path"], original_path)

    def test_trim_real_title_clip_does_not_gain_loud_tail(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sample_rel = "voicelines/voiceline_e68a7990dac94b1db5679449c69ccfb8_narrator.mp3"
        sample_full = os.path.join(repo_root, sample_rel)
        if not os.path.exists(sample_full):
            self.skipTest(f"Real sample clip not found: {sample_rel}")

        manager = ProjectManager(repo_root)
        trim_cfg = manager._resolve_trim_config(SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-50.0,
            trim_min_silence_len_ms=80,
            trim_keep_padding_ms=0,
        ))

        # Force a recompute so this test verifies the current trim behavior.
        cache_key = manager._build_trim_cache_key(sample_full, trim_cfg)
        cache_path = os.path.join(manager._trim_cache_dir(), f"{cache_key}.wav")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        trimmed_path, trim_info = manager._resolve_export_audio_path(sample_full, trim_cfg)
        self.assertTrue(trim_info["trimmed"], "Expected sample clip to be trimmed for this regression test")
        self.assertGreater(trim_info["tail_ms"], 0, "Expected tail silence to be removed in this regression test")

        original = AudioSegment.from_file(sample_full)
        trimmed = AudioSegment.from_file(trimmed_path)

        original_samples = np.array(original.get_array_of_samples(), dtype=np.float64)
        trimmed_samples = np.array(trimmed.get_array_of_samples(), dtype=np.float64)
        if original.channels > 1:
            original_samples = original_samples.reshape((-1, original.channels)).mean(axis=1)
        if trimmed.channels > 1:
            trimmed_samples = trimmed_samples.reshape((-1, trimmed.channels)).mean(axis=1)

        original_terminal_level = float(abs(original_samples[-1]))
        trimmed_terminal_level = float(abs(trimmed_samples[-1]))

        # Regression guard: the final sample level of the trimmed clip should stay
        # near the true final level of the original clip (no hard-edge endpoint pop).
        self.assertLessEqual(
            trimmed_terminal_level,
            max(20.0, original_terminal_level * 8.0),
            (
                f"Trimmed terminal sample too loud for real clip "
                f"(original terminal {original_terminal_level:.2f}, trimmed terminal {trimmed_terminal_level:.2f})"
            ),
        )

    def test_trim_guard_discards_longer_than_original_result(self):
        original_path = self._write_tone_with_silence("voicelines/trim_guard.wav", lead_s=0.05, tone_s=0.20, tail_s=0.05)
        source = AudioSegment.from_file(original_path)

        injected_longer = source + AudioSegment.silent(duration=50, frame_rate=source.frame_rate)
        trim_cfg = self.manager._resolve_trim_config(SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-45.0,
            trim_min_silence_len_ms=50,
            trim_keep_padding_ms=0,
        ))
        cache_key = self.manager._build_trim_cache_key(original_path, trim_cfg)
        cache_path = os.path.join(self.manager._trim_cache_dir(), f"{cache_key}.mp3")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        with patch.object(self.manager, "_trim_audio_segment_boundaries", return_value=(injected_longer, 0, 0, True)):
            with patch("builtins.print") as print_mock:
                resolved_path, info = self.manager._resolve_export_audio_path(original_path, trim_cfg)

        self.assertEqual(resolved_path, cache_path)
        self.assertTrue(os.path.exists(resolved_path))
        resolved_segment = AudioSegment.from_file(resolved_path)
        self.assertEqual(len(resolved_segment), len(source))
        self.assertFalse(info["trimmed"])
        self.assertEqual(info["lead_ms"], 0)
        self.assertEqual(info["tail_ms"], 0)
        self.assertGreaterEqual(print_mock.call_count, 1)
        printed = " ".join(str(arg) for arg in print_mock.call_args[0])
        self.assertIn("Trim result longer than original", printed)

    def test_trim_real_clip_refines_start_edge_for_assembly(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sample_rel = "voicelines/voiceline_f7a3ab93afc94813b75c02d2240cd6b3_narrator.mp3"
        sample_full = os.path.join(repo_root, sample_rel)
        if not os.path.exists(sample_full):
            self.skipTest(f"Real sample clip not found: {sample_rel}")

        manager = ProjectManager(repo_root)
        trim_cfg = manager._resolve_trim_config(SimpleNamespace(
            trim_clip_silence_enabled=True,
            trim_silence_threshold_dbfs=-50.0,
            trim_min_silence_len_ms=80,
            trim_keep_padding_ms=0,
        ))
        cache_key = manager._build_trim_cache_key(sample_full, trim_cfg)
        cache_path = os.path.join(manager._trim_cache_dir(), f"{cache_key}.wav")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        source = AudioSegment.from_file(sample_full)
        source_samples = np.array(source.get_array_of_samples(), dtype=np.int64)
        if source.channels > 1:
            source_samples = source_samples.reshape((-1, source.channels)).mean(axis=1).astype(np.int64)
        source_start = abs(int(source_samples[0]))

        trimmed_path, _trim_info = manager._resolve_export_audio_path(sample_full, trim_cfg)
        trimmed = AudioSegment.from_file(trimmed_path)
        trimmed_samples = np.array(trimmed.get_array_of_samples(), dtype=np.int64)
        if trimmed.channels > 1:
            trimmed_samples = trimmed_samples.reshape((-1, trimmed.channels)).mean(axis=1).astype(np.int64)
        trimmed_start = abs(int(trimmed_samples[0]))

        self.assertLess(
            trimmed_start,
            source_start,
            f"Expected trim edge refinement to lower start sample level (source={source_start}, trimmed={trimmed_start})",
        )
        self.assertLessEqual(
            trimmed_start,
            64,
            f"Expected trimmed start sample to be near zero for smoother assembly (got {trimmed_start})",
        )

    def test_repair_legacy_chunk_order_rewrites_chunks_from_editor_order(self):
        original = [
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "First",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip1.wav",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Second",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
        ]
        self.manager.save_chunks(original)

        repaired = self.manager.repair_legacy_chunk_order([
            {
                "id": 99,
                "speaker": "Narrator",
                "text": "Second",
                "instruct": "",
                "chapter": "Chapter 1",
                "status": "done",
                "audio_path": "voicelines/clip2.wav",
            },
            {
                "id": 42,
                "speaker": "Narrator",
                "text": "Replacement First",
                "instruct": "calm",
                "chapter": "Chapter 1",
                "status": "pending",
                "audio_path": None,
            },
        ])

        self.assertEqual([chunk["id"] for chunk in repaired], [0, 1])
        self.assertEqual([chunk["text"] for chunk in repaired], ["Second", "Replacement First"])
        persisted = self.manager.load_chunks()
        self.assertEqual([chunk["text"] for chunk in persisted], ["Second", "Replacement First"])

    def test_load_chunks_backfills_stable_uids_for_legacy_rows(self):
        legacy = [
            {"id": 0, "speaker": "Narrator", "text": "One", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 1, "speaker": "Narrator", "text": "Two", "instruct": "", "status": "pending", "audio_path": None},
        ]
        with open(os.path.join(self.root_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(legacy, f, indent=2)

        loaded = self.manager.load_chunks()

        self.assertEqual(loaded, [])

    def test_delete_and_restore_use_stable_uid(self):
        self.manager.save_chunks([
            {"id": 0, "speaker": "Narrator", "text": "One", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 1, "speaker": "Narrator", "text": "Two", "instruct": "", "status": "pending", "audio_path": None},
            {"id": 2, "speaker": "Narrator", "text": "Three", "instruct": "", "status": "pending", "audio_path": None},
        ])
        initial = self.manager.load_chunks()
        deleted_uid = initial[1]["uid"]
        previous_uid = initial[0]["uid"]

        deleted, remaining, restore_after_uid = self.manager.delete_chunk(deleted_uid)
        self.assertEqual(deleted["text"], "Two")
        self.assertEqual(restore_after_uid, previous_uid)
        self.assertEqual([chunk["text"] for chunk in remaining], ["One", "Three"])

        restored = self.manager.restore_chunk(0, deleted, after_uid=restore_after_uid)
        self.assertEqual([chunk["text"] for chunk in restored], ["One", "Two", "Three"])

class DecomposeLongSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.manager.shutdown_script_store(flush=True)
        self.temp_dir.cleanup()

    def test_recursively_splits_unsynthesized_segments_until_within_limit(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": (
                "One two three four five six seven eight. "
                "Nine ten eleven twelve thirteen fourteen fifteen sixteen. "
                "Seventeen eighteen nineteen twenty twenty one twenty two twenty three twenty four. "
                "Twenty five twenty six twenty seven twenty eight twenty nine thirty thirty one thirty two."
            ),
            "instruct": "Warm and steady",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=15)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 3)
        self.assertEqual(len(updated), 4)
        self.assertTrue(all(self.manager._count_words(chunk["text"]) <= 15 for chunk in updated))
        self.assertTrue(all(chunk["speaker"] == "Narrator" for chunk in updated))
        self.assertTrue(all(chunk["instruct"] == "Warm and steady" for chunk in updated))
        self.assertEqual([chunk["id"] for chunk in updated], [0, 1, 2, 3])

    def test_does_not_split_segment_that_already_has_audio(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": (
                "One two three four five six seven eight nine ten. "
                "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty."
            ),
            "instruct": "",
            "status": "done",
            "audio_path": "voicelines/existing.wav",
            "audio_validation": {"is_valid": True},
            "auto_regen_count": 0,
            "chapter": "Chapter 1",
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=5)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["audio_path"], "voicelines/existing.wav")

    def test_does_not_split_without_sentence_boundaries(self):
        chunks = [{
            "id": 0,
            "speaker": "Narrator",
            "text": " ".join(f"word{i}" for i in range(30)),
            "instruct": "",
            "status": "pending",
            "audio_path": None,
            "audio_validation": None,
            "auto_regen_count": 0,
        }]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(max_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["text"], chunks[0]["text"])

    def test_limits_to_requested_chapter(self):
        chunks = [
            {
                "id": 0,
                "speaker": "Narrator",
                "text": "One two three four five six. Seven eight nine ten eleven twelve.",
                "instruct": "Calm",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 1,
                "speaker": "Narrator",
                "text": "Alpha beta gamma delta epsilon zeta. Eta theta iota kappa lambda mu.",
                "instruct": "Calm",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ]
        self.manager.save_chunks(chunks)

        result = self.manager.decompose_long_segments(chapter="Chapter 2", max_words=6)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 3)
        self.assertEqual(updated[0]["chapter"], "Chapter 1")
        self.assertEqual(updated[0]["text"], chunks[0]["text"])
        self.assertEqual(updated[1]["chapter"], "Chapter 2")
        self.assertEqual(updated[2]["chapter"], "Chapter 2")

class MergeOrphanSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        os.makedirs(os.path.join(self.root_dir, "voicelines"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "app"), exist_ok=True)

        with open(os.path.join(self.root_dir, "annotated_script.json"), "w", encoding="utf-8") as f:
            json.dump({"entries": [], "dictionary": []}, f)

        self.manager = ProjectManager(self.root_dir)

    def tearDown(self):
        self.manager.shutdown_script_store(flush=True)
        self.temp_dir.cleanup()

    def _chapter_intro_chunks(self, chapter, start_id=0, speaker="Narrator"):
        chunks = []
        for offset in range(5):
            chunks.append({
                "id": start_id + offset,
                "speaker": speaker,
                "text": f"Protected intro line {offset}.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": chapter,
            })
        return chunks

    def test_merges_short_segment_with_adjacent_same_speaker_until_threshold(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Hello there friend.",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/a.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "We meet again tonight.",
                "instruct": "Gentle",
                "status": "done",
                "audio_path": "voicelines/b.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 7,
                "speaker": "Narrator",
                "text": "Stay close now.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 2)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["text"], "Hello there friend. We meet again tonight. Stay close now.")
        self.assertEqual(updated[5]["instruct"], "Gentle")
        self.assertEqual(updated[5]["status"], "pending")
        self.assertIsNone(updated[5]["audio_path"])
        self.assertIsNone(updated[5]["audio_validation"])

    def test_prefers_first_non_empty_instruction_and_can_merge_forward(self):
        chunks = self._chapter_intro_chunks("Chapter 1", speaker="Alice")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Alice",
                "text": "Hi there.",
                "instruct": "Bright",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Alice",
                "text": "Come inside now please.",
                "instruct": "Serious",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["instruct"], "Bright")

    def test_limits_merge_to_requested_chapter(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend(self._chapter_intro_chunks("Chapter 2", start_id=5))
        chunks.extend([
            {
                "id": 10,
                "speaker": "Narrator",
                "text": "One two three.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 11,
                "speaker": "Narrator",
                "text": "Four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
            {
                "id": 12,
                "speaker": "Narrator",
                "text": "Seven eight nine ten eleven.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(chapter="Chapter 2", min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 12)
        self.assertEqual(updated[10]["chapter"], "Chapter 1")
        self.assertEqual(updated[10]["text"], "One two three.")
        self.assertEqual(updated[11]["text"], "Four five six. Seven eight nine ten eleven.")

    def test_does_not_merge_across_chapter_boundaries_in_whole_project_mode(self):
        chunks = self._chapter_intro_chunks("Chapter 1")
        chunks.extend(self._chapter_intro_chunks("Chapter 2", start_id=5))
        chunks.extend([
            {
                "id": 10,
                "speaker": "Narrator",
                "text": "One two three.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 11,
                "speaker": "Narrator",
                "text": "Four five six.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 2",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 0)
        self.assertEqual(len(updated), 12)
        self.assertEqual(updated[10]["chapter"], "Chapter 1")
        self.assertEqual(updated[11]["chapter"], "Chapter 2")

    def test_preserves_exact_chapter_label_when_merging(self):
        chunks = self._chapter_intro_chunks("Chapter 12A")
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Alpha beta gamma.",
                "instruct": "",
                "status": "done",
                "audio_path": "voicelines/a.wav",
                "audio_validation": {"is_valid": True},
                "auto_regen_count": 0,
                "chapter": "Chapter 12A",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "Delta epsilon zeta eta theta iota.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 12A",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        self.assertEqual(updated[5]["chapter"], "Chapter 12A")
        self.assertIsNone(updated[5]["audio_path"])

    def test_skips_first_five_samples_of_each_chapter(self):
        chunks = []
        for i in range(5):
            chunks.append({
                "id": i,
                "speaker": "Narrator",
                "text": f"Short intro {i}.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            })
        chunks.extend([
            {
                "id": 5,
                "speaker": "Narrator",
                "text": "Tiny tail.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
            {
                "id": 6,
                "speaker": "Narrator",
                "text": "This should merge with the tiny tail now.",
                "instruct": "",
                "status": "pending",
                "audio_path": None,
                "audio_validation": None,
                "auto_regen_count": 0,
                "chapter": "Chapter 1",
            },
        ])
        self.manager.save_chunks(chunks)

        result = self.manager.merge_orphan_segments(min_words=10)
        updated = self.manager.load_chunks()

        self.assertEqual(result["changed"], 1)
        self.assertEqual(len(updated), 6)
        for i in range(5):
            self.assertEqual(updated[i]["text"], f"Short intro {i}.")
        self.assertEqual(updated[5]["text"], "Tiny tail. This should merge with the tiny tail now.")
