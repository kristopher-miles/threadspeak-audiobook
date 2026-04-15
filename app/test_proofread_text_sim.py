import json
import wave
from pathlib import Path

import pytest

from e2e_sim.proofread_text_sim import ProofreadTextSimProvider
from project import ProjectManager


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_silent_wav(path: Path, *, duration_seconds: float = 0.2, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(sample_rate * duration_seconds))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def _create_project_root(root: Path) -> None:
    (root / "voicelines").mkdir(parents=True, exist_ok=True)
    (root / "app").mkdir(parents=True, exist_ok=True)
    _write_json(root / "annotated_script.json", {"entries": [], "dictionary": []})


def test_proofread_text_provider_returns_fixture_hit(tmp_path: Path):
    fixture_path = tmp_path / "proofread_fixture.json"
    _write_json(
        fixture_path,
        {
            "strict": True,
            "fallback_mode": "chunk_text",
            "entries": [
                {
                    "audio_path": "voicelines/example.wav",
                    "transcript_text": "Fixture transcript line.",
                }
            ],
        },
    )

    provider = ProofreadTextSimProvider(str(fixture_path))
    result = provider.resolve_transcript("voicelines/example.wav")
    assert result == "Fixture transcript line."


def test_proofread_text_provider_uses_chunk_text_fallback(tmp_path: Path):
    fixture_path = tmp_path / "proofread_fixture.json"
    _write_json(
        fixture_path,
        {
            "strict": True,
            "fallback_mode": "chunk_text",
            "entries": [],
        },
    )

    provider = ProofreadTextSimProvider(str(fixture_path))
    result = provider.resolve_transcript("voicelines/missing.wav", fallback_text="Derived fallback transcript.")
    assert result == "Derived fallback transcript."


def test_proofread_text_provider_raises_on_missing_strict_entry(tmp_path: Path):
    fixture_path = tmp_path / "proofread_fixture.json"
    _write_json(
        fixture_path,
        {
            "strict": True,
            "fallback_mode": "fail",
            "entries": [],
        },
    )

    provider = ProofreadTextSimProvider(str(fixture_path))
    with pytest.raises(AssertionError):
        provider.resolve_transcript("voicelines/missing.wav")


def test_transcribe_audio_path_uses_proofread_fixture_without_real_asr(tmp_path: Path, monkeypatch):
    _create_project_root(tmp_path)
    audio_rel_path = "voicelines/fixture_hit.wav"
    _write_silent_wav(tmp_path / audio_rel_path)

    fixture_path = tmp_path / "proofread_fixture.json"
    _write_json(
        fixture_path,
        {
            "strict": True,
            "fallback_mode": "chunk_text",
            "entries": [
                {
                    "audio_path": audio_rel_path,
                    "transcript_text": "Proofread fixture transcript.",
                }
            ],
        },
    )

    monkeypatch.setenv("THREADSPEAK_E2E_PROOFREAD_FIXTURE", str(fixture_path))
    monkeypatch.setenv("THREADSPEAK_E2E_PROOFREAD_FALLBACK", "chunk_text")

    manager = ProjectManager(str(tmp_path))
    try:
        manager.get_asr_engine = lambda: (_ for _ in ()).throw(AssertionError("Real ASR should not be called"))

        first = manager.transcribe_audio_path(audio_rel_path)
        second = manager.transcribe_audio_path(audio_rel_path)

        assert first["text"] == "Proofread fixture transcript."
        assert first["cached"] is False
        assert first["simulated"] is True
        assert second["text"] == "Proofread fixture transcript."
        assert second["cached"] is True
    finally:
        manager.shutdown_script_store(flush=True)


def test_transcribe_audio_path_falls_back_to_chunk_text_in_test_mode(tmp_path: Path, monkeypatch):
    _create_project_root(tmp_path)
    audio_rel_path = "voicelines/chunk_fallback.wav"
    _write_silent_wav(tmp_path / audio_rel_path)

    fixture_path = tmp_path / "proofread_fixture.json"
    _write_json(
        fixture_path,
        {
            "strict": False,
            "fallback_mode": "chunk_text",
            "entries": [],
        },
    )

    monkeypatch.setenv("THREADSPEAK_E2E_PROOFREAD_FIXTURE", str(fixture_path))
    monkeypatch.setenv("THREADSPEAK_E2E_PROOFREAD_FALLBACK", "chunk_text")

    manager = ProjectManager(str(tmp_path))
    try:
        manager.get_asr_engine = lambda: (_ for _ in ()).throw(AssertionError("Real ASR should not be called"))
        manager.save_chunks(
            [
                {
                    "id": 0,
                    "uid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "speaker": "Narrator",
                    "text": "Fallback transcript from chunk text.",
                    "instruct": "",
                    "status": "done",
                    "audio_path": audio_rel_path,
                    "audio_validation": None,
                    "auto_regen_count": 0,
                }
            ]
        )

        result = manager.transcribe_audio_path(audio_rel_path)
        assert result["text"] == "Fallback transcript from chunk text."
        assert result["cached"] is False
        assert result["simulated"] is True
    finally:
        manager.shutdown_script_store(flush=True)

