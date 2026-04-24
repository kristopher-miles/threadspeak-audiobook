"""API endpoint tests split by behavior domain."""

from ._common import *  # noqa: F401,F403
from . import _common as common

def test_voice_design_list():
    r = get("/api/voice_design/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")

def test_voice_design_delete_404():
    r = delete(f"/api/voice_design/{TEST_PREFIX}fake_id")
    assert_status(r, 404)

def test_voice_design_preview():
    require_full_mode()
    r = post("/api/voice_design/preview", json={
        "description": "A clear young male voice with a steady tone",
        "sample_text": "This is a test of voice design.",
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")
    shared["preview_file"] = data["audio_url"].split("/")[-1]

def test_voice_design_save_and_delete():
    require_full_mode()
    preview_file = shared.get("preview_file")
    if not preview_file:
        raise TestFailure("SKIP: no preview file from previous test")

    r = post("/api/voice_design/save", json={
        "name": f"{TEST_PREFIX}voice_design",
        "description": "Test voice",
        "sample_text": "Test text",
        "preview_file": preview_file
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    voice_id = data["voice_id"]

    # Delete it
    r = delete(f"/api/voice_design/{voice_id}")
    assert_status(r, 200)


# ── Section 9b: Clone Voices ────────────────────────────────

def test_clone_voices_list():
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")

def test_clone_voices_upload_bad_format():
    files = {"file": ("test.txt", b"not audio", "text/plain")}
    r = requests.post(f"{common.BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 400)

def test_clone_voices_delete_404():
    r = delete(f"/api/clone_voices/{TEST_PREFIX}fake_id")
    assert_status(r, 404)

def test_clone_voices_upload_and_delete():
    # Create a minimal WAV file (44-byte header + silence)
    import struct
    sample_rate = 16000
    num_samples = 16000  # 1 second
    data_size = num_samples * 2
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size)
    wav_bytes = wav_header + b'\x00' * data_size

    files = {"file": (f"{TEST_PREFIX}clone_test.wav", wav_bytes, "audio/wav")}
    r = requests.post(f"{common.BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    assert_key(data, "filename")
    voice_id = data["voice_id"]

    # Verify it appears in list
    r = get("/api/clone_voices/list")
    assert_status(r, 200)
    found = any(v["id"] == voice_id for v in r.json())
    if not found:
        raise TestFailure(f"Uploaded voice {voice_id} not found in list")

    # Delete it
    r = delete(f"/api/clone_voices/{voice_id}")
    assert_status(r, 200)

    # Verify it's gone
    r = get("/api/clone_voices/list")
    found = any(v["id"] == voice_id for v in r.json())
    if found:
        raise TestFailure(f"Deleted voice {voice_id} still in list")


def _load_voice_profile_fixture():
    import json
    from pathlib import Path

    manifest_path = Path(REPO_DIR) / "app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json"
    if not manifest_path.exists():
        raise TestFailure(f"Missing fixture manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    profiles = manifest.get("voice_profiles") or []
    if not profiles:
        raise TestFailure("No fixture voice profiles available for realistic clone audio tests")
    return profiles


def test_clone_voices_upload_with_transcript_metadata():
    from pathlib import Path

    audio_path = Path(REPO_DIR) / "app/test_fixtures/e2e_sim/audio/GF-Ember.wav"
    if not audio_path.exists():
        raise TestFailure(f"Missing fixture clone audio: {audio_path}")

    sample_text = (
        "Please, the mark isn't my fault. I've had it since I was born, "
        "to curse and torment me. It can't hurt you"
    )

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f.read(), "audio/wav")}
    try:
        r = requests.post(f"{common.BASE_URL}/api/clone_voices/upload", files=files)
        assert_status(r, 200)
        data = r.json()
        assert_key(data, "voice_id")
        voice_id = data["voice_id"]
        if (data.get("sample_text") or "").strip() != sample_text:
            raise TestFailure(
                f"Expected upload response sample_text {sample_text!r}, got {data.get('sample_text')!r}"
            )

        r = get("/api/clone_voices/list")
        assert_status(r, 200)
        entry = next((item for item in r.json() if item.get("id") == voice_id), None)
        if not entry:
            raise TestFailure(f"Uploaded voice {voice_id} not found in list")

        stored_text = (entry.get("sample_text") or "").strip()
        if stored_text != sample_text:
            raise TestFailure(
                f"Expected uploaded clone sample_text {sample_text!r}, got {stored_text!r}"
            )
    finally:
        voice_id = data["voice_id"] if "data" in locals() else None
        if voice_id:
            try:
                delete(f"/api/clone_voices/{voice_id}")
            except Exception:
                pass


def test_clone_voices_upload_without_transcript_metadata():
    import struct

    sample_rate = 16000
    num_samples = 16000
    data_size = num_samples * 2
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size)
    wav_bytes = wav_header + b'\x00' * data_size
    files = {"file": (f"{TEST_PREFIX}clone_nometadata.wav", wav_bytes, "audio/wav")}

    r = requests.post(f"{common.BASE_URL}/api/clone_voices/upload", files=files)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "voice_id")
    assert_key(data, "filename")
    voice_id = data["voice_id"]

    try:
        r = get("/api/clone_voices/list")
        assert_status(r, 200)
        entry = next((item for item in r.json() if item.get("id") == voice_id), None)
        if not entry:
            raise TestFailure(f"Uploaded voice {voice_id} not found in list")
        if entry.get("sample_text"):
            raise TestFailure(
                f"Expected no embedded sample_text when none is present, got {entry.get('sample_text')!r}"
            )
    finally:
        # Ensure cleanup even on assertion failure.
        try:
            delete(f"/api/clone_voices/{voice_id}")
        except Exception:
            pass


def _extract_riff_wav_metadata(wav_bytes):
    import struct

    tags = {}
    if len(wav_bytes) < 12:
        return tags
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return tags

    idx = 12
    size = len(wav_bytes)
    while idx + 8 <= size:
        chunk_id = wav_bytes[idx:idx+4]
        chunk_size = struct.unpack("<I", wav_bytes[idx+4:idx+8])[0]
        data_start = idx + 8
        data_end = data_start + chunk_size
        if data_end > size:
            break

        if chunk_id == b"LIST" and data_end - data_start >= 4 and wav_bytes[data_start:data_start+4] == b"INFO":
            info_idx = data_start + 4
            while info_idx + 8 <= data_end:
                info_id = wav_bytes[info_idx:info_idx+4]
                if len(info_id) != 4:
                    break
                try:
                    key = info_id.decode("ascii")
                except Exception:
                    break
                if not key.isalnum():
                    break
                info_value_len = struct.unpack("<I", wav_bytes[info_idx+4:info_idx+8])[0]
                value_start = info_idx + 8
                value_end = value_start + info_value_len
                if value_end > data_end:
                    break
                value = wav_bytes[value_start:value_end].decode("utf-8", "replace").rstrip("\x00").strip()
                if key == "INAM":
                    tags["title"] = value
                elif key == "IART":
                    tags["artist"] = value
                elif key == "ICMT":
                    tags["comment"] = value
                info_idx = value_end + (info_value_len % 2)

        idx = data_end + (chunk_size % 2)

    return tags


def test_clone_voices_download_with_transcript_metadata():
    import json
    import os
    import tempfile
    from pathlib import Path
    from ffmpeg_utils import get_ffprobe_exe
    import subprocess

    manifest_path = Path(REPO_DIR) / "app/test_fixtures/e2e_sim/voice_profiles_test_book_manifest.json"
    if not manifest_path.exists():
        raise TestFailure(f"Missing fixture manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    profiles = manifest.get("voice_profiles") or []
    if not profiles:
        raise TestFailure("No fixture voice profiles available for realistic download metadata test")

    profile = None
    for candidate in profiles:
        sample = candidate.get("sample_text", "")
        try:
            sample.encode("ascii")
        except UnicodeEncodeError:
            continue
        profile = candidate
        break
    if profile is None:
        profile = profiles[0]

    audio_path = Path(REPO_DIR) / profile.get("fixture_audio_path", "")
    if not audio_path.exists():
        raise TestFailure(f"Missing fixture clone audio: {audio_path}")
    speaker = profile.get("speaker", "")
    ref_text = profile.get("sample_text", "")

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f.read(), "audio/wav")}

    try:
        r = requests.post(f"{common.BASE_URL}/api/clone_voices/upload", files=files)
        assert_status(r, 200)
        data = r.json()
        assert_key(data, "voice_id")
        voice_id = data["voice_id"]

        response = requests.get(
            f"{common.BASE_URL}/api/clone_voices/{voice_id}/download",
            params={"speaker": speaker, "ref_text": ref_text},
        )
        assert_status(response, 200)
        wav_bytes = response.content
        if not str(response.headers.get("Content-Type", "")).startswith("audio/wav"):
            raise TestFailure(f"Unexpected content type: {response.headers.get('Content-Type')}")

        tags = _extract_riff_wav_metadata(wav_bytes)
        if "title" not in tags or "artist" not in tags or "comment" not in tags:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                temp_file.write(wav_bytes)
                temp_file.flush()
                temp_file.close()
                ffprobe = get_ffprobe_exe()
                proc = subprocess.run(
                    [
                        ffprobe,
                        "-v", "error",
                        "-show_entries", "format_tags",
                        "-of", "json",
                        temp_file.name,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise TestFailure(f"ffprobe failed to inspect downloaded WAV: {proc.stderr.strip() or proc.stdout.strip()}")
                try:
                    probe_data = json.loads((proc.stdout or "{}").strip() or "{}")
                except Exception as exc:
                    raise TestFailure(f"ffprobe output could not be parsed as JSON: {exc}") from exc
                format_tags = (probe_data.get("format") or {}).get("tags", {})
                fallback = {
                    "title": (
                        format_tags.get("title")
                        or format_tags.get("Title")
                        or format_tags.get("INAM")
                    ),
                    "artist": (
                        format_tags.get("artist")
                        or format_tags.get("Artist")
                        or format_tags.get("IART")
                    ),
                    "comment": (
                        format_tags.get("comment")
                        or format_tags.get("Comment")
                        or format_tags.get("ICMT")
                    ),
                }
                tags = {**fallback, **{k: v for k, v in tags.items() if v}}
            finally:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

        if tags.get("artist") != speaker:
            raise TestFailure(f"Expected artist={speaker!r}, got {tags.get('artist')!r}")
        if tags.get("comment") != ref_text:
            raise TestFailure(f"Expected comment/ref text metadata to match source profile sample text")
        if not (tags.get("title") or "").startswith("Clone voice reference"):
            raise TestFailure(f"Expected title to include clone reference marker, got {tags.get('title')!r}")
        if not str(tags.get("title")).endswith(speaker):
            raise TestFailure(f"Expected title to end with speaker name {speaker!r}, got {tags.get('title')!r}")

    finally:
        # Clean up uploaded fixture data so test state stays minimal
        voice_id = data["voice_id"] if "data" in locals() else None
        if voice_id:
            try:
                delete(f"/api/clone_voices/{voice_id}")
            except Exception:
                pass

# ── Section 10: LoRA Datasets ───────────────────────────────

def test_lora_list_datasets():
    r = get("/api/lora/datasets")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")

def test_lora_delete_dataset_404():
    r = delete(f"/api/lora/datasets/{TEST_PREFIX}fake_ds")
    assert_status(r, 404)

def test_lora_upload_bad_file():
    files = {"file": (f"{TEST_PREFIX}bad.txt", io.BytesIO(b"not a zip"), "text/plain")}
    r = post("/api/lora/upload_dataset", files=files)
    # Should fail — not a valid zip
    if r.status_code < 400:
        raise TestFailure(f"Expected error for non-zip upload, got {r.status_code}")


# ── Section 11: LoRA Models ─────────────────────────────────

def test_lora_list_models():
    r = get("/api/lora/models")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")
    # Verify built-in adapters have 'downloaded' field
    for m in data:
        if m.get("builtin"):
            if "downloaded" not in m:
                raise TestFailure(f"Built-in adapter {m['id']} missing 'downloaded' field")
    shared["lora_models"] = data

def test_lora_download_invalid():
    r = post(f"/api/lora/download/{TEST_PREFIX}fake_adapter", json={})
    if r.status_code < 400:
        raise TestFailure(f"Expected error for invalid adapter, got {r.status_code}")

def test_lora_delete_model_404():
    r = delete(f"/api/lora/models/{TEST_PREFIX}fake_model")
    assert_status(r, 404)

def test_lora_train_bad_dataset():
    r = post("/api/lora/train", json={
        "name": f"{TEST_PREFIX}model",
        "dataset_id": f"{TEST_PREFIX}nonexistent_ds"
    })
    # Should fail — dataset does not exist
    if r.status_code < 400:
        raise TestFailure(f"Expected error for bad dataset, got {r.status_code}")

def test_lora_preview_404():
    r = post(f"/api/lora/preview/{TEST_PREFIX}fake_adapter")
    assert_status(r, 404)

def test_lora_preview():
    require_full_mode()
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post(f"/api/lora/preview/{adapter['id']}", timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")


# ── Section 12: Dataset Builder CRUD ────────────────────────

def test_dataset_builder_list():
    r = get("/api/dataset_builder/list")
    assert_status(r, 200)
    data = r.json()
    if not isinstance(data, list):
        raise TestFailure(f"Expected list, got {type(data).__name__}")

def test_dataset_builder_create():
    r = post("/api/dataset_builder/create", json={
        "name": f"{TEST_PREFIX}builder_proj"
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "name")

def test_dataset_builder_update_meta():
    r = post("/api/dataset_builder/update_meta", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "description": "A test voice description",
        "global_seed": "42"
    })
    assert_status(r, 200)

def test_dataset_builder_update_rows():
    r = post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "rows": [
            {"emotion": "neutral", "text": "Hello world.", "seed": ""},
            {"emotion": "happy", "text": "Great to see you!", "seed": ""}
        ]
    })
    assert_status(r, 200)
    data = r.json()
    if data.get("sample_count") != 2:
        raise TestFailure(f"Expected sample_count=2, got {data.get('sample_count')}")

def test_dataset_builder_status():
    r = get(f"/api/dataset_builder/status/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "description")
    assert_key(data, "samples")
    assert_key(data, "running")
    assert_key(data, "logs")
    if len(data["samples"]) != 2:
        raise TestFailure(f"Expected 2 samples, got {len(data['samples'])}")

def test_dataset_builder_cancel():
    r = post("/api/dataset_builder/cancel")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") not in ("not_running", "cancelling"):
        raise TestFailure(f"Unexpected cancel status: {data}")

def test_dataset_builder_save_no_samples():
    r = post("/api/dataset_builder/save", json={
        "name": f"{TEST_PREFIX}builder_proj",
        "ref_index": 0
    })
    # Should fail — no completed samples
    if r.status_code < 400:
        raise TestFailure(f"Expected error for save with no samples, got {r.status_code}")

def test_dataset_builder_delete():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}builder_proj")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "deleted":
        raise TestFailure(f"Expected status=deleted, got {data}")

def test_dataset_builder_delete_404():
    r = delete(f"/api/dataset_builder/{TEST_PREFIX}nonexistent")
    assert_status(r, 404)


# ── Section 13: Merge / Export ──────────────────────────────

def test_lora_test_model():
    require_full_mode()
    models = shared.get("lora_models", [])
    if not models:
        raise TestFailure("SKIP: no LoRA models available")
    adapter = models[0]
    r = post("/api/lora/test", json={
        "adapter_id": adapter["id"],
        "text": "This is a test of the LoRA voice.",
        "instruct": "Neutral, even delivery."
    }, timeout=120)
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "audio_url")

def test_lora_generate_dataset():
    require_full_mode()
    r = post("/api/lora/generate_dataset", json={
        "name": f"{TEST_PREFIX}dataset",
        "description": "A clear young male voice",
        "samples": [
            {"emotion": "neutral", "text": "Hello, this is a test sample."},
            {"emotion": "happy", "text": "Great to see you today!"}
        ]
    })
    if r.status_code == 400:
        raise TestFailure("SKIP: already running or bad request")
    assert_status(r, 200)
    data = r.json()
    if data.get("status") != "started":
        raise TestFailure(f"Expected status=started, got {data}")

def test_dataset_builder_generate_sample():
    require_full_mode()
    # Create a temp project for this test
    post("/api/dataset_builder/create", json={"name": f"{TEST_PREFIX}gen_proj"})
    post("/api/dataset_builder/update_rows", json={
        "name": f"{TEST_PREFIX}gen_proj",
        "rows": [{"emotion": "neutral", "text": "Hello world.", "seed": ""}]
    })

    r = post("/api/dataset_builder/generate_sample", json={
        "description": "A clear male voice",
        "text": "Hello world.",
        "dataset_name": f"{TEST_PREFIX}gen_proj",
        "sample_index": 0,
        "seed": -1
    })
    assert_status(r, 200)
    data = r.json()
    assert_key(data, "status")

    # Cleanup
    delete(f"/api/dataset_builder/{TEST_PREFIX}gen_proj")


# ── Run all tests ────────────────────────────────────────────
