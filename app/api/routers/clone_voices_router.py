from fastapi import APIRouter
from starlette.background import BackgroundTask
from starlette.responses import FileResponse
from .. import shared as _shared
from ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

CLONE_VOICES_MANIFEST = os.path.join(CLONE_VOICES_DIR, "manifest.json")
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def _cleanup_file(path: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _sanitize_metadata(value: str) -> str:
    text = (value or "").strip().replace("\r", " ").replace("\n", " ")
    return text[:4096]


def _build_clone_voice_download_filename(entry: dict, voice_id: str) -> str:
    base_name = str(entry.get("name") or "").strip()
    safe_base = _sanitize_name(base_name) or _sanitize_name(voice_id) or "clone_voice"
    return f"{safe_base}.wav"


def _write_clone_voice_with_metadata(source_path: str, output_path: str, *, speaker: str, ref_text: str) -> str:
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", source_path,
        "-c:a", "pcm_s16le",
        "-f", "wav",
    ]
    title = "Clone voice reference"
    if speaker:
        title = f"{title}: {_sanitize_metadata(speaker)}"
        cmd += ["-metadata", f"title={title}", "-metadata", f"artist={_sanitize_metadata(speaker)}"]
    if ref_text:
        cmd += ["-metadata", f"comment={_sanitize_metadata(ref_text)}"]
    cmd += ["-map_metadata", "-1", output_path]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clone voice metadata encode failed (code {result.returncode}): {result.stderr or result.stdout}".strip()
        )
    return output_path


def _parse_clone_upload_metadata(path: str) -> str:
    ffprobe_exe = get_ffprobe_exe()
    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-show_entries", "format_tags",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return ""

    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return ""

    tags = (payload.get("format") or {}).get("tags") or {}
    for key in ("comment", "Comment", "ICMT", "description", "DESCRIPTION"):
        value = (tags.get(key) or "").strip()
        if value:
            return value[:4096]
    return ""


@router.get("/api/clone_voices/list")
async def clone_voices_list():
    """List all uploaded clone voices."""
    return _load_manifest(CLONE_VOICES_MANIFEST)

@router.post("/api/clone_voices/upload")
async def clone_voices_upload(file: UploadFile = File(...)):
    """Upload an audio file for voice cloning."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {', '.join(ALLOWED_AUDIO_EXTS)}")

    base_name = os.path.splitext(file.filename)[0]
    safe_name = _sanitize_name(base_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    voice_id = f"{safe_name}_{int(time.time())}"
    dest_filename = f"{voice_id}{ext}"
    dest_path = os.path.join(CLONE_VOICES_DIR, dest_filename)

    async with aiofiles.open(dest_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    sample_text = _parse_clone_upload_metadata(dest_path)

    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    manifest.append({
        "id": voice_id,
        "name": base_name,
        "filename": dest_filename,
        "sample_text": sample_text,
    })
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice uploaded: '{base_name}' as {dest_filename}")
    return {
        "status": "uploaded",
        "voice_id": voice_id,
        "filename": dest_filename,
        "sample_text": sample_text,
    }


@router.get("/api/clone_voices/{voice_id}/download")
async def clone_voices_download(
    voice_id: str,
    speaker: str = "",
    ref_text: str = "",
):
    """Download a selected clone voice as WAV with readable metadata."""
    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Clone voice not found")

    source_path = os.path.join(CLONE_VOICES_DIR, entry.get("filename", ""))
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Clone voice file not found")

    filename = _build_clone_voice_download_filename(entry, voice_id)
    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".wav" and not speaker and not ref_text:
        return FileResponse(
            source_path,
            media_type="audio/wav",
            filename=filename,
        )

    output_path = _make_runtime_temp_file("clone_voice_download_", suffix=".wav")
    try:
        _write_clone_voice_with_metadata(source_path, output_path, speaker=speaker, ref_text=ref_text)
    except Exception as exc:
        _cleanup_file(output_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=filename,
        background=BackgroundTask(_cleanup_file, output_path),
    )

@router.delete("/api/clone_voices/{voice_id}")
async def clone_voices_delete(voice_id: str):
    """Delete an uploaded clone voice."""
    manifest = _load_manifest(CLONE_VOICES_MANIFEST)
    entry = next((v for v in manifest if v["id"] == voice_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Clone voice not found")

    wav_path = os.path.join(CLONE_VOICES_DIR, entry["filename"])
    if os.path.exists(wav_path):
        os.remove(wav_path)

    manifest = [v for v in manifest if v["id"] != voice_id]
    _save_manifest(CLONE_VOICES_MANIFEST, manifest)

    logger.info(f"Clone voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}
