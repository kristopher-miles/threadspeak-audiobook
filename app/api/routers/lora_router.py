from fastapi import APIRouter
from .. import shared as _shared

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

router = APIRouter()

LORA_MODELS_MANIFEST = os.path.join(LORA_MODELS_DIR, "manifest.json")

def _load_builtin_lora_manifest():
    """Load built-in LoRA manifest from HF (with local fallback). Returns ALL entries with download status."""
    entries = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    result = []
    for entry in entries:
        entry = dict(entry)  # avoid mutating cached list
        local_id = entry["id"] if entry["id"].startswith("builtin_") else f"builtin_{entry['id']}"
        downloaded = is_adapter_downloaded(local_id, BUILTIN_LORA_DIR)
        entry["id"] = local_id
        entry["builtin"] = True
        entry["downloaded"] = downloaded
        entry["adapter_path"] = f"builtin_lora/{local_id}" if downloaded else None
        result.append(entry)
    return result

@router.post("/api/lora/upload_dataset")
async def lora_upload_dataset(file: UploadFile = File(...)):
    """Upload a ZIP containing WAV files and metadata.jsonl."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    # Derive dataset name from ZIP filename
    dataset_name = re.sub(r'[^\w\- ]', '', os.path.splitext(file.filename)[0]).strip()
    dataset_name = re.sub(r'\s+', '_', dataset_name).lower()
    if not dataset_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name from filename")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' already exists")

    # Save ZIP temporarily, then extract
    tmp_path = os.path.join(LORA_DATASETS_DIR, f"_tmp_{dataset_name}.zip")
    try:
        async with aiofiles.open(tmp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(dataset_dir)

        # Check for metadata.jsonl (may be inside a subdirectory)
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            # Check one level deep
            for entry in os.listdir(dataset_dir):
                candidate = os.path.join(dataset_dir, entry, "metadata.jsonl")
                if os.path.isdir(os.path.join(dataset_dir, entry)) and os.path.exists(candidate):
                    # Move contents up
                    nested = os.path.join(dataset_dir, entry)
                    for item in os.listdir(nested):
                        shutil.move(os.path.join(nested, item), os.path.join(dataset_dir, item))
                    os.rmdir(nested)
                    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
                    break

        if not os.path.exists(metadata_path):
            shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=400, detail="ZIP must contain metadata.jsonl")

        # Count samples
        sample_count = 0
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_count += 1

        logger.info(f"LoRA dataset uploaded: '{dataset_name}' ({sample_count} samples)")
        return {"status": "uploaded", "dataset_id": dataset_name, "sample_count": sample_count}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/api/lora/generate_dataset")
async def lora_generate_dataset(request: LoraGenerateDatasetRequest, background_tasks: BackgroundTasks):
    """Generate a LoRA training dataset using Voice Designer.

    Generates multiple audio samples with the same voice description,
    saving them as a ready-to-train dataset.
    """
    if process_state["dataset_gen"]["running"]:
        raise HTTPException(status_code=400, detail="Dataset generation already running")

    # Build unified sample list from either format
    sample_list = []
    if request.samples:
        for s in request.samples:
            if s.text.strip():
                sample_list.append({"emotion": s.emotion.strip(), "text": s.text.strip()})
    elif request.texts:
        for t in request.texts:
            if t.strip():
                sample_list.append({"emotion": "", "text": t.strip()})

    if not sample_list:
        raise HTTPException(status_code=400, detail="Provide at least one sample text")

    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")

    dataset_dir = os.path.join(LORA_DATASETS_DIR, safe_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{safe_name}' already exists")

    total = len(sample_list)
    root_description = request.description.strip()

    def task():
        process_state["dataset_gen"]["running"] = True
        process_state["dataset_gen"]["logs"] = [
            f"Generating {total} samples with VoiceDesign..."
        ]
        try:
            engine = project_manager.get_engine()
            if not engine:
                process_state["dataset_gen"]["logs"].append("Error: TTS engine not initialized")
                return

            os.makedirs(dataset_dir, exist_ok=True)
            metadata_lines = []
            completed = 0

            for i, sample in enumerate(sample_list):
                text = sample["text"]
                emotion = sample["emotion"]
                # Build full description: root + emotion if provided
                description = f"{root_description}, {emotion}" if emotion else root_description

                process_state["dataset_gen"]["logs"].append(
                    f"[{i+1}/{total}] {('[' + emotion + '] ' if emotion else '')}\"{ text[:60]}{'...' if len(text) > 60 else ''}\""
                )
                try:
                    wav_path, sr = engine.generate_voice_design(
                        description=description,
                        sample_text=_apply_project_dictionary(text),
                        language=request.language,
                    )
                    # Copy to dataset dir with sequential name
                    dest_filename = f"sample_{i:03d}.wav"
                    dest_path = os.path.join(dataset_dir, dest_filename)
                    shutil.copy2(wav_path, dest_path)

                    # Save first successful sample as ref.wav for consistent speaker embedding
                    if completed == 0:
                        shutil.copy2(wav_path, os.path.join(dataset_dir, "ref.wav"))

                    metadata_lines.append(json.dumps({
                        "audio_filepath": dest_filename,
                        "text": text,
                        "ref_audio": "ref.wav",
                    }, ensure_ascii=False))
                    completed += 1
                    process_state["dataset_gen"]["logs"].append(
                        f"  Saved {dest_filename}"
                    )
                except Exception as e:
                    process_state["dataset_gen"]["logs"].append(
                        f"  Failed: {e}"
                    )

            # Write metadata.jsonl
            metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("\n".join(metadata_lines) + "\n")

            process_state["dataset_gen"]["logs"].append(
                f"Dataset '{safe_name}' complete: {completed}/{total} samples generated."
            )
            logger.info(f"LoRA dataset generated: '{safe_name}' ({completed} samples)")

        except Exception as e:
            process_state["dataset_gen"]["logs"].append(f"Error: {e}")
            logger.error(f"Dataset generation error: {e}")
            # Clean up partial dataset on failure
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        finally:
            process_state["dataset_gen"]["running"] = False

    background_tasks.add_task(task)
    return {"status": "started", "dataset_id": safe_name, "total": total}

@router.get("/api/lora/datasets")
async def lora_list_datasets():
    """List uploaded LoRA training datasets."""
    datasets = []
    if not os.path.exists(LORA_DATASETS_DIR):
        return datasets

    for name in sorted(os.listdir(LORA_DATASETS_DIR)):
        dataset_dir = os.path.join(LORA_DATASETS_DIR, name)
        if not os.path.isdir(dataset_dir):
            continue
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        sample_count = 0
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_count += 1
        datasets.append({"dataset_id": name, "sample_count": sample_count})
    return datasets

@router.delete("/api/lora/datasets/{dataset_id}")
async def lora_delete_dataset(dataset_id: str):
    """Delete an uploaded dataset."""
    dataset_dir = os.path.join(LORA_DATASETS_DIR, dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(dataset_dir)
    logger.info(f"LoRA dataset deleted: {dataset_id}")
    return {"status": "deleted", "dataset_id": dataset_id}

@router.post("/api/lora/train")
async def lora_start_training(request: LoraTrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training as a subprocess."""
    if process_state["lora_training"]["running"]:
        raise HTTPException(status_code=400, detail="LoRA training already running")

    # Validate dataset exists
    dataset_dir = os.path.join(LORA_DATASETS_DIR, request.dataset_id)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{request.dataset_id}' not found")

    # Build output directory
    safe_name = _sanitize_name(request.name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid adapter name")

    adapter_id = f"{safe_name}_{int(time.time())}"
    output_dir = os.path.join(LORA_MODELS_DIR, adapter_id)

    # Unload TTS engine to free GPU
    if project_manager.engine is not None:
        logger.info("Unloading TTS engine for LoRA training...")
        project_manager.engine = None
        gc.collect()

    # Build subprocess command
    command = [
        sys.executable, "-u", "train_lora.py",
        "--data_dir", dataset_dir,
        "--output_dir", output_dir,
        "--epochs", str(request.epochs),
        "--lr", str(request.lr),
        "--batch_size", str(request.batch_size),
        "--lora_r", str(request.lora_r),
        "--lora_alpha", str(request.lora_alpha),
        "--gradient_accumulation_steps", str(request.gradient_accumulation_steps),
    ]
    run_id = _start_task_run("lora_training")

    def on_training_complete():
        """After training subprocess finishes, update manifest if adapter was saved."""
        run_process(command, "lora_training", run_id)

        # Check if training produced an adapter
        if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "training_meta.json")):
            try:
                with open(os.path.join(output_dir, "training_meta.json"), "r") as f:
                    meta = json.load(f)

                manifest = _load_manifest(LORA_MODELS_MANIFEST)
                manifest.append({
                    "id": adapter_id,
                    "name": request.name,
                    "dataset_id": request.dataset_id,
                    "epochs": meta.get("epochs", request.epochs),
                    "final_loss": meta.get("final_loss"),
                    "sample_count": meta.get("num_samples"),
                    "lora_r": meta.get("lora_r"),
                    "lr": meta.get("lr"),
                    "created": time.time(),
                })
                _save_manifest(LORA_MODELS_MANIFEST, manifest)
                logger.info(f"LoRA adapter registered: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to update LoRA manifest: {e}")

    background_tasks.add_task(on_training_complete)
    return {"status": "started", "adapter_id": adapter_id, "run_id": run_id}

@router.get("/api/lora/models")
async def lora_list_models():
    """List all LoRA adapters (built-in + user-trained)."""
    models = _load_builtin_lora_manifest() + _load_manifest(LORA_MODELS_MANIFEST)
    for m in models:
        is_builtin = m.get("builtin", False)
        is_downloaded = m.get("downloaded", True)  # user-trained are always downloaded

        if not is_downloaded:
            m["preview_audio_url"] = None
            continue

        if is_builtin:
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, m["id"])
            url_prefix = f"/builtin_lora/{m['id']}"
        else:
            adapter_dir = os.path.join(LORA_MODELS_DIR, m["id"])
            url_prefix = f"/lora_models/{m['id']}"
        preview_path = os.path.join(adapter_dir, "preview_sample.wav")
        m["preview_audio_url"] = f"{url_prefix}/preview_sample.wav" if os.path.exists(preview_path) else None
    return models

@router.delete("/api/lora/models/{adapter_id}")
async def lora_delete_model(adapter_id: str):
    """Delete a trained LoRA adapter. Built-in adapters cannot be deleted."""
    builtin = _load_builtin_lora_manifest()
    if any(m["id"] == adapter_id for m in builtin):
        raise HTTPException(status_code=403, detail="Built-in adapters cannot be deleted")
    manifest = _load_manifest(LORA_MODELS_MANIFEST)
    entry = next((m for m in manifest if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    # Delete adapter directory
    adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
    if os.path.isdir(adapter_dir):
        shutil.rmtree(adapter_dir)

    # Remove from manifest
    manifest = [m for m in manifest if m["id"] != adapter_id]
    _save_manifest(LORA_MODELS_MANIFEST, manifest)

    logger.info(f"LoRA adapter deleted: {adapter_id}")
    return {"status": "deleted", "adapter_id": adapter_id}

@router.post("/api/lora/download/{adapter_id}")
async def lora_download_builtin(adapter_id: str):
    """Download a built-in LoRA adapter from HuggingFace."""
    manifest = fetch_builtin_manifest(BUILTIN_LORA_DIR)
    hf_name = adapter_id.replace("builtin_", "", 1)
    entry = next((e for e in manifest if e["id"] == hf_name or e["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown built-in adapter: {adapter_id}")

    if is_adapter_downloaded(adapter_id, BUILTIN_LORA_DIR):
        return {"status": "already_downloaded", "adapter_id": adapter_id}

    try:
        download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
        logger.info(f"Built-in adapter downloaded: {adapter_id}")
        return {"status": "downloaded", "adapter_id": adapter_id}
    except Exception as e:
        logger.error(f"Download failed for {adapter_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/lora/test")
async def lora_test_model(request: LoraTestRequest):
    """Generate test audio using a LoRA adapter (built-in or user-trained)."""
    # Check both manifests
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == request.adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        audio_url_prefix = f"/builtin_lora/{request.adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, request.adapter_id)
        audio_url_prefix = f"/lora_models/{request.adapter_id}"

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(request.adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, request.adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        output_filename = f"test_{request.adapter_id}_{int(time.time())}.wav"
        output_path = os.path.join(adapter_dir, output_filename)

        voice_data = {
            "type": "lora",
            "adapter_id": request.adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_test_": voice_data}
        engine.generate_voice(
            text=_apply_project_dictionary(request.text),
            instruct_text=request.instruct or "",
            speaker="_lora_test_",
            voice_config=voice_config,
            output_path=output_path,
        )

        return {
            "status": "ok",
            "audio_url": f"{audio_url_prefix}/{output_filename}",
        }
    except Exception as e:
        logger.error(f"LoRA test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

LORA_PREVIEW_TEXT = "The ancient library stood at the crossroads of two forgotten paths, its weathered stone walls covered in ivy that had been growing for centuries."

@router.post("/api/lora/preview/{adapter_id}")
async def lora_preview(adapter_id: str):
    """Generate or return cached preview audio for a LoRA adapter."""
    builtin = _load_builtin_lora_manifest()
    user_trained = _load_manifest(LORA_MODELS_MANIFEST)
    all_adapters = builtin + user_trained
    entry = next((m for m in all_adapters if m["id"] == adapter_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Adapter not found")

    is_builtin = entry.get("builtin", False)
    if is_builtin:
        adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        url_prefix = f"/builtin_lora/{adapter_id}"
    else:
        adapter_dir = os.path.join(LORA_MODELS_DIR, adapter_id)
        url_prefix = f"/lora_models/{adapter_id}"

    if not os.path.isdir(adapter_dir) and is_builtin:
        try:
            download_builtin_adapter(adapter_id, BUILTIN_LORA_DIR)
            adapter_dir = os.path.join(BUILTIN_LORA_DIR, adapter_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Auto-download failed: {e}")
    elif not os.path.isdir(adapter_dir):
        raise HTTPException(status_code=404, detail="Adapter files not found")

    preview_path = os.path.join(adapter_dir, "preview_sample.wav")

    # Return cached if exists
    if os.path.exists(preview_path):
        return {"status": "cached", "audio_url": f"{url_prefix}/preview_sample.wav"}

    # Generate preview
    engine = project_manager.get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS engine")

    try:
        voice_data = {
            "type": "lora",
            "adapter_id": adapter_id,
            "adapter_path": adapter_dir,
        }
        voice_config = {"_lora_preview_": voice_data}
        engine.generate_voice(
            text=_apply_project_dictionary(LORA_PREVIEW_TEXT),
            instruct_text="",
            speaker="_lora_preview_",
            voice_config=voice_config,
            output_path=preview_path,
        )
        return {"status": "generated", "audio_url": f"{url_prefix}/preview_sample.wav"}
    except Exception as e:
        logger.error(f"LoRA preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

## ── Dataset Builder ──────────────────────────────────────────
