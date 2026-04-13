<img width="475" height="467" alt="Threadspeak Logo" src="https://github.com/kristopher-miles/threadspeak-audiobook/blob/main/icon.png?raw=true"/>

# Threadspeak Audiobook Generator

Transform a source script into a production-ready audiobook with an end-to-end pipeline for paragraph/chapter ingestion, dialogue attribution, temperament extraction, voice design, editing, local proofreading, and export.

## Example: [sample.mp3](https://github.com/user-attachments/files/25276110/sample.mp3)

## Features

### AI-Powered Pipeline
- **Script-first ingestion** - Upload a script document and split it into structured paragraphs and chapters
- **Dialogue attribution stage** - Identify speaker blocks across dialogue and narration
- **Temperament extraction stage** - Infer delivery/temperament for both dialogue and non-dialogue text
- **Script construction stage** - Build editable line-level script entries from processed blocks
- **Chapter-aware processing** - Preserve chapter boundaries throughout edit, render, proofread, and export
- **Project-safe workflow state** - Pause/resume long jobs and continue where you left off

### Voice Generation
- **Built-in TTS Engine** - Qwen3-TTS runs locally with no external server required
- **External Server Mode** - Optionally connect to a remote Qwen3-TTS Gradio server
- **Multi-Language Support** - English, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish, or Auto-detect
- **Custom Voices** - 9 pre-trained voices with instruct-based emotion/tone control
- **Voice Cloning** - Clone any voice from a 5-15 second reference audio sample
- **Voice Designer + Suggestion** - Generate candidate voice profiles using source excerpts and Qwen voice design
- **Editable character profiles** - Suggested voices are starting points; tune profiles manually to match your cast
- **LoRA Voice Training** - Fine-tune the Base model on custom voice datasets to create persistent voice identities with instruct-following
- **Built-in LoRA Presets** - Pre-trained voice adapters included out of the box, ready to assign to characters
- **Dataset Builder** - Interactive tool for creating LoRA training datasets with per-sample text, emotion, and audio preview
- **Batch Processing** - Generate many chunks simultaneously with high throughput
- **Codec Compilation** - Optional `torch.compile` optimization for 3-4x faster batch decoding
- **Silence-aware timeline** - Insert and tune silence blocks in the editor/export flow

### Web UI Editor
- **Pipeline tabs aligned to production flow** - Setup, Script, Voices, Editor, Proofread, Export
- **Line-level correction** - Fix attribution errors, rewrite lines, adjust temperament, and add silence blocks
- **Chapter timeline management** - Chapters are visible and editable while rendering
- **Performance telemetry** - Running throughput metrics and completion-time estimates during generation
- **Proofread feedback loop** - Local Whisper grading to find and repair weak lines before final export
- **Project continuity** - Suspend/resume active projects and reload finished projects later

### Export Options
- **Combined Audiobook** - Single MP3 with all voices and natural pauses
- **Individual Voicelines** - Separate MP3 per line for DAW editing (Audacity, etc.)
- **Audacity Export** - Zip with per-speaker WAV tracks, LOF project file, and labels for multi-track editing
- **Export controls** - Configure silence lengths and normalization at export time
- **M4B Audiobook** - Chaptered M4B (AAC) with per-chunk or auto-detected chapter markers for audiobook players (Audiobookshelf, Apple Books, VLC, etc.)

## Requirements

- [Pinokio](https://pinokio.computer/)
- LLM server (one of the following):
  - [LM Studio](https://lmstudio.ai/) (local) - recommended: Qwen3 or similar
  - [Ollama](https://ollama.ai/) (local)
  - [OpenAI API](https://platform.openai.com/) (cloud)
  - Any OpenAI-compatible API
- **GPU:** 8 GB VRAM minimum, 16 GB+ recommended — see compatibility table below
  - Each TTS model uses ~3.4 GB; remaining VRAM determines batch size
  - CPU mode available on all platforms but significantly slower
- **RAM:** 16 GB recommended (8 GB minimum)
- **Disk:** ~20 GB (8 GB venv/PyTorch, ~7 GB for model weights, working space for audio)

### GPU Compatibility

| GPU | OS | Status | Driver Requirement | Notes |
|-----|-----|--------|-------------------|-------|
| **NVIDIA** | Windows | Full support | Driver 550+ (CUDA 12.8) | Flash attention included for faster encoding |
| **NVIDIA** | Linux | Full support | Driver 550+ (CUDA 12.8) | Flash attention + triton included |
| **AMD** | Linux | Full support | ROCm 6.3 | ROCm optimizations applied automatically |
| **AMD** | Windows | CPU only | N/A | GPU acceleration is not supported — the app runs in CPU mode. For GPU acceleration with AMD, use Linux |
| **Apple Silicon** | macOS | Full support | N/A | Auto-detects Apple Silicon and uses MPS-compatible Qwen3 assets. |
| **Intel** | macOS | CPU only | N/A | Functional, but slow. |

> **Note:** No external TTS server is required. Threadspeak includes a built-in Qwen3-TTS engine that loads models directly. Model weights are downloaded automatically on first use (~3.5 GB per model variant).

## Installation

### Option A: Pinokio (Recommended)

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. Open Threadspeak in Pinokio
   - In Pinokio, click **Download** and paste `https://github.com/kristopher-miles/threadspeak-audiobook`
3. Click **Install** to set up dependencies
4. Click **Start** to launch the web interface

Pinokio installs both runtime and test dependencies into `app/env`. Use the Pinokio `Run Tests` action, or run `app/env/bin/python -m pytest -q` from the repo root to execute the test suite against the app-managed environment.

## First Launch

See [First Launch](wiki/First-Launch.md).

## Quick Start

Threadspeak follows a script-ingestion production pipeline:

### Core Pipeline

**Step 1 — Setup**
- Configure LLM + TTS connectivity and runtime options
- Set `LLM Base URL`, API key, and model name
- Use built-in `local` TTS mode unless you intentionally run an external server
- Save configuration

**Step 2 — Script (Ingestion + Analysis)**
- Upload the source script document
- Process the document into paragraphs and chapters
- Run dialogue speaker identification
- Run temperament extraction over dialogue and non-dialogue text
- Build the editable line-level script

**Step 3 — Voices**
- Open generated speaker cards and assign voice strategy (custom, clone, LoRA, or design)
- Use suggested source excerpts and voice design outputs to bootstrap character voices
- Edit voice profiles manually so they match your intended cast

**Step 4 — Editor**
- Correct attribution mistakes, rewrite lines, adjust instruct/temperament, and insert silence blocks
- Render pending chunks and regenerate selectively
- Monitor chapter-level progress, throughput, and completion estimates

**Step 5 — Proofread**
- Run local Whisper-based validation on rendered audio
- Review low-confidence lines, then retry or edit until quality is acceptable

**Step 6 — Export**
- Export merged MP3 and/or Audacity project
- Tune silence lengths and normalization settings before final output

Projects can be paused/resumed safely, and completed projects can be saved and reloaded for later revision.

## Web Interface

### Setup Tab
Configure connections to your LLM and TTS engine.

**TTS Settings:**
- **Mode** - `local` (built-in engine) or `external` (connect to Gradio server)
- **Device** - `auto` (recommended), `cuda`, `cpu`, or `mps`
- **Language** - TTS synthesis language: English (default), Chinese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish, or Auto (let the model detect)
- **Parallel Workers** - Batch size for fast batch rendering (higher = more VRAM usage)
- **Batch Seed** - Fixed seed for reproducible batch output (leave empty for random)
- **Compile Codec** - Enable `torch.compile` for 3-4x faster batch decoding (adds ~30-60s warmup on first generation)
- **Sub-batching** - Split batches by text length to reduce wasted GPU compute on padding (enabled by default)
- **Min Sub-batch Size** - Minimum chunks per sub-batch before allowing a split (default: 4)
- **Length Ratio** - Maximum longest/shortest text length ratio before forcing a sub-batch split (default: 5)
- **Max Chars** - Maximum total characters per sub-batch; lower values reduce VRAM usage (default: 3000)

**Prompt Settings (Advanced):**
- **Pipeline stage settings** - Tune ingestion, attribution, temperament, and script-building behavior
- **LLM sampling parameters** - Temperature, Top P, Top K, Min P, and Presence Penalty
- **Banned tokens** - Comma-separated list of tokens to suppress in model output (useful when a model emits unwanted control text)
- **Prompt customization** - System/user prompts are loaded from prompt files and can be overridden in-session. Use "Reset to Defaults" to reload file defaults without restarting
- **Voice suggestion prompt** - Controls how source excerpts are converted into character voice-profile suggestions

### Script Tab
Script tab is the ingestion and analysis control center:
- Upload source document
- Build paragraph and chapter structure
- Run dialogue attribution
- Run temperament extraction
- Assemble the editable script document used by editor and audio stages

This stage handles structure and initial labeling. Final edge-case corrections happen in **Editor** and **Proofread**.

### Voices Tab
Voices are derived from the script's detected cast and then refined:
- Generate suggested character voice profiles from source excerpts
- Use Qwen voice design to create candidate voices
- Edit each speaker profile to match your intended character identity
- Assign custom, clone, LoRA, or design voice modes as needed
- Save and reuse designed voices for future projects

### Editor Tab
Editor is the main production workspace:
- Correct speaker attribution mistakes
- Edit line text and temperament/instruct
- Add silence blocks where pacing needs manual control
- Regenerate individual lines or render pending batches
- Work chapter-by-chapter with active performance stats and ETA tracking
- Iterate until script and audio are ready for proofread

### Proofread Tab
Proofread uses a local Whisper model to grade generated audio against expected text:
- Flag low-confidence or mismatched lines
- Review failures quickly
- Retry generation and/or edit text/instruct
- Re-run checks until quality threshold is met

### Render Modes

There are two modes to render audio.

#### Render Pending (Standard)
The default rendering mode. Sends individual TTS calls in parallel using the configured worker count.

- **Per-speaker seeds** - Each voice uses its configured seed for reproducible output
- **Voice cloning support** - Works with both custom voices and cloned voices

#### Batch (Fast)
High-speed rendering that sends multiple lines to the TTS engine in a single batched call. Chunks are sorted by text length and processed in optimized sub-batches to minimize padding waste.

- **3-6x real-time throughput** - With codec compilation enabled, batches of 20-60 chunks process at 3-6x real-time speed
- **Sub-batching** - Automatically groups similarly-sized chunks together for efficient GPU utilization
- **Single seed** - All voices share the `Batch Seed` from config (set empty for random)
- **All voice types supported** - Custom, Clone, and LoRA voices are batched; Voice Design is sequential
- **Parallel Workers** setting controls batch size (higher values use more VRAM)

### Export Tab
The final export workflow is run from the export tab:
- Configure export pacing controls (silence lengths)
- Configure normalization controls
- Export merged MP3
- Export Audacity project (LOF + labels + tracks)
- Optionally export chaptered M4B

> **Note:** Some Linux audiobook players (e.g. Cozy) have limited M4B support and may not detect the file. The M4B output has been tested with VLC, Haruna, and Audiobookshelf.

## More Info

- [Wiki Home](wiki/Home.md)
- [LoRA Training Guide](wiki/LoRA-Training-Guide.md)
- [Voice Reference](wiki/Voice-Reference.md)
- [API Reference](wiki/API-Reference.md)
- [Automation Examples (Python/JavaScript)](wiki/Automation-Examples.md)
- [Project Structure](wiki/Project-Structure.md)
- [First Launch](wiki/First-Launch.md)
- [Performance and ROCm](wiki/Performance-and-ROCm.md)
- [Script Format](wiki/Script-Format.md)
- [Output Formats](wiki/Output-Formats.md)
- [Model Recommendations](wiki/Model-Recommendations.md)
- [Prompt Customization](wiki/Prompt-Customization.md)
- [Troubleshooting](wiki/Troubleshooting.md)

## License

MIT

> Threadspeak originated as a fork of [Alexandria Audiobook Generator](https://github.com/Finrandojin/alexandria-audiobook) by Finrandojin.

### Third-Party Licenses
- [qwen_tts](https://github.com/Qwen/Qwen3-TTS) — Apache License 2.0, Copyright Alibaba Qwen Team
