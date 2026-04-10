# Output Formats

Threadspeak supports iterative editing and export after render.

## Final audiobook

- `cloned_audiobook.mp3` - merged audiobook output

## Individual voicelines

Saved as per-line files for DAW workflows (for example in `voicelines/`).

## Audacity export

`audacity_export.zip` typically contains:
- `project.lof`
- `labels.txt`
- one WAV track per speaker

Tracks are time-aligned with silence where that speaker is inactive.

## M4B export

- `audiobook.m4b` - chapter-capable AAC audiobook output
- chapters can be auto-detected or configured per chunk

## Project continuity

- Projects can be paused/resumed safely.
- Finished projects can be saved/loaded for later edits.
