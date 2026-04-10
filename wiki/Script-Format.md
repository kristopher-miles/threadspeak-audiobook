# Script Format

Threadspeak builds a line-level working script from paragraph/chapter ingestion plus attribution and temperament stages.

## Core entry shape

Typical entries include:
- `speaker`
- `text`
- `instruct`

Example:

```json
[
  {"speaker": "NARRATOR", "text": "The door creaked open slowly.", "instruct": "Calm, even narration."},
  {"speaker": "ELENA", "text": "Ah! Who's there?", "instruct": "Startled and fearful, sharp whispered question."}
]
```

## `instruct` guidance

`instruct` is the voice-delivery direction passed to TTS. Keep it focused on delivery/emotion/tone, not physical stage directions.

## Non-verbal sounds

Use pronounceable text directly (not bracket tags), for example:
- Gasps: `Ah!`, `Oh!`
- Sighs: `Haah...`
- Laughter: `Haha...`
- Crying sounds: `Hic... sniff...`
