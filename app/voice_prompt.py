import os
from runtime_layout import LAYOUT

_PROMPT_FILE = LAYOUT.prompt_voice_path


def load_voice_prompt():
    """Read voice_prompt.txt from disk and return the prompt template."""
    try:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"voice_prompt.txt not found at {os.path.abspath(_PROMPT_FILE)}. "
            "This file is required for voice suggestion defaults."
        )

    if not prompt:
        raise RuntimeError("voice_prompt.txt is empty.")

    return prompt


VOICE_PROMPT = load_voice_prompt()
