import os
from runtime_layout import LAYOUT

_PROMPTS_FILE = LAYOUT.prompt_attribution_path


def load_attribution_prompts():
    """Read attribution_prompts.txt from disk and return (system_prompt, user_prompt)."""
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"attribution_prompts.txt not found at {os.path.abspath(_PROMPTS_FILE)}. "
            "This file is required for dialogue-attribution pruning."
        )

    parts = raw.split("---SEPARATOR---", maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(
            "attribution_prompts.txt is malformed: expected exactly one '---SEPARATOR---' delimiter."
        )

    return parts[0].strip(), parts[1].strip()


ATTRIBUTION_SYSTEM_PROMPT, ATTRIBUTION_USER_PROMPT = load_attribution_prompts()
