import os

_BASE_DIR = os.path.dirname(__file__)
_PROMPT_DEFAULTS_DIR = os.path.join(_BASE_DIR, "prompt_defaults")
_SEPARATOR = "---SEPARATOR---"


def _read_text(path: str, label: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"{label} not found at {os.path.abspath(path)}")
    if not value:
        raise RuntimeError(f"{label} is empty")
    return value


def _read_prompt_pair(filename: str, label: str):
    raw = _read_text(os.path.join(_PROMPT_DEFAULTS_DIR, filename), label)
    parts = raw.split(_SEPARATOR, maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(f"{label} is malformed: expected one '{_SEPARATOR}' delimiter")
    return parts[0].strip(), parts[1].strip()


def load_factory_default_prompts():
    system_prompt, user_prompt = _read_prompt_pair("default_prompts.txt", "default_prompts.txt")
    review_system_prompt, review_user_prompt = _read_prompt_pair("review_prompts.txt", "review_prompts.txt")
    attribution_system_prompt, attribution_user_prompt = _read_prompt_pair(
        "attribution_prompts.txt", "attribution_prompts.txt"
    )
    voice_prompt = _read_text(os.path.join(_PROMPT_DEFAULTS_DIR, "voice_prompt.txt"), "voice_prompt.txt")
    dialogue_prompt = _read_text(
        os.path.join(_PROMPT_DEFAULTS_DIR, "dialogue_identification_system_prompt.txt"),
        "dialogue_identification_system_prompt.txt",
    )
    temperament_prompt = _read_text(
        os.path.join(_PROMPT_DEFAULTS_DIR, "temperament_extraction_system_prompt.txt"),
        "temperament_extraction_system_prompt.txt",
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "review_system_prompt": review_system_prompt,
        "review_user_prompt": review_user_prompt,
        "attribution_system_prompt": attribution_system_prompt,
        "attribution_user_prompt": attribution_user_prompt,
        "voice_prompt": voice_prompt,
        "dialogue_identification_system_prompt": dialogue_prompt,
        "temperament_extraction_system_prompt": temperament_prompt,
    }
