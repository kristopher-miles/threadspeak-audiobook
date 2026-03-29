import os

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "..", "default_prompts.txt")
_DIALOGUE_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "dialogue_identification_system_prompt.txt"
)
_TEMPERAMENT_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "temperament_extraction_system_prompt.txt"
)


def load_default_prompts():
    """Read default_prompts.txt from disk and return (system_prompt, user_prompt).

    Re-reads on every call so edits are picked up without restarting the app.
    """
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"default_prompts.txt not found at {os.path.abspath(_PROMPTS_FILE)}. "
            "This file is required for LLM prompt defaults."
        )

    parts = raw.split("---SEPARATOR---", maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(
            "default_prompts.txt is malformed: expected exactly one '---SEPARATOR---' delimiter."
        )

    return parts[0].strip(), parts[1].strip()


_FALLBACK_TEMPERAMENT_EXTRACTION_PROMPT = """\
You are a narrative tone specialist helping to produce an audiobook. \
Your sole task is to identify the emotional sentiment and delivery style \
of a given paragraph of prose.

You will receive:
  - A passage of text from the book (context window centred on the target paragraph)
  - The target paragraph (or dialogue lines) you must evaluate

Rules:
  - Respond with a single concise sentence describing how the reader should emotionally \
deliver the paragraph (e.g. "Read with quiet melancholy and a slow, deliberate pace").
  - You MUST call the identify_sentiment tool with your answer. Do not respond in plain text.\
"""

_FALLBACK_DIALOGUE_IDENTIFICATION_PROMPT = """\
You are a dialogue attribution specialist helping to build an audiobook script. \
Your sole task is to identify who speaks a given piece of dialogue based on the surrounding narrative.

You will receive:
  - A passage of text from the book (context window centred on the target paragraph)
  - A list of character names already encountered in the story
  - The quoted dialogue lines from the target paragraph

Rules:
  - One paragraph = one speaker. Do not split attribution.
  - If a known character fits the context clues (e.g. "he said", "Sarah replied", speech patterns), \
use their exact name from the list.
  - If the dialogue belongs to a character not yet seen, provide that character's name.
  - Use NARRATOR only for clearly unvoiced internal thoughts printed inside quotation marks \
(a rare edge case).
  - You MUST call the identify_dialogue tool with your answer. Do not respond in plain text.\
"""


def _load_single_prompt_file(path: str, label: str, fallback: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except FileNotFoundError:
        return fallback
    if not value:
        raise RuntimeError(f"{label} is empty.")
    return value


def load_dialogue_identification_prompt():
    return _load_single_prompt_file(
        _DIALOGUE_PROMPT_FILE,
        "dialogue_identification_system_prompt.txt",
        _FALLBACK_DIALOGUE_IDENTIFICATION_PROMPT,
    )


def load_temperament_extraction_prompt():
    return _load_single_prompt_file(
        _TEMPERAMENT_PROMPT_FILE,
        "temperament_extraction_system_prompt.txt",
        _FALLBACK_TEMPERAMENT_EXTRACTION_PROMPT,
    )


# Cached at import time — used by subprocess scripts (fresh import each run)
DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT = load_default_prompts()
DEFAULT_DIALOGUE_IDENTIFICATION_PROMPT = load_dialogue_identification_prompt()
DEFAULT_TEMPERAMENT_EXTRACTION_PROMPT = load_temperament_extraction_prompt()
