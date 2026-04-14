"""Central structured output contracts for adaptive tool/json execution."""

from __future__ import annotations

from typing import Any, Dict

from .models import StructuredOutputContract


def _entry_item_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "text": {"type": "string"},
            "instruct": {"type": "string"},
        },
        "required": ["speaker", "text", "instruct"],
    }


SCRIPT_ENTRIES_CONTRACT = StructuredOutputContract(
    name="script_entries",
    root_type="array",
    tool_name="submit_script_entries",
    tool_schema={
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": _entry_item_schema(),
            }
        },
        "required": ["entries"],
    },
    tool_instruction=(
        "Return your final answer by calling the submit_script_entries tool once. "
        "Do not return plain text."
    ),
    json_instruction=(
        "Return ONLY a valid JSON array of objects with keys \"speaker\", \"text\", and \"instruct\". "
        "Do not include markdown or explanations."
    ),
    unwrap_field="entries",
)

VOICE_DESCRIPTION_CONTRACT = StructuredOutputContract(
    name="voice_description",
    root_type="object",
    tool_name="submit_voice_description",
    tool_schema={
        "type": "object",
        "properties": {
            "voice": {"type": "string"},
        },
        "required": ["voice"],
    },
    tool_instruction=(
        "Return your final answer by calling the submit_voice_description tool once. "
        "Do not return plain text."
    ),
    json_instruction=(
        "Return ONLY a valid JSON object with exactly one key: {\"voice\":\"...\"}. "
        "Do not include markdown or explanations."
    ),
)

ATTRIBUTION_DECISION_CONTRACT = StructuredOutputContract(
    name="attribution_decision",
    root_type="object",
    tool_name="submit_attribution_decision",
    tool_schema={
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "enum": ["TRUE", "FALSE"],
            },
        },
        "required": ["result"],
    },
    tool_instruction=(
        "Return your final answer by calling the submit_attribution_decision tool once "
        "with result set to TRUE or FALSE. Do not return plain text."
    ),
    json_instruction=(
        "Return ONLY a valid JSON object in this shape: {\"result\":\"TRUE\"} or {\"result\":\"FALSE\"}."
    ),
)

DIALOGUE_SPEAKER_CONTRACT = StructuredOutputContract(
    name="dialogue_speaker",
    root_type="object",
    tool_name="identify_dialogue",
    tool_schema={
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
        },
        "required": ["speaker"],
    },
    tool_instruction=(
        "Return your final answer by calling the identify_dialogue tool once. "
        "Do not return plain text."
    ),
    json_instruction=(
        "Return ONLY a valid JSON object with exactly one key: {\"speaker\":\"...\"}."
    ),
)

SENTIMENT_MOOD_CONTRACT = StructuredOutputContract(
    name="sentiment_mood",
    root_type="object",
    tool_name="identify_sentiment",
    tool_schema={
        "type": "object",
        "properties": {
            "mood": {"type": "string"},
        },
        "required": ["mood"],
    },
    tool_instruction=(
        "Return your final answer by calling the identify_sentiment tool once. "
        "Do not return plain text."
    ),
    json_instruction=(
        "Return ONLY a valid JSON object with exactly one key: {\"mood\":\"...\"}."
    ),
)
