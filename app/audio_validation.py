import re
from dataclasses import asdict, dataclass

import soundfile as sf

DEFAULT_WORDS_PER_MINUTE = 150.0
DEFAULT_DURATION_TOLERANCE_FACTOR = 2.5
DEFAULT_DURATION_OVERHEAD_SECONDS = 0.35
MIN_AUDIO_FILE_SIZE_BYTES = 10 * 1024

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass
class AudioValidationResult:
    is_valid: bool
    error: str | None
    word_count: int
    expected_duration_sec: float
    min_duration_sec: float
    max_duration_sec: float
    actual_duration_sec: float
    file_size_bytes: int
    words_per_minute: float
    tolerance_factor: float

    def to_dict(self):
        return asdict(self)


def count_words(text):
    return len(WORD_RE.findall(text or ""))


def estimate_expected_duration_seconds(
    text=None,
    *,
    word_count=None,
    words_per_minute=DEFAULT_WORDS_PER_MINUTE,
    overhead_seconds=DEFAULT_DURATION_OVERHEAD_SECONDS,
):
    if word_count is None:
        word_count = count_words(text)
    if word_count <= 0:
        return 0.0
    words_per_second = max(words_per_minute / 60.0, 0.1)
    return (word_count / words_per_second) + max(overhead_seconds, 0.0)


def get_audio_duration_seconds(file_path):
    info = sf.info(file_path)
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def validate_audio_clip(
    *,
    text,
    actual_duration_sec,
    file_size_bytes,
    words_per_minute=DEFAULT_WORDS_PER_MINUTE,
    tolerance_factor=DEFAULT_DURATION_TOLERANCE_FACTOR,
):
    word_count = count_words(text)
    expected_duration = estimate_expected_duration_seconds(
        word_count=word_count,
        words_per_minute=words_per_minute,
    )
    min_duration = expected_duration / tolerance_factor if expected_duration > 0 else 0.0
    max_duration = expected_duration * tolerance_factor if expected_duration > 0 else 0.0

    error = None
    if file_size_bytes < MIN_AUDIO_FILE_SIZE_BYTES:
        error = (
            f"Audio file is too small ({file_size_bytes} bytes). "
            f"Anything under {MIN_AUDIO_FILE_SIZE_BYTES} bytes is treated as invalid."
        )
    elif actual_duration_sec <= 0:
        error = "Audio duration is 0 seconds."
    elif word_count > 0 and actual_duration_sec < min_duration:
        error = (
            f"Audio is too short for {word_count} words: "
            f"{actual_duration_sec:.2f}s vs expected {expected_duration:.2f}s "
            f"(minimum {min_duration:.2f}s)."
        )
    elif word_count > 0 and actual_duration_sec > max_duration:
        error = (
            f"Audio is too long for {word_count} words: "
            f"{actual_duration_sec:.2f}s vs expected {expected_duration:.2f}s "
            f"(maximum {max_duration:.2f}s)."
        )

    return AudioValidationResult(
        is_valid=error is None,
        error=error,
        word_count=word_count,
        expected_duration_sec=round(expected_duration, 3),
        min_duration_sec=round(min_duration, 3),
        max_duration_sec=round(max_duration, 3),
        actual_duration_sec=round(float(actual_duration_sec), 3),
        file_size_bytes=int(file_size_bytes),
        words_per_minute=float(words_per_minute),
        tolerance_factor=float(tolerance_factor),
    )
