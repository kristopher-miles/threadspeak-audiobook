"""Shared constants used across ``ProjectManager`` mixins.

Constant groups:
- Chunking and script parsing defaults.
- ASR/proofreading thresholds and batching controls.
- Export trim-cache and silence trimming defaults.
"""

import re

MAX_CHUNK_CHARS = 500
PROOFREAD_DURATION_OUTLIER_SECONDS = 10.0
PROOFREAD_LONG_CHUNK_WORD_THRESHOLD = 25
PROOFREAD_LONG_CHUNK_DURATION_OUTLIER_SECONDS = 25.0
PROOFREAD_SHORT_AUDIO_FORCE_ASR_SECONDS = 2.0
PROOFREAD_BATCH_COMMIT_SIZE = 25
REPAIR_BATCH_COMMIT_SIZE = 25
CHAPTER_HEADING_RE = re.compile(
    r'^(chapter|part|book|volume|prologue|epilogue|introduction|conclusion|act|section)\b',
    re.IGNORECASE,
)
COMMON_PROOFREAD_ABBREVIATIONS = {
    "adm": ("admiral",),
    "approx": ("approximately",),
    "capt": ("captain",),
    "cmdr": ("commander",),
    "col": ("colonel",),
    "dept": ("department",),
    "dr": ("doctor",),
    "etc": ("et", "cetera"),
    "gen": ("general",),
    "gov": ("governor",),
    "jr": ("junior",),
    "lt": ("lieutenant",),
    "mr": ("mister",),
    "mrs": ("missus",),
    "ms": ("miss",),
    "no": ("number",),
    "pres": ("president",),
    "prof": ("professor",),
    "rep": ("representative",),
    "rev": ("reverend",),
    "sen": ("senator",),
    "sgt": ("sergeant",),
    "sr": ("senior",),
    "vs": ("versus",),
}
TRIM_CACHE_VERSION = 4
TRIM_SILENCE_THRESHOLD_DBFS = -50.0
TRIM_MIN_SILENCE_LEN_MS = 150
TRIM_KEEP_PADDING_MS = 40
