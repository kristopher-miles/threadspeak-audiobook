import os
import threading
from typing import Optional


class LocalASRUnavailableError(RuntimeError):
    pass


class LocalASREngine:
    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = "en",
        beam_size: int = 1,
    ):
        self.model_size = (model_size or "small.en").strip()
        self.device = (device or "auto").strip().lower()
        self.compute_type = (compute_type or "auto").strip().lower()
        self.language = (language or "en").strip() or None
        self.beam_size = max(int(beam_size or 1), 1)
        self._model = None
        self._lock = threading.Lock()

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _resolve_compute_type(self, resolved_device: str):
        if self.compute_type != "auto":
            return self.compute_type
        if resolved_device == "cuda":
            return "float16"
        return "int8"

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception as e:
                raise LocalASRUnavailableError(
                    "Local ASR is not available. Install the 'faster-whisper' package first."
                ) from e

            resolved_device = self._resolve_device()
            resolved_compute_type = self._resolve_compute_type(resolved_device)
            self._model = WhisperModel(
                self.model_size,
                device=resolved_device,
                compute_type=resolved_compute_type,
            )
            return self._model

    def transcribe_file(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(file_path or "Missing audio path")

        model = self._ensure_model()
        segments, info = model.transcribe(
            file_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(
            (segment.text or "").strip()
            for segment in segments
            if (segment.text or "").strip()
        ).strip()
        return {
            "text": text,
            "language": getattr(info, "language", self.language),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "model_size": self.model_size,
            "device": self._resolve_device(),
            "compute_type": self._resolve_compute_type(self._resolve_device()),
        }
