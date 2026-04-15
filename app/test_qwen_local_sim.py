import json
import os
import tempfile
import unittest
import wave

import numpy as np

from e2e_sim.qwen_local_sim import QwenLocalSimProvider


def _write_pcm16_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class QwenLocalSimTests(unittest.TestCase):
    def test_generate_voice_design_supports_audio_wav_path(self):
        with tempfile.TemporaryDirectory(prefix="qwen-sim-test-") as root:
            sr = 24000
            t = np.linspace(0.0, 0.2, int(sr * 0.2), endpoint=False)
            audio = (0.05 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
            wav_path = os.path.join(root, "audio", "voice.wav")
            _write_pcm16_wav(wav_path, audio, sr)

            fixture_path = os.path.join(root, "fixture.json")
            with open(fixture_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "strict": True,
                        "methods": {
                            "generate_voice_design": [
                                {
                                    "expect": {
                                        "text": "Sample text",
                                        "instruct": "Sample description",
                                    },
                                    "audio_wav_path": "audio/voice.wav",
                                }
                            ]
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            provider = QwenLocalSimProvider(fixture_path)
            model = provider.get_model("voice_design")
            wavs, actual_sr = model.generate_voice_design(text="Sample text", instruct="Sample description")

            self.assertEqual(actual_sr, sr)
            self.assertEqual(len(wavs), 1)
            self.assertGreater(len(wavs[0]), 0)
            provider.assert_all_consumed()


if __name__ == "__main__":
    unittest.main()
