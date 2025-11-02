import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from src import web_service


class PrepareUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="web_service_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_wave(self, path: Path, sr: int, duration_s: float = 45.0, channels: int = 2) -> None:
        t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
        audio = []
        for ch in range(channels):
            audio.append(np.sin(2 * np.pi * (ch + 1) * 110.0 * t))
        stacked = np.stack(audio, axis=1)
        sf.write(path, stacked, sr)

    def test_prepare_upload_trims_and_resamples(self) -> None:
        src = self.tmpdir / "upload.wav"
        self._write_wave(src, sr=48000, duration_s=45.0, channels=2)

        trimmed = web_service._prepare_upload(src)

        self.assertEqual(trimmed.name, web_service.TRIMMED_UPLOAD_NAME)
        self.assertFalse(src.exists(), "Original upload should be removed after preparation")

        data, sr = sf.read(trimmed, always_2d=True)
        self.assertEqual(sr, web_service.TARGET_SAMPLE_RATE)
        self.assertGreater(data.shape[0], 0)
        self.assertLessEqual(
            data.shape[0],
            int(web_service.MAX_UPLOAD_DURATION_S * web_service.TARGET_SAMPLE_RATE) + 1,
        )

    def test_prepare_upload_falls_back_to_librosa(self) -> None:
        src = self.tmpdir / "fallback.wav"
        self._write_wave(src, sr=44100, duration_s=5.0, channels=1)

        with mock.patch("src.web_service._read_audio_with_soundfile", return_value=None), \
            mock.patch("src.web_service._load_with_librosa", wraps=web_service._load_with_librosa) as mocked_librosa:
            trimmed = web_service._prepare_upload(src)

        self.assertTrue(mocked_librosa.called, "librosa fallback should run when other decoders fail")
        self.assertFalse(src.exists())
        data, sr = sf.read(trimmed, always_2d=True)
        self.assertEqual(sr, web_service.TARGET_SAMPLE_RATE)
        self.assertGreater(data.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
