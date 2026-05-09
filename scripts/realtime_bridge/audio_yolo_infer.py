"""
Shared mel-spectrogram + YOLO inference (keep aligned with ros2 audio_detector node).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

DEFAULT_LABELS = ["woda", "pozar", "kolumna", "krab", "bomba", "karabin"]


class AudioYoloRunner:
    def __init__(self, model_path: str, labels: List[str], imgsz: int, conf: float):
        if YOLO is None:
            raise RuntimeError("Install ultralytics (and torch) for inference.")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        self.labels = labels if labels else list(DEFAULT_LABELS)
        self.imgsz = max(64, int(imgsz))
        self.conf = float(conf)
        self.model = YOLO(str(path))

    def _audio_to_spectrogram_image(self, audio_frame: np.ndarray, sample_rate: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio_frame,
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax,
        )
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        rgb = image[:, :, :3].copy()
        plt.close(fig)
        return rgb

    def infer(self, audio_frame: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        image_rgb = self._audio_to_spectrogram_image(audio_frame, sample_rate)
        image_bgr = image_rgb[:, :, ::-1]
        results = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results:
            return "unknown", 0.0
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return "unknown", 0.0
        conf_tensor = result.boxes.conf
        cls_tensor = result.boxes.cls
        top_idx = int(np.argmax(conf_tensor.cpu().numpy()))
        conf_val = float(conf_tensor[top_idx].item())
        cls_id = int(cls_tensor[top_idx].item())
        if 0 <= cls_id < len(self.labels):
            return self.labels[cls_id], conf_val
        return f"class_{cls_id}", conf_val
