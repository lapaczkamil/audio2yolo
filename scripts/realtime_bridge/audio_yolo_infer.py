"""
Shared mel-spectrogram + YOLO inference (keep aligned with ros2 audio_detector node).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mel_features import audio_to_spectrogram_rgb  # noqa: E402

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
        return audio_to_spectrogram_rgb(audio_frame, sample_rate)

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
