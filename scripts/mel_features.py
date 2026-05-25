"""
Wspólna konfiguracja mel dla treningu, walidacji i inferencji.
"""
from __future__ import annotations

import os
from typing import Any

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

SR = 44100

def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


# Domyślne wartości
N_FFT = _env_int("AUDIO2YOLO_N_FFT", 2048)
HOP_LENGTH = _env_int("AUDIO2YOLO_HOP_LENGTH", 256)
N_MELS = _env_int("AUDIO2YOLO_N_MELS", 128)
FMIN = _env_float("AUDIO2YOLO_FMIN", 150.0)
FMAX = _env_float("AUDIO2YOLO_FMAX", 10000.0)
TOP_DB = _env_float("AUDIO2YOLO_TOP_DB", 80.0)
PREEMPH_COEF = 0.0 if _env_bool("AUDIO2YOLO_NO_PREEMPH") else _env_float("AUDIO2YOLO_PREEMPH", 0.97)


def apply_mel_argparse_args(args: Any) -> None:
    set_mel_overrides(
        n_fft=getattr(args, "mel_n_fft", None),
        hop_length=getattr(args, "mel_hop_length", None),
        n_mels=getattr(args, "mel_n_mels", None),
        fmin=getattr(args, "mel_fmin", None),
        fmax=getattr(args, "mel_fmax", None),
        top_db=getattr(args, "mel_top_db", None),
        preemph_coef=getattr(args, "mel_preemph", None),
        preemph_disable=True if getattr(args, "mel_no_preemph", False) else None,
    )


def set_mel_overrides(
    *,
    n_fft: int | None = None,
    hop_length: int | None = None,
    n_mels: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    top_db: float | None = None,
    preemph_coef: float | None = None,
    preemph_disable: bool | None = None,
) -> None:
    global N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, TOP_DB, PREEMPH_COEF
    if n_fft is not None:
        N_FFT = int(n_fft)
    if hop_length is not None:
        HOP_LENGTH = int(hop_length)
    if n_mels is not None:
        N_MELS = int(n_mels)
    if fmin is not None:
        FMIN = float(fmin)
    if fmax is not None:
        FMAX = float(fmax)
    if top_db is not None:
        TOP_DB = float(top_db)
    if preemph_disable is True:
        PREEMPH_COEF = 0.0
    elif preemph_coef is not None:
        PREEMPH_COEF = float(preemph_coef)


def mel_config_dict() -> dict[str, Any]:
    return {
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
        "top_db": TOP_DB,
        "preemph_coef": PREEMPH_COEF,
    }


def mel_config_summary() -> str:
    return ", ".join(f"{k}={v}" for k, v in mel_config_dict().items())


def waveform_to_mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros((N_MELS, 1), dtype=np.float32)
    if PREEMPH_COEF:
        y = librosa.effects.preemphasis(y, coef=float(PREEMPH_COEF))
    nyq = sr / 2.0
    fmax_use = min(float(FMAX), nyq - 1.0)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(N_FFT),
        hop_length=int(HOP_LENGTH),
        n_mels=int(N_MELS),
        fmin=float(FMIN),
        fmax=fmax_use,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=float(TOP_DB))


def mel_db_to_rgb_image(mel_db: np.ndarray, sr: int) -> np.ndarray:
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    fmax_use = min(float(FMAX), sr / 2.0 - 1.0)
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=int(HOP_LENGTH),
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        fmin=float(FMIN),
        fmax=fmax_use,
        ax=ax,
    )
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    rgb = image[:, :, :3].copy()
    plt.close(fig)
    return rgb


def audio_to_spectrogram_rgb(y: np.ndarray, sr: int) -> np.ndarray:
    mel_db = waveform_to_mel_db(y, sr)
    return mel_db_to_rgb_image(mel_db, sr)


def save_mel_png(y: np.ndarray, sr: int, output_path: str) -> None:
    mel_db = waveform_to_mel_db(y, sr)
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    fmax_use = min(float(FMAX), sr / 2.0 - 1.0)
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=int(HOP_LENGTH),
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        fmin=float(FMIN),
        fmax=fmax_use,
        ax=ax,
    )
    plt.savefig(output_path, pad_inches=0)
    plt.close(fig)
