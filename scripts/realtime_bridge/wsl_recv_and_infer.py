#!/usr/bin/env python3
"""
WSL side: TCP server — receives mono float32 PCM from Windows sender, runs YOLO on each window.

Usage (from repo, after training / with best.pt):
  cd pipelines/audio2yolo
  python scripts/realtime_bridge/wsl_recv_and_infer.py --model runs/detect/yolo_experiment/weights/best.pt

Requires: numpy, librosa, matplotlib, ultralytics, torch
"""
from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Allow import of audio_yolo_infer when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audio_yolo_infer import DEFAULT_LABELS, AudioYoloRunner  # noqa: E402


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("peer closed connection")
        buf.extend(chunk)
    return bytes(buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="WSL: receive system audio from Windows, run YOLO")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/yolo_experiment/weights/best.pt",
        help="Path to best.pt (relative to cwd or absolute)",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Must match sender")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Analysis window length")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated class names (order = model class id)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model not found: {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    runner = AudioYoloRunner(
        model_path=str(model_path.resolve()),
        labels=labels,
        imgsz=args.yolo_imgsz,
        conf=args.yolo_conf,
    )

    window_samples = max(1, int(args.sample_rate * args.window_sec))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    wsl_ip = ""
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        wsl_ip = (out.split() or [""])[0].strip()
    except Exception:
        pass
    print(
        f"Listening on {args.host}:{args.port} — uruchom nadawcę na Windows.\n"
        f"  Windows: --host 127.0.0.1  (jeśli odmowa połączenia, użyj zamiast tego IP WSL poniżej)\n"
        f"  IP WSL (pierwszy): {wsl_ip or '(uruchom: hostname -I)'}\n"
        f"Expecting mono float32 @ {args.sample_rate} Hz, {window_samples} samples per frame.\n",
        flush=True,
    )

    conn, addr = srv.accept()
    print(f"Connected from {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    buffer = np.array([], dtype=np.float32)
    try:
        while True:
            n = struct.unpack("<I", recv_exact(conn, 4))[0]
            if n == 0 or n > 10_000_000:
                print(f"Invalid chunk size {n}, closing.", file=sys.stderr)
                break
            payload = recv_exact(conn, n * 4)
            chunk = np.frombuffer(payload, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= window_samples:
                frame = buffer[:window_samples].copy()
                buffer = buffer[window_samples:]
                t0 = time.perf_counter()
                label, conf = runner.infer(frame, args.sample_rate)
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"[{time.strftime('%H:%M:%S')}] {label:12s}  conf={conf:.3f}  infer={dt:.0f} ms")
    except (EOFError, BrokenPipeError, ConnectionResetError) as e:
        print(f"Stream ended: {e}")
    finally:
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()
