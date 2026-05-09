#!/usr/bin/env python3
"""
Windows side: capture system / "what you hear" audio (loopback) and stream to WSL over TCP.

Run in Windows Python (not inside WSL), after WSL receiver is listening.

  pip install numpy sounddevice
  # Standardowy sounddevice z PyPI nie ma WasapiSettings(loopback=...) — wtedy:
  pip install soundcard

  python win_send_system_audio.py --host 127.0.0.1 --port 8765

Jeśli „connection refused”: najpierw uruchom odbiornik w WSL; jeśli dalej błąd — użyj
IP z pierwszej linii `wsl hostname -I` jako --host (localhost nie zawsze forwarduje do WSL).

Optional: --list-devices — indeksy zależą od backendu (soundcard vs sounddevice).
"""
from __future__ import annotations

import argparse
import inspect
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np

if sys.platform != "win32":
    print("This sender must run on Windows (loopback capture).", file=sys.stderr)
    sys.exit(1)

import sounddevice as sd  # noqa: E402


def _wasapi_supports_loopback() -> bool:
    try:
        return "loopback" in inspect.signature(sd.WasapiSettings).parameters
    except (TypeError, ValueError):
        return False


def _try_import_soundcard():
    try:
        import soundcard as sc  # type: ignore[import-untyped]

        return sc
    except ImportError:
        return None


def _soundcard_loopback_microphone(sc, speaker):
    """
    Na Windowsie loopback to „mikrofon” powiązany z danym wyjściem — nie speaker.recorder().
    """
    name = str(speaker.name)
    kw = {"include_loopback": True}
    try:
        return sc.get_microphone(id=name, **kw)
    except TypeError:
        pass
    try:
        return sc.get_microphone(name, **kw)
    except TypeError:
        pass
    sid = getattr(speaker, "id", None)
    if sid is not None:
        try:
            return sc.get_microphone(id=sid, **kw)
        except TypeError:
            return sc.get_microphone(sid, **kw)
    raise RuntimeError("get_microphone(..., include_loopback=True) nie działa w tej wersji soundcard")


def _connect_tcp(host: str, port: int, wait_sec: int) -> socket.socket:
    """Łączy z serwerem WSL; przy ConnectionRefused ponawia i podpowiada IP WSL."""
    deadline = time.monotonic() + wait_sec
    attempt = 0
    print(
        f"Łączenie z {host}:{port} (max {wait_sec}s)…\n"
        "  W WSL najpierw: cd …/audio2yolo && python scripts/realtime_bridge/wsl_recv_and_infer.py …\n"
        "  Jeśli nadal odmowa: w PowerShell → wsl -e hostname -I  → użyj pierwszego IP jako --host",
        flush=True,
    )
    while time.monotonic() < deadline:
        attempt += 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if attempt > 1:
                print("Połączono.", flush=True)
            return sock
        except (ConnectionRefusedError, OSError) as e:
            try:
                sock.close()
            except OSError:
                pass
            if attempt == 1 or attempt % 10 == 0:
                print(
                    f"  [{attempt}] {e!s} — czekam na serwer WSL…",
                    flush=True,
                )
            time.sleep(1.0)
    print(
        f"Brak połączenia z {host}:{port} po {wait_sec}s.\n"
        "Sprawdź firewall, uruchom odbiornik w WSL, albo:\n"
        "  wsl -e hostname -I\n"
        "i uruchom ten skrypt z --host <pierwszy_adres>.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows: system audio loopback -> TCP to WSL")
    parser.add_argument("--host", default="127.0.0.1", help="WSL receiver address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--blocksize", type=int, default=4096, help="Frames per block")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--wait-connect",
        type=int,
        default=90,
        help="Sekundy ponawiania TCP zanim wyjście (domyślnie 90)",
    )
    args = parser.parse_args()

    use_soundcard = not _wasapi_supports_loopback()
    sc = _try_import_soundcard() if use_soundcard else None

    if args.list_devices:
        if use_soundcard:
            if sc is None:
                print("Zainstaluj: py -m pip install soundcard", file=sys.stderr)
                sys.exit(1)
            print("SoundCard — głośniki (indeks = --output-device; nagrywane jest wyjście systemowe):")
            for i, speaker in enumerate(sc.all_speakers()):
                print(f"  {i}: {speaker.name}")
        else:
            print(sd.query_devices())
        return

    if use_soundcard and sc is None:
        print(
            "Ten build sounddevice nie obsługuje WasapiSettings(loopback=True).\n"
            "Zainstaluj backend loopback:\n"
            "  py -m pip install soundcard",
            file=sys.stderr,
        )
        sys.exit(1)

    sock = _connect_tcp(args.host, args.port, args.wait_connect)

    def send_chunk(mono: np.ndarray) -> None:
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        payload = mono.tobytes()
        packet = struct.pack("<I", mono.shape[0]) + payload
        sock.sendall(packet)

    try:
        if sc is not None:
            speakers = sc.all_speakers()
            if not speakers:
                print("Brak urządzeń wyjścia (SoundCard).", file=sys.stderr)
                sys.exit(1)
            if args.output_device is not None:
                if args.output_device < 0 or args.output_device >= len(speakers):
                    print(f"Zły indeks {args.output_device}, dostępne 0..{len(speakers) - 1}", file=sys.stderr)
                    sys.exit(1)
                speaker = speakers[args.output_device]
            else:
                speaker = sc.default_speaker()

            try:
                loopback_mic = _soundcard_loopback_microphone(sc, speaker)
            except Exception as e:
                print(
                    f"Nie udało się otworzyć loopback dla „{speaker.name}”: {e}\n"
                    "Spróbuj innego --output-device (--list-devices).",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"SoundCard loopback: {speaker.name} -> {args.host}:{args.port}")
            with loopback_mic.recorder(samplerate=args.sample_rate, blocksize=args.blocksize) as recorder:
                while True:
                    data = recorder.record(numframes=args.blocksize)
                    if data.ndim > 1 and data.shape[1] > 1:
                        mono = np.mean(data, axis=1)
                    else:
                        mono = data.reshape(-1)
                    send_chunk(mono.astype(np.float32, copy=False))
        else:
            wasapi = sd.WasapiSettings(loopback=True)
            default_out = sd.default.device[1]
            device = args.output_device if args.output_device is not None else default_out
            dev_info = sd.query_devices(device)
            out_ch = int(dev_info.get("max_output_channels", 0))
            if out_ch < 1:
                print(
                    f"Device {device} nie jest wyjściem (max_output_channels={out_ch}).",
                    file=sys.stderr,
                )
                sys.exit(1)
            channels = min(2, out_ch) if out_ch >= 2 else 1
            print(f"sounddevice WASAPI loopback: {dev_info.get('name')} -> {args.host}:{args.port}")

            send_lock = threading.Lock()
            q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

            def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
                if status:
                    print("PortAudio status:", status, file=sys.stderr)
                if channels > 1:
                    mono = np.mean(indata, axis=1).astype(np.float32)
                else:
                    mono = indata[:, 0].astype(np.float32)
                try:
                    q.put_nowait(mono.copy())
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait(mono.copy())

            with sd.InputStream(
                device=device,
                samplerate=args.sample_rate,
                blocksize=args.blocksize,
                channels=channels,
                dtype="float32",
                callback=callback,
                extra_settings=wasapi,
            ):
                while True:
                    mono = q.get()
                    send_chunk(mono)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
