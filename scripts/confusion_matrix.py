#!/usr/bin/env python3
"""
Walidacja YOLO na zbiorze z data.yaml — zapis m.in. confusion_matrix.png (wbudowane w Ultralytics).

Uruchom z katalogu pipelines/audio2yolo:

  python scripts/confusion_matrix.py --model runs/detect/yolo_experiment/weights/best.pt

Wyniki: runs/detect/val*/confusion_matrix.png (folder zależy od nazwy projektu / kolejnego runu).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO val + confusion matrix")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/yolo_experiment-2/weights/best.pt",
        help="Ścieżka do best.pt (względem cwd lub absolutna)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="4_yolo_dataset/data.yaml",
        help="data.yaml zbioru",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="np. 0, cpu, cuda:0 — puste = auto",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Katalog projektu Ultralytics",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="val_confusion",
        help="Nazwa runu (podfolder z wykresami)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Brak modelu: {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"Brak data.yaml: {data_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Zainstaluj: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    device = args.device if args.device.strip() else None
    model = YOLO(str(model_path.resolve()))

    print(f"Walidacja: model={model_path} data={data_path} cwd={root}")
    metrics = model.val(
        data=str(data_path.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        device=device,
    )

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        cm = Path(save_dir) / "confusion_matrix.png"
        cm_norm = Path(save_dir) / "confusion_matrix_normalized.png"
        print(f"\nZapisano m.in.:\n  {cm}")
        if cm_norm.is_file():
            print(f"  {cm_norm}")
    else:
        print("\nSprawdź runs/detect/*/ — confusion_matrix.png po walidacji.")


if __name__ == "__main__":
    main()
