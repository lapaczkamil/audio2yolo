from ultralytics import YOLO

model = YOLO("runs/detect/yolo_experiment-2/weights/best.pt")

source_to_test = "4_yolo_dataset/images/val"

print("Rozpoczynam testowanie modelu...")
results = model.predict(
    source=source_to_test, 
    conf=0.5,
    save=True,
    show=False
)

print("\n[ZAKOŃCZONO] Sprawdź folder 'runs/detect/predict' aby zobaczyć wyniki!")