from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
result = model.train(data="4_yolo_dataset/data.yaml", epochs=50, imgsz=640, device=0, name="yolo_experiment")