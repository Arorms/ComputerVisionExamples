from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data='dataset/data.yaml',
    epochs=20,
    batch=16,
    imgsz=1024,
    name='yolov8s-football-player-detection',
    project='yolo-football-player-detection',
)