
# WATER LEAKAGE MODEL TRAINING

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():

    model = YOLO("yolov8m.pt")

    model.train(
        data=str(BASE_DIR / "leakage3.v2i.yolov8.zip"),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="models",
        name="leakage"
    )

if __name__ == "__main__":
    main()
