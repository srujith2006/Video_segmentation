# ============================================================
# MOBILE 2 - CRACK DETECTION MODULE
# ============================================================

from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CRACK_MODEL_PATH = BASE_DIR / "models" / "crack" / "weights" / "best.pt"
crack_model = None

# ============================================================
# LOAD MODEL
# ============================================================

def get_crack_model():
    global crack_model

    if crack_model is None:
        if not CRACK_MODEL_PATH.exists():
            return None

        crack_model = YOLO(str(CRACK_MODEL_PATH))

    return crack_model

# ============================================================
# PROCESS FRAME
# ============================================================

def process_hazard_frame(frame):

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    annotated_frame = frame.copy()

    detected_hazards = []

    model = get_crack_model()

    if model is None:
        return (
            annotated_frame,
            detected_hazards
        )

    results = model(
        frame,
        conf=0.4
    )

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            detected_hazards.append(
                {
                    "type": label,
                    "confidence": conf,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
            )

    cv2.putText(
        annotated_frame,
        f"Cracks Detected: {len(detected_hazards)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return (
        annotated_frame,
        detected_hazards
    )
