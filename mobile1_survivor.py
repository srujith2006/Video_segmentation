
# MOBILE 1 - SURVIVOR DETECTION MODULE
from ultralytics import YOLO
import cv2
import numpy as np
import requests

# LOAD MODEL
survivor_model = YOLO("yolov8m.pt")

# PROCESS FRAME

def process_survivor_frame(frame, gps_url):

    frame = cv2.resize(frame, (640, 480))

    frame = cv2.bilateralFilter(
        frame,
        9,
        75,
        75
    )

    results = survivor_model(frame)

    annotated_frame = frame.copy()

    # ROOM MAP

    room_map = np.ones(
        (480, 400, 3),
        dtype=np.uint8
    ) * 255

    cv2.rectangle(
        room_map,
        (20, 20),
        (380, 460),
        (0, 0, 0),
        2
    )

    cv2.putText(
        room_map,
        "ESTIMATED SURVIVOR LAYOUT",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # ENTRY

    entry_point = (40, 240)

    cv2.circle(
        room_map,
        entry_point,
        10,
        (255, 0, 0),
        -1
    )

    cv2.putText(
        room_map,
        "ENTRY",
        (55, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

    # GPS FETCH

    gps_available = False
    lat = None
    lon = None

    try:

        gps_data = requests.get(
            gps_url,
            timeout=1
        ).json()

        lat = gps_data["latitude"]
        lon = gps_data["longitude"]

        gps_available = True

    except:
        gps_available = False

    # CLASSES
    human_survivors = []

    animal_survivors = []

    animal_classes = [
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bird"
    ]

    # DETECTIONS
    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            label = survivor_model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # HUMAN
            if label == "person":

                color = (0, 255, 0)

                survivor_type = "HUMAN"

                if gps_available:

                    human_survivors.append(
                        [label, lat, lon]
                    )

                else:

                    human_survivors.append(
                        [label, "GPS N/A", "GPS N/A"]
                    )

            # ANIMAL

            elif label in animal_classes:

                color = (255, 0, 255)

                survivor_type = "ANIMAL"

                if gps_available:

                    animal_survivors.append(
                        [label, lat, lon]
                    )

                else:

                    animal_survivors.append(
                        [label, "GPS N/A", "GPS N/A"]
                    )

            else:
                continue

            # =================================================
            # BOX
            # =================================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated_frame,
                f"{survivor_type}: {label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # =================================================
            # GPS TEXT
            # =================================================

            if gps_available:

                gps_text = (
                    f"Lat:{lat:.5f} "
                    f"Lon:{lon:.5f}"
                )

            else:

                gps_text = "GPS Unavailable"

            cv2.putText(
                annotated_frame,
                gps_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2
            )

            # =================================================
            # ROOM MAP POSITION
            # =================================================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            map_x = int(
                center_x / 640 * 340
            ) + 20

            map_y = int(
                center_y / 480 * 420
            ) + 20

            cv2.circle(
                room_map,
                (map_x, map_y),
                8,
                color,
                -1
            )

            cv2.line(
                room_map,
                entry_point,
                (map_x, map_y),
                (180, 180, 180),
                1
            )

    # ========================================================
    # HUMAN TABLE
    # ========================================================

    y = 80

    cv2.putText(
        room_map,
        "HUMAN SURVIVORS",
        (90, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 150, 0),
        2
    )

    y += 25

    for i, item in enumerate(human_survivors):

        row = f"{i+1}. {item[0]}"

        cv2.putText(
            room_map,
            row,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )

        y += 20

    y += 20

    cv2.putText(
        room_map,
        "ANIMAL SURVIVORS",
        (90, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2
    )

    y += 25

    for i, item in enumerate(animal_survivors):

        row = f"{i+1}. {item[0]}"

        cv2.putText(
            room_map,
            row,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )

        y += 20

    return (
        annotated_frame,
        room_map,
        human_survivors,
        animal_survivors
    )