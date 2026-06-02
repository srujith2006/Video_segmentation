
# MAIN DRIVER PROGRAM
import cv2
import numpy as np

from mobile1_survivor import process_survivor_frame
from mobile2_hazard import process_hazard_frame

FRAME_SIZE = (640, 480)
ROOM_MAP_SIZE = (400, 480)
OUTPUT_SIZE = (1280, 720)

# MOBILE IPS
ip1 = "192.168.1.102:8080"
ip2 = "192.168.1.127:8080"

video_url1 = f"http://{ip1}/video"
video_url2 = f"http://{ip2}/video"

gps_url1 = f"http://{ip1}/gps.json"

# VIDEO CAPTURE
cap1 = cv2.VideoCapture(video_url1)
cap2 = cv2.VideoCapture(video_url2)

# VIDEO WRITER
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "advanced_disaster_output.mp4",
    fourcc,
    20.0,
    OUTPUT_SIZE
)

def make_blank_panel(title, message, size=FRAME_SIZE):
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        title,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (180, 180, 180),
        2
    )

    cv2.putText(
        frame,
        message,
        (80, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    return frame

def make_blank_room_map(message):
    width, height = ROOM_MAP_SIZE
    room_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    cv2.rectangle(
        room_map,
        (20, 20),
        (width - 20, height - 20),
        (0, 0, 0),
        2
    )

    cv2.putText(
        room_map,
        message,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    return room_map

# MAIN LOOP
while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # MOBILE 1
    if ret1:
        try:
            (
                survivor_frame,
                room_map,
                human_survivors,
                animal_survivors
            ) = process_survivor_frame(
                frame1,
                gps_url1
            )
        except Exception as error:
            survivor_frame = make_blank_panel(
                "SURVIVOR ANALYSIS",
                f"Mobile 1 error: {type(error).__name__}"
            )
            room_map = make_blank_room_map("Mobile 1 processing error")
    else:
        survivor_frame = make_blank_panel(
            "SURVIVOR ANALYSIS",
            "Mobile 1 not connected"
        )
        room_map = make_blank_room_map("Mobile 1 not connected")

    # MOBILE 2
    if ret2:
        try:
            (
                hazard_frame,
                detected_hazards
            ) = process_hazard_frame(
                frame2
            )
        except Exception as error:
            hazard_frame = make_blank_panel(
                "HAZARD ANALYSIS",
                f"Mobile 2 error: {type(error).__name__}"
            )
    else:
        hazard_frame = make_blank_panel(
            "HAZARD ANALYSIS",
            "Mobile 2 not connected"
        )

    # TITLES

    cv2.putText(
        survivor_frame,
        "SURVIVOR ANALYSIS",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        hazard_frame,
        "HAZARD ANALYSIS",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    
    # COMBINE
    top_row = np.hstack(
        (
            survivor_frame,
            hazard_frame
        )
    )

    room_map = cv2.resize(
        room_map,
        (1280, 240)
    )

    final_output = np.vstack(
        (
            top_row,
            room_map
        )
    )


    # SAVE
  

    out.write(final_output)

    cv2.imshow(
        "Advanced Disaster Response System",
        top_row
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap1.release()
cap2.release()

out.release()

cv2.destroyAllWindows()

print("System Finished Successfully")
