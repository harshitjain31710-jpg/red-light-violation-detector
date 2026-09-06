import cv2
import os
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------

folder = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    folder,
    "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
)

# -----------------------------
# Model
# -----------------------------

model = YOLO("yolo11n.pt")

# -----------------------------
# Video
# -----------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# -----------------------------
# Vehicle classes
# -----------------------------

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# -----------------------------
# YOUR STOP LINE
# -----------------------------

LINE_P1 = (842, 497)
LINE_P2 = (396, 503)

# -----------------------------
# Tracking memory
# -----------------------------

previous_sides = {}
track_frame_count = {}
crossed_vehicles = set()

MIN_TRACK_FRAMES = 5

frame_number = 0


# -----------------------------
# Determine which side of line
# a point is on
# -----------------------------

def get_side(point, line_p1, line_p2):

    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    elif value < 0:
        return -1

    else:
        return 0


# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # -------------------------
    # Track vehicles
    # -------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(vehicle_classes.keys()),
        conf=0.3,
        verbose=False
    )

    result = results[0]

    # -------------------------
    # Draw stop line
    # -------------------------

    cv2.line(
        frame,
        LINE_P1,
        LINE_P2,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "STOP LINE",
        (LINE_P2[0], LINE_P2[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -------------------------
    # Process vehicles
    # -------------------------

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(result.boxes, track_ids):

            class_id = int(box.cls[0])
            vehicle = vehicle_classes[class_id]

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            center = (center_x, center_y)

            # -------------------------
            # Count frames for this ID
            # -------------------------

            track_frame_count[track_id] = (
                track_frame_count.get(track_id, 0) + 1
            )

            # -------------------------
            # Determine side of line
            # -------------------------

            current_side = get_side(
                center,
                LINE_P1,
                LINE_P2
            )

            # -------------------------
            # Check crossing
            # -------------------------

            if track_id in previous_sides:

                previous_side = previous_sides[track_id]

                crossed = (
                    previous_side != current_side
                    and current_side != 0
                    and previous_side != 0
                )

                if (
                    crossed
                    and track_frame_count[track_id] >= MIN_TRACK_FRAMES
                    and track_id not in crossed_vehicles
                ):

                    crossed_vehicles.add(track_id)

                    print(
                        f"🚨 VEHICLE CROSSED LINE | "
                        f"ID: {track_id} | "
                        f"Vehicle: {vehicle} | "
                        f"Frame: {frame_number}"
                    )

            # Save current side
            previous_sides[track_id] = current_side

            # -------------------------
            # Draw vehicle
            # -------------------------

            if track_id in crossed_vehicles:

                label = f"{vehicle} ID:{track_id} CROSSED"

            else:

                label = f"{vehicle} ID:{track_id}"

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                center,
                5,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # -------------------------
    # Display
    # -------------------------

    cv2.imshow(
        "Line Crossing Detector",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()

print()
print("Finished!")

print("Vehicles that crossed:")
print(crossed_vehicles)