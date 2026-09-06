import cv2
import os
from ultralytics import YOLO

# =========================================================
# PATHS
# =========================================================

folder = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    folder,
    "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
)

# =========================================================
# YOLO MODEL
# =========================================================

model = YOLO("yolo11n.pt")

# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# =========================================================
# VEHICLE CLASSES
# =========================================================

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# =========================================================
# STOP LINE
# =========================================================

LINE_P1 = (842, 497)
LINE_P2 = (396, 503)

# =========================================================
# TRAFFIC LIGHT REGION
# =========================================================

LIGHT_X1 = 673
LIGHT_Y1 = 281
LIGHT_X2 = 683
LIGHT_Y2 = 306

# =========================================================
# TRACKING MEMORY
# =========================================================

previous_sides = {}
track_frame_count = {}

# Vehicles that already crossed
crossed_vehicles = set()

# Vehicles that have already been reported as violations
violations = set()

MIN_TRACK_FRAMES = 5

frame_number = 0


# =========================================================
# FUNCTION: WHICH SIDE OF THE LINE?
# =========================================================

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

    return 0


# =========================================================
# FUNCTION: DETECT RED LIGHT
# =========================================================

def is_red_light(frame):

    roi = frame[
        LIGHT_Y1:LIGHT_Y2,
        LIGHT_X1:LIGHT_X2
    ]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # Red HSV ranges
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 100, 100)
    upper_red2 = (179, 255, 255)

    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = mask1 | mask2

    red_pixels = cv2.countNonZero(red_mask)

    total_pixels = roi.shape[0] * roi.shape[1]

    if total_pixels == 0:
        return False

    red_percentage = (
        red_pixels / total_pixels
    ) * 100

    return red_percentage > 10


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # =====================================================
    # TRAFFIC LIGHT
    # =====================================================

    red_light = is_red_light(frame)

    # Draw traffic-light ROI

    cv2.rectangle(
        frame,
        (LIGHT_X1, LIGHT_Y1),
        (LIGHT_X2, LIGHT_Y2),
        (255, 255, 255),
        2
    )

    if red_light:

        light_text = "TRAFFIC LIGHT: RED"

    else:

        light_text = "TRAFFIC LIGHT: NOT RED"

    cv2.putText(
        frame,
        light_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # YOLO TRACKING
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(vehicle_classes.keys()),
        conf=0.3,
        verbose=False
    )

    result = results[0]

    # =====================================================
    # DRAW STOP LINE
    # =====================================================

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

    # =====================================================
    # VEHICLES
    # =====================================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for box, track_id in zip(
            result.boxes,
            track_ids
        ):

            class_id = int(box.cls[0])

            vehicle = vehicle_classes[class_id]

            # Bounding box

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            # Vehicle center

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            center = (
                center_x,
                center_y
            )

            # =================================================
            # TRACK AGE
            # =================================================

            track_frame_count[track_id] = (
                track_frame_count.get(
                    track_id,
                    0
                ) + 1
            )

            # =================================================
            # LINE SIDE
            # =================================================

            current_side = get_side(
                center,
                LINE_P1,
                LINE_P2
            )

            # =================================================
            # CHECK CROSSING
            # =================================================

            crossed = False

            if track_id in previous_sides:

                previous_side = (
                    previous_sides[track_id]
                )

                if (
                    previous_side != current_side
                    and previous_side != 0
                    and current_side != 0
                ):

                    crossed = True

            # =================================================
            # VEHICLE CROSSED
            # =================================================

            if (
                crossed
                and track_frame_count[track_id]
                >= MIN_TRACK_FRAMES
                and track_id not in crossed_vehicles
            ):

                crossed_vehicles.add(track_id)

                print(
                    f"Vehicle crossed | "
                    f"ID: {track_id} | "
                    f"Type: {vehicle} | "
                    f"Frame: {frame_number} | "
                    f"RED: {red_light}"
                )

                # =============================================
                # RED LIGHT VIOLATION
                # =============================================

                if red_light:

                    violations.add(track_id)

                    print(
                        "🚨🚨 RED LIGHT VIOLATION 🚨🚨"
                    )

                    print(
                        f"Vehicle ID: {track_id}"
                    )

                    print(
                        f"Frame: {frame_number}"
                    )

            # Save current side

            previous_sides[track_id] = current_side

            # =================================================
            # DRAW VEHICLE
            # =================================================

            if track_id in violations:

                label = (
                    f"🚨 VIOLATION ID:{track_id}"
                )

            elif track_id in crossed_vehicles:

                label = (
                    f"CROSSED ID:{track_id}"
                )

            else:

                label = (
                    f"{vehicle} ID:{track_id}"
                )

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
                (
                    int(x1),
                    int(y1) - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # =====================================================
    # DISPLAY VIOLATION COUNT
    # =====================================================

    cv2.putText(
        frame,
        f"Violations: {len(violations)}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW VIDEO
    # =====================================================

    cv2.imshow(
        "Red Light Violation Detector",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print()
print("===================================")
print("FINISHED")
print("===================================")

print(
    f"Total vehicles crossing: "
    f"{len(crossed_vehicles)}"
)

print(
    f"Total red-light violations: "
    f"{len(violations)}"
)

print()
print("Violation IDs:")
print(violations)