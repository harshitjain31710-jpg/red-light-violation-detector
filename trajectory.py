import cv2
import os
import csv
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------

folder = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    folder,
    "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
)

csv_path = os.path.join(
    folder,
    "vehicle_trajectories.csv"
)

# -----------------------------
# Load YOLO
# -----------------------------

model = YOLO("yolo11n.pt")

# -----------------------------
# Open video
# -----------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# -----------------------------
# Create CSV
# -----------------------------

csv_file = open(csv_path, "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "frame",
    "track_id",
    "vehicle",
    "confidence",
    "center_x",
    "center_y"
])

# Vehicle classes from COCO
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

frame_number = 0

# -----------------------------
# Process video
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(vehicle_classes.keys()),
        conf=0.3,
        verbose=False
    )

    result = results[0]

    # Check if tracking IDs exist
    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(result.boxes, track_ids):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Center of vehicle
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            vehicle = vehicle_classes[class_id]

            # Save trajectory point
            writer.writerow([
                frame_number,
                track_id,
                vehicle,
                round(confidence, 3),
                center_x,
                center_y
            ])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2
            )

            # Draw ID
            label = f"{vehicle} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 255, 255),
                -1
            )

    # Show video
    cv2.imshow("Vehicle Trajectories", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print()
print("Finished!")
print("Trajectory data saved to:")
print(csv_path)