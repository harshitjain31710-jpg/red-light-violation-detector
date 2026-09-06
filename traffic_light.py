import cv2
import os

folder = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    folder,
    "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

# Your traffic-light ROI
X1, Y1 = 673, 281
X2, Y2 = 683, 306

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Extract traffic-light region
    roi = frame[Y1:Y2, X1:X2]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV
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

    # Percentage of ROI that is red
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    red_percentage = (red_pixels / total_pixels) * 100

    # Decide whether red is active
    if red_percentage > 10:
        state = "RED"
    else:
        state = "NOT RED"

    # Draw ROI
    cv2.rectangle(
        frame,
        (X1, Y1),
        (X2, Y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic Light: {state}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Traffic Light Detector",
        frame
    )

    # Show the actual ROI enlarged
    enlarged_roi = cv2.resize(
        roi,
        None,
        fx=10,
        fy=10,
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        "Traffic Light ROI",
        enlarged_roi
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()