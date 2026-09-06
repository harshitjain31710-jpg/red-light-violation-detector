import cv2
import os

folder = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    folder,
    "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
)

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("Could not read video")
    exit()

cv2.imshow("First Frame", frame)

print("Frame size:", frame.shape)
print("Click two points to define the stop line.")
print("Press Q when finished.")

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Point {len(points)}: ({x}, {y})")

        cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)

        if len(points) == 2:
            cv2.line(
                frame,
                points[0],
                points[1],
                (255, 255, 255),
                3
            )

        cv2.imshow("First Frame", frame)


cv2.setMouseCallback("First Frame", mouse_callback)

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nSelected points:")
print(points)