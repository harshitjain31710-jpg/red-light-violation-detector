"""Detect vehicles crossing a stop line while a traffic signal is red.

The stop-line and traffic-light coordinates are intentionally command-line
options: they must be calibrated for each camera angle.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Settings:
    input_video: Path
    model: str
    stop_line: tuple[int, int, int, int]
    light_roi: tuple[int, int, int, int]
    confidence: float
    min_track_frames: int
    display: bool
    output_video: Path | None
    trajectory_csv: Path | None


def parse_rectangle(value: str) -> tuple[int, int, int, int]:
    """Parse `x1,y1,x2,y2` and reject malformed values early."""
    try:
        coordinates = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Coordinates must be integers.") from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("Expected four values: x1,y1,x2,y2")
    return coordinates  # type: ignore[return-value]


def line_side(point: tuple[int, int], line: tuple[int, int, int, int]) -> int:
    """Return the side of an infinite line containing the point (-1, 0, or 1)."""
    x, y = point
    x1, y1, x2, y2 = line
    value = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return (value > 0) - (value < 0)


def traffic_light_is_red(frame, roi: tuple[int, int, int, int]) -> bool:
    """Classify the selected traffic-light ROI with two HSV red ranges."""
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    high_red = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
    red_fraction = cv2.countNonZero(low_red | high_red) / crop.shape[0] / crop.shape[1]
    return red_fraction > 0.10


def draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect(settings: Settings) -> None:
    cap = cv2.VideoCapture(str(settings.input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {settings.input_video}")

    writer = None
    if settings.output_video:
        settings.output_video.parent.mkdir(parents=True, exist_ok=True)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(str(settings.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    csv_file = None
    csv_writer = None
    if settings.trajectory_csv:
        settings.trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = settings.trajectory_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track_id", "vehicle", "confidence", "center_x", "center_y"])

    model = YOLO(settings.model)
    previous_sides: dict[int, int] = {}
    track_ages: dict[int, int] = {}
    crossed, violations = set(), set()
    frame_number = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            red_light = traffic_light_is_red(frame, settings.light_roi)
            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=list(VEHICLE_CLASSES),
                conf=settings.confidence, verbose=False,
            )
            annotated = frame.copy()
            x1, y1, x2, y2 = settings.stop_line
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.rectangle(annotated, settings.light_roi[:2], settings.light_roi[2:], (255, 255, 255), 2)
            status = "RED" if red_light else "NOT RED"
            draw_label(annotated, f"Traffic light: {status}", (25, 35), (0, 0, 255) if red_light else (0, 255, 0))

            boxes = results[0].boxes
            if boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id.int().cpu().tolist()):
                    class_id = int(box.cls[0])
                    vehicle = VEHICLE_CLASSES.get(class_id, "vehicle")
                    confidence = float(box.conf[0])
                    left, top, right, bottom = map(int, box.xyxy[0].tolist())
                    center = ((left + right) // 2, (top + bottom) // 2)
                    track_ages[track_id] = track_ages.get(track_id, 0) + 1
                    current_side = line_side(center, settings.stop_line)
                    crossed_now = (
                        track_id in previous_sides
                        and current_side != 0
                        and previous_sides[track_id] != 0
                        and current_side != previous_sides[track_id]
                    )
                    if crossed_now and track_ages[track_id] >= settings.min_track_frames and track_id not in crossed:
                        crossed.add(track_id)
                        if red_light:
                            violations.add(track_id)
                            print(f"Violation: {vehicle} ID {track_id} at frame {frame_number}")
                    previous_sides[track_id] = current_side
                    if csv_writer:
                        csv_writer.writerow([frame_number, track_id, vehicle, f"{confidence:.3f}", *center])
                    is_violation = track_id in violations
                    color = (0, 0, 255) if is_violation else (255, 255, 255)
                    cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
                    draw_label(annotated, f"{'VIOLATION ' if is_violation else ''}{vehicle} ID:{track_id}", (left, max(20, top - 10)), color)
                    cv2.circle(annotated, center, 4, color, -1)

            draw_label(annotated, f"Violations: {len(violations)}", (25, 65), (0, 0, 255))
            if writer:
                writer.write(annotated)
            if settings.display:
                cv2.imshow("Red-light violation detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

    print(f"Finished. Vehicles crossed: {len(crossed)} | Red-light violations: {len(violations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a video file")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or local path")
    parser.add_argument("--stop-line", type=parse_rectangle, default=(842, 497, 396, 503), metavar="X1,Y1,X2,Y2")
    parser.add_argument("--light-roi", type=parse_rectangle, default=(673, 281, 683, 306), metavar="X1,Y1,X2,Y2")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--min-track-frames", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--csv", type=Path, help="Optional trajectory CSV output")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window")
    args = parser.parse_args()
    detect(Settings(args.input, args.model, args.stop_line, args.light_roi, args.confidence,
                    args.min_track_frames, not args.no_display, args.output, args.csv))


if __name__ == "__main__":
    main()
