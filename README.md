# Red-Light Violation Detector

A computer-vision learning project that uses **YOLO11** and **ByteTrack** to track road vehicles, detect when they cross a configured stop line, and flag crossings that occur while a selected traffic signal appears red.

> This is a learning/demo project, not a production enforcement system. Its HSV traffic-light rule and manually selected coordinates must be recalibrated for every video/camera view.

## Features

- Vehicle detection for cars, motorcycles, buses, and trucks
- Persistent multi-object tracking with ByteTrack
- Stop-line crossing detection based on a vehicle's bounding-box center
- HSV-based red traffic-light detection
- Optional annotated MP4 and vehicle trajectory CSV exports
- Headless mode for non-GUI environments

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Bring your own road video; videos and model weights are excluded from version control.

```bash
python app.py --input path/to/traffic-video.mp4 --output outputs/annotated.mp4 --csv outputs/trajectories.csv
```

The first run downloads `yolo11n.pt` automatically if it is not already available. Press `q` or `Esc` to stop the preview. For a server or terminal-only run, use `--no-display`.

## Camera calibration

The default coordinates are calibrated only for the original learning video. Supply coordinates for your footage:

```bash
python app.py --input my-video.mp4 \
  --stop-line 842,497,396,503 \
  --light-roi 673,281,683,306
```

- `--stop-line X1,Y1,X2,Y2`: two endpoints of the stop line.
- `--light-roi X1,Y1,X2,Y2`: rectangle around the active lamp (top-left to bottom-right).
- `select_line.py` is a small helper that lets you click two stop-line endpoints on the first video frame.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Main, self-contained detector application |
| `select_line.py` | Interactive stop-line coordinate helper |
| `plot_trajectory.py` | Visualizes an exported trajectory CSV |
| `vehicle_trajectories.csv` | Example trajectory output from the learning video |
| `trajectory.py`, `line_crossing.py`, `traffic_light.py`, `red_light_detector.py` | Original learning iterations retained for reference |

## Limitations

Occlusion, tracking ID switches, camera motion, small/distant vehicles, illumination changes, and an imprecise ROI can all cause incorrect results. A real-world deployment needs calibrated cameras, stronger signal classification, direction-aware line logic, evidence capture, and human review.

## License

MIT
