# PredReliance data collector

One-command analysis for a processed Aria eye-camera video.

The tool expects a video with the left and right eye views side by side. It writes blink events and pupil-diameter measurements to a folder on the Desktop.

## Algorithms

- Blink count: Pupil Labs Pupil Core Blink Detector-style binocular confidence-drop detection.
- Pupil diameter: Pupil Labs `pye3d` 3D eye model.

Aria exports no native Pupil Labs 2D confidence stream, so the blink detector uses binocular pupil-detection success as a documented proxy. Treat blink results as an estimate until the Aria eye-camera calibration and a native confidence stream are available.

## Run

```bash
python src/analyze_eye_video.py /path/to/processed_eye_video.mp4
```

By default, results are written to `~/Desktop/<video-name>_blink_pupil/`:

- `eye_analysis_summary.csv`
- `blink_events.csv`
- `eye_analysis_timeseries.csv`

## Environment

Python 3.10+ with OpenCV, NumPy, and `pye3d` is required. The existing DeTeReliance environment can be used directly:

```bash
/Users/rain/Desktop/DeTeReliance/tools/boxer_env/bin/python \
  src/analyze_eye_video.py /path/to/processed_eye_video.mp4
```
