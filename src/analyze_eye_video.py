#!/usr/bin/env python3
"""One-command Aria eye-video analysis.

Input: a processed Aria eye video with left and right eye views side by side.
Output: pupil diameter time series/statistics and blink events/count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode


PUPIL_DIAMETER_MIN_MM = 1.0
PUPIL_DIAMETER_MAX_MM = 9.0
BLINK_MIN_DURATION_S = 0.15
BLINK_MAX_DURATION_S = 0.80


def pupil_ellipse(gray: np.ndarray) -> dict | None:
    h, w = gray.shape
    roi_x, roi_y = int(w * 0.10), int(h * 0.15)
    roi = gray[roi_y : int(h * 0.85), roi_x : int(w * 0.90)]
    threshold = min(55.0, float(np.percentile(roi, 25)))
    mask = (roi <= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        (_, _), (axis_a, axis_b), angle = ellipse
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        if major < 15 or major > min(w, h) * 0.35 or minor / major < 0.45:
            continue
        cx, cy = ellipse[0]
        candidates.append((area * minor / major, cx + roi_x, cy + roi_y, major, minor, angle))
    if not candidates:
        return None
    _, cx, cy, major, minor, angle = max(candidates, key=lambda item: item[0])
    return {
        "center": np.asarray([cx, cy], dtype=np.float64),
        "axes": np.asarray([minor, major], dtype=np.float64),
        "angle": float(angle),
    }


def detect_blinks(rows: list[dict], fps: float) -> list[dict]:
    # Pupil Labs' blink detector uses a binocular confidence drop and recovery.
    # Aria exports no pupil-confidence stream, so detection success is used as
    # the conservative confidence proxy for this eye-only video.
    low = np.asarray(
        [not (row["left_pupil_detected"] and row["right_pupil_detected"]) for row in rows],
        dtype=bool,
    )
    min_frames = max(1, round(BLINK_MIN_DURATION_S * fps))
    max_frames = max(min_frames, round(BLINK_MAX_DURATION_S * fps))
    events = []
    start = None
    for index, is_low in enumerate(low):
        if is_low and start is None:
            start = index
        if (not is_low or index == len(low) - 1) and start is not None:
            end = index if is_low and index == len(low) - 1 else index - 1
            length = end - start + 1
            if min_frames <= length <= max_frames:
                events.append({
                    "id": len(events) + 1,
                    "start_time_s": rows[start]["time_s"],
                    "end_time_s": rows[end]["time_s"] + 1.0 / fps,
                    "duration_s": length / fps,
                    "start_frame": start,
                    "end_frame": end,
                })
            start = None
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one processed Aria eye video")
    parser.add_argument("video", type=Path, help="Processed eye video; left/right eye views side by side")
    parser.add_argument("--output-dir", type=Path, help="Output directory (defaults next to the video)")
    args = parser.parse_args()
    output_dir = args.output_dir or Path.home() / "Desktop" / f"{args.video.stem}_blink_pupil"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    split = width // 2
    camera = CameraModel(focal_length=561.5, resolution=[split, height])
    detectors = [
        Detector3D(camera=camera, long_term_mode=DetectorMode.asynchronous),
        Detector3D(camera=camera, long_term_mode=DetectorMode.asynchronous),
    ]

    rows = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        eye_data = []
        for eye_index, eye in enumerate((frame[:, :split], frame[:, split:])):
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            ellipse = pupil_ellipse(gray)
            diameter = ""
            model_confidence = ""
            if ellipse is not None:
                try:
                    result = detectors[eye_index].update_and_detect(
                        {"ellipse": ellipse, "confidence": 1.0, "timestamp": frame_index / fps},
                        gray,
                    )
                    diameter = float(result.get("diameter_3d", ""))
                    model_confidence = float(result.get("model_confidence", ""))
                except Exception:
                    pass
            eye_data.append((ellipse, diameter, model_confidence))
        rows.append({
            "frame": frame_index,
            "time_s": frame_index / fps,
            "left_pupil_detected": int(eye_data[0][0] is not None),
            "right_pupil_detected": int(eye_data[1][0] is not None),
            "left_pupil_diameter_3d_mm": eye_data[0][1],
            "right_pupil_diameter_3d_mm": eye_data[1][1],
            "left_model_confidence": eye_data[0][2],
            "right_model_confidence": eye_data[1][2],
            "left_pupil_x_px": "" if eye_data[0][0] is None else float(eye_data[0][0]["center"][0]),
            "left_pupil_y_px": "" if eye_data[0][0] is None else float(eye_data[0][0]["center"][1]),
            "right_pupil_x_px": "" if eye_data[1][0] is None else float(eye_data[1][0]["center"][0]),
            "right_pupil_y_px": "" if eye_data[1][0] is None else float(eye_data[1][0]["center"][1]),
        })
        frame_index += 1
    cap.release()

    events = detect_blinks(rows, fps)
    left = [r["left_pupil_diameter_3d_mm"] for r in rows if isinstance(r["left_pupil_diameter_3d_mm"], float) and PUPIL_DIAMETER_MIN_MM <= r["left_pupil_diameter_3d_mm"] <= PUPIL_DIAMETER_MAX_MM]
    right = [r["right_pupil_diameter_3d_mm"] for r in rows if isinstance(r["right_pupil_diameter_3d_mm"], float) and PUPIL_DIAMETER_MIN_MM <= r["right_pupil_diameter_3d_mm"] <= PUPIL_DIAMETER_MAX_MM]

    timeseries_path = output_dir / "eye_analysis_timeseries.csv"
    with timeseries_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    events_path = output_dir / "blink_events.csv"
    with events_path.open("w", newline="") as handle:
        fields = ["id", "start_time_s", "end_time_s", "duration_s", "start_frame", "end_frame"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)

    summary_path = output_dir / "eye_analysis_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["video", str(args.video)])
        writer.writerow(["frames", len(rows)])
        writer.writerow(["fps", fps])
        writer.writerow(["blink_algorithm", "Pupil Labs Pupil Core confidence-drop style"])
        writer.writerow(["blink_confidence_source", "Aria binocular pupil-detection proxy"])
        writer.writerow(["blink_count", len(events)])
        writer.writerow(["left_pupil_diameter_3d_mm_mean", np.mean(left) if left else ""])
        writer.writerow(["left_pupil_diameter_3d_mm_median", np.median(left) if left else ""])
        writer.writerow(["right_pupil_diameter_3d_mm_mean", np.mean(right) if right else ""])
        writer.writerow(["right_pupil_diameter_3d_mm_median", np.median(right) if right else ""])
        writer.writerow(["left_pupil_valid_samples", len(left)])
        writer.writerow(["right_pupil_valid_samples", len(right)])

    print(f"blink_count={len(events)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {events_path}")
    print(f"Wrote {timeseries_path}")


if __name__ == "__main__":
    main()
