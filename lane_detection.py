"""A first, classical-computer-vision lane detector for recorded road video."""

import argparse
import random
import time

import cv2
import numpy as np


def detect_lanes(frame: np.ndarray) -> np.ndarray:
    """Fit and draw left/right lane lines from edges and common lane colors."""
    height, width = frame.shape[:2]
    # Use lane-paint color masks. General Canny edges picked up the windshield,
    # wipers, guardrails, and shadows in this particular dashcam clip.
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 190, 0]), np.array([180, 255, 110]))
    yellow = cv2.inRange(hls, np.array([12, 70, 80]), np.array([40, 255, 255]))
    lane_features = cv2.bitwise_or(white, yellow)

    # Include changing road geometry, but stop above the windshield/dashboard.
    detection_top = int(height * 0.45)
    detection_bottom = int(height * 0.78)
    mask = np.zeros_like(lane_features)
    polygon = np.array(
        [[(0, detection_bottom),
          (int(width * 0.25), detection_top),
          (int(width * 0.75), detection_top),
          (width - 1, detection_bottom)]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    roi_features = cv2.bitwise_and(lane_features, mask)

    segments = cv2.HoughLinesP(
        roi_features, rho=1, theta=np.pi / 180, threshold=16,
        minLineLength=max(20, width // 80), maxLineGap=max(40, width // 12),
    )
    lane_segments = []
    if segments is not None:
        # OpenCV builds can return HoughLinesP as either (N, 1, 4) or (N, 4).
        for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
            dy = y2 - y1
            if abs(dy) < 1:
                continue
            slope = (x2 - x1) / dy  # x as a function of y
            midpoint_x = (x1 + x2) / 2
            # Reject near-horizontal vehicle/road edges and very steep noise.
            if not 0.10 <= abs(slope) <= 4.5:
                continue
            is_left = slope < 0 and midpoint_x < width * 0.65
            is_right = slope > 0 and midpoint_x > width * 0.35
            if not (is_left or is_right):
                continue
            lane_segments.append((int(x1), int(y1), int(x2), int(y2)))

    overlay = np.zeros_like(frame)
    # Mark only observed paint segments; don't extrapolate them across glass or road.
    for x1, y1, x2, y2 in lane_segments:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)
    return cv2.addWeighted(frame, 1.0, overlay, 0.9, 0)

def draw_simulated_speed(frame: np.ndarray, speed_mph: int) -> np.ndarray:
    """Add a dashboard-style speed readout; this is not measured from the video."""
    output = frame.copy()
    panel = output.copy()
    cv2.rectangle(panel, (24, 24), (300, 150), (15, 22, 30), -1)
    cv2.addWeighted(panel, 0.78, output, 0.22, 0, output)
    cv2.putText(output, "SIMULATED SPEED", (42, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (180, 210, 220), 2, cv2.LINE_AA)
    cv2.putText(output, str(speed_mph), (42, 124), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (80, 235, 130), 4, cv2.LINE_AA)
    cv2.putText(output, "mph", (145, 124), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (235, 245, 245), 2, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to a road video, or camera index such as 0")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video or camera: {args.source}")

    speed_mph = random.randint(58, 66)
    next_speed_change = time.monotonic() + 2.5
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = time.monotonic()
        if now >= next_speed_change:
            speed_mph = max(45, min(75, speed_mph + random.choice([-2, -1, 0, 1, 2])))
            next_speed_change = now + 2.5
        display = draw_simulated_speed(detect_lanes(frame), speed_mph)
        cv2.imshow("Lane detection (press q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
