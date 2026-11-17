# ADAS Vision — Lane Detection and Driver Assistance System

## Overview

ADAS Vision is a Python-based Advanced Driver Assistance System (ADAS) project that uses computer vision to analyze road conditions and detect lane markings from dashcam video.

The goal of this project is to develop a system that can recognize road lane boundaries, determine a vehicle's position within its lane, detect potential lane drifting, and provide visual warnings to simulate real-world driver assistance technology.

The project uses OpenCV and NumPy to process video frames and identify lane markings. Future development will expand the system to include vehicle and pedestrian detection, lane departure warnings, and a visual dashboard.

## Project Goals

The main goal is to understand and implement computer vision techniques used in modern automotive safety systems.

This project focuses on:

- Detecting road lane markings from video.
- Identifying left and right lane boundaries.
- Calculating the center of the driving lane.
- Estimating the vehicle's position relative to lane boundaries.
- Detecting potential lane departures.
- Providing visual driver warnings.
- Detecting nearby vehicles and pedestrians.
- Developing a real-time driver assistance interface.

## Features

### Current Development
- Video input and frame-by-frame processing using OpenCV.
- A first lane-marking demo using grayscale conversion, Gaussian blur, Canny edges, a road region of interest, and probabilistic Hough line segments.

### Planned Features
- Left and right lane boundary detection.
- Lane curvature estimation.
- Vehicle position and lane offset calculation.
- Lane departure warning system.
- Vehicle detection.
- Pedestrian detection.
- Real-time camera processing.
- ADAS monitoring dashboard.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Computer vision and image processing |
| NumPy | Mathematical calculations and image manipulation |
| Matplotlib | Image visualization and debugging |
| YOLO | Planned vehicle and pedestrian detection |
| PyTorch | Planned machine learning development |

## How It Works

The system processes road video through several computer vision stages:

1. **Video Input:** Loads a recorded dashcam video and extracts individual frames.
2. **Grayscale Conversion:** Converts each frame into grayscale to simplify image processing.
3. **Gaussian Blur:** Reduces image noise and unwanted details.
4. **Canny Edge Detection:** Identifies edges and possible lane markings.
5. **Region of Interest:** Focuses processing on the road area where lane markings are expected.
6. **Lane Detection:** Identifies and draws the detected lane boundaries.
7. **Lane Position Estimation:** Calculates the vehicle's position relative to the detected lane.
8. **Driver Warnings:** Identifies potential lane drifting and displays visual warnings.

## Get Started

Begin with a short, front-facing road video with visible lane markings. This first demo draws detected line segments; it does not yet reliably join them into lane boundaries or issue safety warnings.

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run a video file: `python lane_detection.py path/to/road_video.mp4`
4. To try a connected camera instead: `python lane_detection.py 0`. Press `q` to close the preview.

For example, on Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python lane_detection.py path\to\road_video.mp4
```

This is a learning prototype for recorded footage and experimentation, not a driver safety device. The next step is to group line segments into left and right lane estimates and smooth those estimates across frames.


## Development Roadmap

- [x] Create initial project structure.
- [x] Set up Python development environment.
- [x] Implement video input and frame processing.
- [x] Implement a first-pass lane-marking detector.
- [ ] Improve left and right lane boundary detection.
- [ ] Calculate lane center and vehicle position.
- [ ] Implement lane departure warnings.
- [ ] Add vehicle detection.
- [ ] Add pedestrian detection.
- [ ] Develop an ADAS dashboard.
- [ ] Test and improve detection accuracy.

## Future Improvements

Future versions will focus on improving detection accuracy in different driving conditions, including curved roads, nighttime driving, shadows, and poor weather.

Machine learning techniques may also be introduced to improve lane recognition and object detection.

The long-term goal is to create a more complete ADAS simulation that demonstrates how computer vision can contribute to safer driving.
