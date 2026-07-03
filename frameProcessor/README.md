# Autonomous Vehicle Perception Utilities

Reusable computer vision utilities for autonomous vehicle perception.

## Project Layout

```text
project/
├── main.py
├── models/
├── external/
├── util/
└── outputs/
```

### util/

| File | Purpose |
|------|---------|
| `image_utils.py` | Image resizing and helper functions |
| `visualization.py` | Draw masks and detections |
| `segmentation.py` | SegFormer and TwinLiteNet segmentation |
| `detection.py` | YOLO object detection and segmentation |
| `lane_detection.py` | Convert lane predictions into OpenCV masks |
| `ufldv2_loader.py` | Load and configure UFLDv2 |

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## UFLDv2 Setup

This project uses the official **Ultra-Fast-Lane-Detection-v2** repository for the network architecture and configuration files, while the trained checkpoint is stored separately.

### 1. Clone the official repository

Clone the repository into the `external/` folder(run command from project directory):

```bash
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git external/Ultra-Fast-Lane-Detection-v2
```

### 2. Download a trained checkpoint

Download a supported UFLDv2 checkpoint (for example `culane_res18.pth`) and place it in:

```text
models/
└── ufldv2/
    └── culane_res18.pth
```

### 3. Expected directory structure

```text
project/
├── models/
│   └── ufldv2/
│       └── culane_res18.pth
├── external/
│   └── Ultra-Fast-Lane-Detection-v2/
│       ├── configs/
│       └── model/
└── util/
```

### 4. Load the model

```python
from util import load_lane_model

lane_model = load_lane_model()
```

The loader automatically:

- finds the UFLDv2 repository
- finds the checkpoint
- loads the correct configuration
- initializes the network
- returns a ready-to-use `UFLDv2LaneDetector`

### Optional

If your repository or checkpoint is stored elsewhere, specify the paths manually:

```python
from util import load_ufldv2_lane_detector

lane_model = load_ufldv2_lane_detector(
    repo_path="path/to/Ultra-Fast-Lane-Detection-v2",
    checkpoint_path="path/to/culane_res18.pth",
    config_path="path/to/culane_res18.py",
)
```

## Video Sources

```python
VIDEO_SOURCE = "/dev/video0"          # Laptop webcam
VIDEO_SOURCE = "/dev/video42"         # Android phone (scrcpy)
VIDEO_SOURCE = "videos/roadVideo.mp4" # Video file
```

## Design

```
models/   -> model weights/checkpoints
external/ -> third-party repositories
util/     -> reusable perception utilities
main.py   -> application/demo entry point
```
