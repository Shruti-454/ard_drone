# Face Recognition Tracker 🎯

A real-time face recognition system that locks onto a specific target face using your camera.
Unknown faces are highlighted in **green**, and the target face turns **red** upon recognition.

---

## Requirements

- Python 3.13+
- Webcam or Arduino camera module

## Installation
```bash
pip install deepface opencv-python numpy tf-keras
```

## Setup

1. Place a clear frontal photo of your target person in the project folder
2. Rename it to `target_face.jpg`
3. Update the path in `tracker.py` if needed:
```python
   TARGET_IMAGE_PATH = r"C:\Users\HP\OneDrive\Desktop\DRONE\final year\target_face.jpg"
```

## Usage
```bash
python tracker.py
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |

## How It Works

1. OpenCV detects all faces in the frame every tick
2. DeepFace compares each detected face against `target_face.jpg`
3. Match below threshold → **RED** box + confidence %
4. No match → **GREEN** box

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_IMAGE_PATH` | `target_face.jpg` | Path to target photo |
| `MODEL_NAME` | `VGG-Face` | Recognition model (`Facenet`, `ArcFace`) |
| `CAMERA_INDEX` | `0` | Camera source (0 = default webcam) |

## Arduino Integration

Set `USE_ARDUINO = True` in `tracker.py` and update `SERIAL_PORT` to your COM port.
The Arduino should stream JPEG frames over Serial bookended by markers `0xFF 0xAA` (start) and `0xFF 0xBB` (end).

## Notes

- First run downloads the VGG-Face model (~500MB, one time only)
- TensorFlow warnings on startup are harmless and can be ignored
- For better accuracy, use a well-lit frontal photo as your target image
