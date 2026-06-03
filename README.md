# ESP32-S3 Drone Camera — Face Tracking System

A real-time face detection, multi-person tracking, and drone command system built on a **Seeed Studio XIAO ESP32-S3** camera stream. Uses ByteTrack-style tracking with face re-identification to maintain stable IDs across occlusions, and outputs directional drone commands based on a locked target's position in the frame.

---

## Features

- **Multi-person tracking with stable IDs** — ByteTrack-style IoU + Kalman filter assigns each person a persistent ID that survives brief occlusions
- **Face embedding memory (re-ID)** — LBP histogram gallery per track; when a person re-enters frame, they get the same ID they had before
- **Click-to-lock target** — click directly on any face in the video window to designate it as the target
- **Key-to-lock target** — press `0`–`9` to lock the Nth detected face
- **Drone command output** — live directional signals (`left`, `right`, `forward`, `back`, `center`, `pan_left`, `pan_right`) printed to the terminal and overlaid on the feed
- **Search mode** — when the target is lost, the system enters a panning search pattern and auto re-acquires when the target returns
- **QR code reader**, **motion detection**, **telemetry overlay**, **snapshot**, **video recording**, **frame rotation**, **grid/crosshair overlays**
- **Self-healing stream** — background capture thread auto-reconnects on drop

---

## File Structure

```
drone_camera/
├── config.py          ← your ESP32's IP address (you create this)
├── main.py            ← main viewer, UI, and drone command logic
├── tracker.py         ← ByteTracker, FaceEmbedder, DroneCommander, TargetLockManager
├── snapshots/         ← auto-created; stores saved snapshots
└── recordings/        ← auto-created; stores video recordings
```

---

## Requirements

### Hardware
- Seeed Studio XIAO ESP32-S3 Sense (or any ESP32-CAM streaming MJPEG over HTTP)
- Board powered and connected to the same WiFi network as your computer

### Software
- Ubuntu Linux (tested on 22.04 / 24.04)
- Python 3.10+
- OpenCV with contrib modules
- NumPy

---

## Installation

**1. Create the project folder and enter it**
```bash
mkdir drone_camera && cd drone_camera
```

**2. Place the downloaded files**
```bash
mv ~/Downloads/main.py .
mv ~/Downloads/tracker.py .
```

**3. Create `config.py`** — set your ESP32's IP address here
```bash
cat > config.py << 'EOF'
ESP32_IP = "http://192.168.1.8"   # ← change to your ESP32's actual IP
EOF
```

> **Finding your ESP32's IP:** open the Arduino/PlatformIO Serial Monitor while the board boots — it prints its IP. Alternatively run:
> ```bash
> sudo arp-scan --localnet | grep -i esp
> ```

**4. Install Python dependencies**
```bash
pip install opencv-contrib-python numpy
```
On Ubuntu 24+ if you get an "externally managed" error:
```bash
pip install opencv-contrib-python numpy --break-system-packages
```

**5. Verify the board is reachable**
```bash
curl http://192.168.1.8/           # should stream raw JPEG bytes
curl http://192.168.1.8/telemetry  # should return JSON
```

**6. Run**
```bash
python main.py
```

---

## Controls

Printed to the terminal on every launch. Quick reference:

| Key / Action | What it does |
|---|---|
| **Click** on a face | Lock that face as the target |
| `0` – `9` | Lock the Nth visible face as the target |
| `u` | Unlock target, return to DETECT mode |
| `f` | Toggle face detection + tracking overlay |
| `z` | Toggle QR code reader |
| `m` | Toggle motion detection |
| `t` | Toggle telemetry overlay (heap, RSSI, uptime, temp) |
| `o` | Rotate frame 90° clockwise (cycles 0 / 90 / 180 / 270) |
| `g` | Toggle rule-of-thirds grid |
| `c` | Toggle centre crosshair |
| `s` | Save snapshot (from ESP32 `/snapshot` endpoint) |
| `r` | Toggle video recording |
| `1` | Set resolution: SVGA (800×600) |
| `2` | Set resolution: UXGA (1600×1200) |
| `l` | Toggle onboard LED on / off |
| `L` (shift) | Flash LED 5 times |
| `h` | Print controls to terminal |
| `q` | Quit |

---

## How It Works

### State Machine

```
  ┌─────────┐   click / 0-9    ┌─────────┐   target lost   ┌────────┐
  │ DETECT  │ ───────────────► │ TRACKED │ ──────────────► │ SEARCH │
  └─────────┘                  └─────────┘                  └────────┘
       ▲                            ▲                            │
       │        u (unlock)          │     re-acquired / timeout  │
       └────────────────────────────┴────────────────────────────┘
```

- **DETECT** — green corner boxes around all confirmed faces, labelled `#ID`. Click or press a number to lock.
- **TRACKED** — red corner box + crosshair on the target. Drone commands flow based on target position.
- **SEARCH** — blinking overlay, panning command output. System tries to re-acquire via face re-ID. Times out after 10 seconds and returns to DETECT.

### Tracking Pipeline (per frame)

```
Raw JPEG  →  Haar face detection (every 3rd frame)
          →  ByteTracker.update()
               ├─ Kalman predict all active tracks
               ├─ IoU match detections → active tracks
               ├─ Unmatched detections → re-ID against lost track gallery
               └─ Still unmatched → new track (new ID)
          →  FaceEmbedder.add_embedding()  (LBP histogram stored per ID)
          →  TargetLockManager.update()    (state transitions)
          →  DroneCommander.compute()      (directional command)
          →  Draw + display
```

### Drone Command Logic

Commands are derived from the target's position relative to the frame centre:

| Condition | Command |
|---|---|
| Target centre within dead zone (±12% of frame) | `center` |
| Target left of dead zone | `left` |
| Target right of dead zone | `right` |
| Target above dead zone | `forward` (in drone frame) |
| Target below dead zone | `back` |
| Target too small (area < 4% of frame) | `forward` |
| Target too large (area > 25% of frame) | `back` |
| Search mode | alternating `pan_left` / `pan_right` |

Magnitude (0.0–1.0) scales with distance from centre and is printed alongside the direction.

### Face Re-Identification

Each track stores a rolling gallery of up to 8 LBP (Local Binary Pattern) histogram embeddings. When a detection cannot be matched to any active track by IoU, its histogram is compared (cosine similarity) against the gallery of recently lost tracks. If similarity exceeds the threshold (default 0.72), the track is recovered under its original ID instead of being assigned a new one.

---

## ESP32 Firmware Endpoints Expected

| Endpoint | Method | Response |
|---|---|---|
| `/` | GET | MJPEG stream |
| `/snapshot` | GET | Single JPEG image |
| `/telemetry` | GET | JSON: `heap`, `uptime`, `rssi`, `resolution`, `temperature`, `free_psram` |
| `/led?state=on\|off` | GET | JSON: `{"success": true}` |
| `/flash?count=N` | GET | JSON: `{"success": true}` |
| `/res?val=SVGA\|UXGA` | GET | JSON: `{"success": true}` |

---

## Troubleshooting

**Stream not connecting**
- Confirm the board is on and the IP in `config.py` matches
- Run `curl http://<IP>/` — if it hangs, the board isn't reachable
- Check both devices are on the same WiFi subnet

**`TrackerCSRT` not found / import error**
- You likely have `opencv-python` (no contrib). Uninstall it and reinstall:
  ```bash
  pip uninstall opencv-python && pip install opencv-contrib-python
  ```

**Faces not detected**
- The ESP32-CAM feed is low-res and low-contrast; press `f` to confirm tracking is on
- Try moving closer or improving lighting
- Histogram equalisation is applied automatically to improve detection in dim conditions

**IDs keep changing**
- This usually means the detection interval (`DETECT_EVERY = 3`) is too high relative to motion speed. Edit `DETECT_EVERY` in `main.py` to `1` for continuous detection at the cost of higher CPU use.

---

## Project Structure (code)

| File | Key classes |
|---|---|
| `tracker.py` | `KalmanTrack`, `ByteTracker`, `FaceEmbedder`, `DroneCommander`, `TargetLockManager` |
| `main.py` | `Viewer`, `ESP32Client`, `StreamBuffer`, `FrameAnalyzer`, `Recorder`, `FPSCounter`, `TelemetryOverlay` |
| `config.py` | `ESP32_IP` string |

---

## License

MIT — free to use and modify for personal and commercial projects.