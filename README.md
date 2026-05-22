Project Hirohito-- Smart Target Tracking System 
Overview
Smart Target Tracker 2 are advanced real-time tactical target tracking systems designed for persistent human target acquisition using live IP camera, RTSP, or drone video streams.

The system combines:

YOLOv8 for real-time human detection
DeepSORT for multi-object tracking
Appearance-based target re-identification
Persistent tactical target lock
Low-latency IP stream synchronization
Automatic stream recovery
Version 7 introduced a redesigned persistent target tracking architecture that enables:

Reliable click-to-lock targeting
Stable target persistence after disappearance
Re-identification after target re-entry
Tactical single-target visualization mode
Background multi-person analysis without visual clutter
Version 7.2 extends the system further by improving:

ESP32/IP camera stream stability
Automatic stream reconnection
MJPEG stream handling
Real-world operational reliability
Low-latency recovery after frame loss
This project is part of Project Silent Reaper, developed under the Skynet-Biogenics initiative.

Key Features
Real-time human detection using YOLOv8
DeepSORT multi-object tracking
Persistent single-target tactical lock
Live IP camera / RTSP stream support
Click-to-lock target acquisition
Automatic target re-identification
Low-latency threaded processing pipeline
Tactical visualization mode
Background multi-target analysis
Automatic stream reconnect system
Apple Silicon GPU acceleration support
Drone-compatible architecture
Project Structure
Project-Silent-Reaper/
│
├── Tracker/
│   └── smart target tracker ver 7/
│       ├── smart_target_tracker7.py
│       ├── smart_target_tracker7.2.py
│       └── README.md
│
└── README.md
Source Code
Version 7
Persistent tactical target tracking implementation:

Tracker/smart target tracker ver 7/smart_target_tracker7.py
GitHub:

https://github.com/Skynet-Biogenics/Project-Silent-Reaper/blob/main/Tracker/smart%20target%20tracker%20ver%207/smart_target_tracker7.py

Version 7.2
Enhanced stability and auto-reconnect implementation:

Tracker/smart target tracker ver 7/smart_target_tracker7.2.py
GitHub:

https://github.com/Skynet-Biogenics/Project-Silent-Reaper/blob/main/Tracker/smart%20target%20tracker%20ver%207/smart_target_tracker7.2.py

Version Differences
Feature	V7	V7.2
Persistent target lock	✅	✅
Automatic re-identification	✅	✅
Tactical single-target display	✅	✅
Background target analysis	✅	✅
Stream auto-reconnect	❌	✅
Improved ESP32 stability	❌	✅
MJPEG stream recovery	❌	✅
IP stream resilience	Medium	High
System Architecture
The tracking pipeline works as follows:

IP Camera / Drone Feed
            ↓
Frame Acquisition
            ↓
YOLOv8 Human Detection
            ↓
DeepSORT Tracking
            ↓
Target Selection (Mouse Click)
            ↓
HSV Appearance Embedding
            ↓
Target Re-Identification
            ↓
Persistent Tactical Target Lock
Requirements
Python 3.9+
IP Camera / RTSP Stream / Drone Camera
macOS / Linux / Windows
Required Python libraries:

ultralytics
deep_sort_realtime
opencv-python
numpy
Installation
Clone the repository:

git clone https://github.com/Skynet-Biogenics/Project-Silent-Reaper.git
cd Project-Silent-Reaper
Create a virtual environment (recommended):

python3 -m venv env
source env/bin/activate
Install dependencies:

python3 -m pip install ultralytics deep_sort_realtime opencv-python numpy
Running the Tracker
Navigate to the tracker directory:

cd Tracker/"smart target tracker ver 7"
Run Version 7:

python3 smart_target_tracker7.py
Run Version 7.2:

python3 smart_target_tracker7.2.py
Tactical Tracking Workflow
Before Target Lock
All detected persons are displayed using green tracking boxes
YOLO continuously analyzes all visible persons
DeepSORT assigns persistent tracking IDs
Operator can manually select a target using mouse click
After Target Lock
Once a target is selected:

The system enters persistent tactical tracking mode
All green boxes disappear
Only the selected target remains visible
The selected target is displayed using a red tactical lock box
Background analysis continues silently
Target Re-Identification
If the target:

leaves the camera frame,
becomes temporarily occluded,
or disappears due to tracking interruption,
the system continues comparing appearance embeddings in the background.

When the target reappears:

the target is automatically re-identified,
the tactical lock is restored,
and the red target box reappears automatically.
V7.2 Stability Improvements
Version 7.2 introduces major stability improvements for ESP32/IP camera systems.

Added Features
Automatic stream reconnection
Better MJPEG handling
Improved IP stream recovery
Reduced crash frequency
Improved frame-loss tolerance
More stable long-duration operation
Stream Recovery Logic
If the stream disconnects temporarily:

Frame lost → Auto reconnect → Continue tracking
This allows the system to recover automatically without restarting the program.

Tactical Visualization Mode
State	Display
Before lock	Green boxes on all persons
After lock	Only locked target visible
Target missing	"SEARCHING TARGET..."
Target reappears	Red target lock restored
Re-Identification Mechanism
The tracker uses runtime appearance-based target memory.

Process:

The selected target is converted into an HSV appearance embedding.
The embedding becomes the temporary runtime target signature.
Every visible person is continuously compared against the stored signature.
The best similarity match is selected.
If similarity exceeds the threshold, target lock is restored automatically.
Unlike traditional systems, this architecture does not require a pre-trained identity dataset for runtime operation.

Camera Optimization
Low-latency optimizations include:

Reduced frame resolution
Threaded YOLO inference
Camera buffer reduction
Lightweight YOLOv8n model usage
Apple Silicon acceleration support
Improved ESP32 compatibility
Recommended ESP32 settings:

Setting	Recommended
Resolution	QVGA
Quality	12–20
WiFi	Strong 2.4GHz signal
Performance
Typical performance on Apple Silicon systems:

Resolution	FPS
416 × 320	~25–30 FPS
640 × 480	~15–20 FPS
Performance depends on:

Network stability
Camera stream quality
Hardware acceleration
Number of visible persons
Limitations
Current limitations include:

HSV histogram ReID remains appearance-based
Similar clothing may reduce accuracy
Severe lighting changes can affect re-identification
Heavy occlusion may temporarily break tracking
Fast camera motion may reduce stability
Future Improvements
Planned upgrades include:

Deep learning ReID models (OSNet / FastReID)
Autonomous drone following
PTZ camera integration
Multi-camera tracking
CUDA acceleration
Edge AI deployment
Kalman trajectory prediction
Real-time telemetry overlay
Target priority classification
Long-range tracking optimization
