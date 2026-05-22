import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------
# ENTER IP CAMERA URL
# -------------------------

ip_camera_url = "http://192.168.137.25:81/stream"# Example:
# rtsp://username:password@192.168.1.10:554/stream
# http://192.168.1.10:8080/video

# -------------------------
# Load YOLO
# -------------------------

model = YOLO("yolov8n.pt")

try:
    model.to("mps")  # Apple GPU
except:
    pass

tracker = DeepSort(max_age=30)

# -------------------------
# Camera
# -------------------------

cap = cv2.VideoCapture(ip_camera_url)

# Reduce delay / latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Optional settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check connection
if not cap.isOpened():
    print("Failed to connect to IP camera.")
    exit()

# -------------------------
# Global variables
# -------------------------

selected_id = None
selected_embedding = None

boxes = []
display_frame = None

similarity_threshold = 0.6

last_click_time = 0
click_delay = 0.3

frame_lock = threading.Lock()
shared_frame = None
detections = []

# -------------------------
# Feature extractor
# -------------------------

def get_embedding(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# -------------------------
# YOLO Thread (NON-BLOCKING)
# -------------------------

def yolo_thread():
    global shared_frame, detections

    while True:

        if shared_frame is None:
            continue

        frame_copy = shared_frame.copy()

        results = model(frame_copy, conf=0.4, verbose=False)[0]

        temp_detections = []

        if results.boxes is not None:

            for box in results.boxes.data.tolist():

                x1,y1,x2,y2,score,cls = box

                if int(cls) == 0:
                    temp_detections.append(
                        ([x1,y1,x2-x1,y2-y1], score, "person")
                    )

        with frame_lock:
            detections = temp_detections

# Start YOLO thread
threading.Thread(target=yolo_thread, daemon=True).start()

# -------------------------
# Mouse click
# -------------------------

def mouse_click(event, x, y, flags, param):

    global selected_id
    global selected_embedding
    global boxes
    global display_frame
    global last_click_time

    if event == cv2.EVENT_LBUTTONDOWN:

        current_time = time.time()

        if current_time - last_click_time < click_delay:
            return

        last_click_time = current_time

        if display_frame is None:
            return

        for (x1,y1,x2,y2,track_id) in boxes:

            if x1 <= x <= x2 and y1 <= y <= y2:

                selected_id = track_id

                crop = display_frame[y1:y2, x1:x2]

                if crop.size != 0:
                    selected_embedding = get_embedding(crop)

                print("Selected target:", track_id)
                break

cv2.namedWindow("Drone Vision")
cv2.setMouseCallback("Drone Vision", mouse_click)

# -------------------------
# Main loop
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        break

    frame = cv2.resize(frame, (416,320))

    # Share frame with YOLO thread
    shared_frame = frame.copy()

    # Get detections safely
    with frame_lock:
        current_detections = detections.copy()

    # -------------------------
    # DeepSORT
    # -------------------------

    tracks = tracker.update_tracks(
        current_detections,
        frame=frame
    )

    boxes = []

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l,t,r,b = track.to_ltrb()

        x1,y1,x2,y2 = int(l),int(t),int(r),int(b)

        boxes.append((x1,y1,x2,y2,track_id))

        person_crop = frame[y1:y2,x1:x2]

        # -------------------------
        # Re-identification
        # -------------------------

        if selected_embedding is not None and person_crop.size != 0:

            embedding = get_embedding(person_crop)

            similarity = cv2.compareHist(
                selected_embedding.astype("float32"),
                embedding.astype("float32"),
                cv2.HISTCMP_CORREL
            )

            if similarity > similarity_threshold:
                selected_id = track_id

        # -------------------------
        # Draw
        # -------------------------

        color = (0,255,0)

        if selected_id == track_id:

            color = (0,0,255)

            cv2.putText(
                frame,
                "TARGET LOCK",
                (x1,y1-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Freeze frame for click accuracy
    display_frame = frame.copy()

    cv2.imshow("Drone Vision", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()
