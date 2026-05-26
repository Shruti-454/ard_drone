# =========================================================
# SMART TARGET TRACKER V9
# Project Silent Reaper
# ONE PERSON LOCK + RE-IDENTIFICATION
# =========================================================

# =========================================================
# AUTO INSTALL REQUIRED PACKAGES
# =========================================================

import subprocess
import sys

packages = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "deep_sort_realtime": "deep_sort_realtime"
}

for import_name, package_name in packages.items():

    try:
        __import__(import_name)

    except ImportError:

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])

# =========================================================
# IMPORTS
# =========================================================

import cv2
import numpy as np
import time
import threading

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================================================
# CAMERA URL
# =========================================================

ip_camera_url = input(
    "Enter IP Camera Stream URL: "
)

# Example:
# http://192.168.137.104:81/stream

# =========================================================
# LOAD YOLO
# =========================================================

model = YOLO("yolov8n.pt")

# Apple Silicon GPU
try:
    model.to("mps")
except:
    pass

# =========================================================
# FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================================================
# DEEPSORT
# =========================================================

tracker = DeepSort(
    max_age=45,
    n_init=2
)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(ip_camera_url)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():

    print("Failed to connect to stream")

    exit()

# =========================================================
# GLOBAL VARIABLES
# =========================================================

selected_id = None

selected_target_locked = False

target_visible = False

# Adaptive appearance memory
adaptive_memory = []

max_memory_size = 40

# Similarity thresholds
similarity_threshold = 0.82

memory_update_threshold = 0.88

# FPS
fps = 0

frame_counter = 0

start_time = time.time()

# Shared objects
shared_frame = None

detections = []

current_tracks = []

display_frame = None

# Thread lock
frame_lock = threading.Lock()

# Mouse
last_click_time = 0

click_delay = 0.3

# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

def enhance_frame(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    enhanced = cv2.merge((cl, a, b))

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced

# =========================================================
# FEATURE EXTRACTOR
# =========================================================

def get_embedding(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [50, 60],
        [0, 180, 0, 256]
    )

    cv2.normalize(hist, hist)

    return hist.flatten()

# =========================================================
# FACE DETECTION
# =========================================================

def detect_face(person_crop):

    try:

        gray = cv2.cvtColor(
            person_crop,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return len(faces) > 0

    except:
        return False

# =========================================================
# MEMORY COMPARISON
# =========================================================

def compare_with_memory(embedding):

    global adaptive_memory

    if len(adaptive_memory) == 0:
        return 0

    similarities = []

    for memory_embedding in adaptive_memory:

        similarity = cv2.compareHist(
            memory_embedding.astype("float32"),
            embedding.astype("float32"),
            cv2.HISTCMP_CORREL
        )

        similarities.append(similarity)

    return np.mean(similarities)

# =========================================================
# UPDATE MEMORY
# =========================================================

def update_memory(embedding):

    global adaptive_memory

    adaptive_memory.append(embedding)

    if len(adaptive_memory) > max_memory_size:

        adaptive_memory.pop(0)

# =========================================================
# YOLO THREAD
# =========================================================

def yolo_thread():

    global shared_frame
    global detections

    while True:

        if shared_frame is None:
            continue

        frame_copy = shared_frame.copy()

        try:

            results = model(
                frame_copy,
                conf=0.60,
                verbose=False
            )[0]

            temp_detections = []

            if results.boxes is not None:

                for box in results.boxes.data.tolist():

                    x1, y1, x2, y2, score, cls = box

                    # PERSON ONLY
                    if int(cls) != 0:
                        continue

                    width = x2 - x1
                    height = y2 - y1

                    # REMOVE SMALL FALSE DETECTIONS
                    if width < 60 or height < 120:
                        continue

                    temp_detections.append(
                        (
                            [x1, y1, width, height],
                            score,
                            "person"
                        )
                    )

            with frame_lock:

                detections = temp_detections

        except:
            pass

# START THREAD
threading.Thread(
    target=yolo_thread,
    daemon=True
).start()

# =========================================================
# MOUSE CLICK
# =========================================================

def mouse_click(event, x, y, flags, param):

    global selected_id
    global adaptive_memory
    global selected_target_locked
    global current_tracks
    global display_frame
    global last_click_time

    if event == cv2.EVENT_LBUTTONDOWN:

        current_time = time.time()

        if current_time - last_click_time < click_delay:
            return

        last_click_time = current_time

        if display_frame is None:
            return

        for track in current_tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l, t, r, b = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                [l, t, r, b]
            )

            # CLICK INSIDE BOX
            if x1 <= x <= x2 and y1 <= y <= y2:

                crop = display_frame[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                selected_id = track_id

                adaptive_memory = []

                embedding = get_embedding(
                    crop
                )

                update_memory(embedding)

                selected_target_locked = True

                print(
                    f"[TARGET LOCKED] ID: {track_id}"
                )

                break

# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow("Drone Vision")

cv2.setMouseCallback(
    "Drone Vision",
    mouse_click
)

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    # =====================================================
    # AUTO RECONNECT
    # =====================================================

    if not ret:

        print(
            "[WARNING] Stream lost... reconnecting"
        )

        cap.release()

        time.sleep(1)

        cap = cv2.VideoCapture(
            ip_camera_url
        )

        continue

    # =====================================================
    # FPS COUNTER
    # =====================================================

    frame_counter += 1

    elapsed = time.time() - start_time

    if elapsed >= 1:

        fps = frame_counter

        frame_counter = 0

        start_time = time.time()

    # =====================================================
    # RESIZE
    # =====================================================

    frame = cv2.resize(
        frame,
        (416, 320)
    )

    # =====================================================
    # ENHANCE FRAME
    # =====================================================

    frame = enhance_frame(frame)

    # =====================================================
    # SHARE FRAME
    # =====================================================

    shared_frame = frame.copy()

    # =====================================================
    # GET DETECTIONS
    # =====================================================

    with frame_lock:

        current_detections = detections.copy()

    # =====================================================
    # TRACKER
    # =====================================================

    tracks = tracker.update_tracks(
        current_detections,
        frame=frame
    )

    current_tracks = tracks

    # =====================================================
    # RESET TARGET VISIBILITY
    # =====================================================

    target_visible = False

    locked_track_found = False

    # =====================================================
    # BEFORE TARGET LOCK
    # =====================================================

    if not selected_target_locked:

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l, t, r, b = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                [l, t, r, b]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # =====================================================
    # TARGET LOCK MODE
    # =====================================================

    else:

        # =================================================
        # FIRST TRY ORIGINAL TRACK ID
        # =================================================

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # ORIGINAL TARGET FOUND
            if track_id == selected_id:

                locked_track_found = True

                target_visible = True

                l, t, r, b = track.to_ltrb()

                x1, y1, x2, y2 = map(
                    int,
                    [l, t, r, b]
                )

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size != 0:

                    embedding = get_embedding(
                        person_crop
                    )

                    similarity = compare_with_memory(
                        embedding
                    )

                    # UPDATE MEMORY
                    if similarity > memory_update_threshold:

                        update_memory(
                            embedding
                        )

                # DRAW TARGET
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "TARGET LOCK",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"TRACK ID: {track_id}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                break

        # =================================================
        # IF TARGET LOST
        # =================================================

        if not locked_track_found:

            best_similarity = 0

            best_track = None

            # SEARCH ALL TRACKS
            for track in tracks:

                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                l, t, r, b = track.to_ltrb()

                x1, y1, x2, y2 = map(
                    int,
                    [l, t, r, b]
                )

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size == 0:
                    continue

                try:

                    embedding = get_embedding(
                        person_crop
                    )

                    similarity = compare_with_memory(
                        embedding
                    )

                    # FACE BONUS
                    if detect_face(
                        person_crop
                    ):

                        similarity += 0.08

                    # BEST MATCH
                    if similarity > best_similarity:

                        best_similarity = similarity

                        best_track = (
                            track_id,
                            x1,
                            y1,
                            x2,
                            y2,
                            embedding
                        )

                except:
                    pass

            # =================================================
            # RE-IDENTIFICATION
            # =================================================

            if (
                best_track is not None and
                best_similarity > similarity_threshold
            ):

                (
                    selected_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    embedding
                ) = best_track

                target_visible = True

                update_memory(
                    embedding
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "TARGET RE-IDENTIFIED",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"SIMILARITY: {best_similarity:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    # =====================================================
    # SEARCH MODE DISPLAY
    # =====================================================

    if (
        selected_target_locked and
        not target_visible
    ):

        cv2.putText(
            frame,
            "SEARCHING TARGET...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # =====================================================
    # FPS DISPLAY
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =====================================================
    # MEMORY DISPLAY
    # =====================================================

    cv2.putText(
        frame,
        f"MEMORY: {len(adaptive_memory)}",
        (140, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SAVE DISPLAY FRAME
    # =====================================================

    display_frame = frame.copy()

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "Drone Vision",
        frame
    )

    # =====================================================
    # EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()
