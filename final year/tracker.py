"""
Face Recognition Tracker — DeepFace Version
=============================================
- Unknown faces  -> GREEN bounding box
- Recognized target face -> RED bounding box

SETUP:
  pip install deepface opencv-python numpy tf-keras

USAGE:
  1. Place a clear frontal photo of your target as 'target_face.jpg'
     in the same folder as this script.
  2. Run: python face_recognition_tracker.py
"""

import cv2
import numpy as np
import os
import sys
from deepface import DeepFace

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TARGET_IMAGE_PATH = r"C:\Users\HP\OneDrive\Desktop\DRONE\final year\target_face.jpg"  # Photo of person to recognize
MODEL_NAME            = "VGG-Face"         # Options: VGG-Face, Facenet, ArcFace
CAMERA_INDEX          = 0                  # 0 = default webcam

# Colors (BGR)
COLOR_UNKNOWN    = (0, 200, 0)    # Green — unknown face
COLOR_RECOGNIZED = (0, 0, 220)    # Red   — target matched
FONT = cv2.FONT_HERSHEY_DUPLEX


# ──────────────────────────────────────────────
# VALIDATE TARGET IMAGE
# ──────────────────────────────────────────────
def validate_target():
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"[ERROR] Target image not found: '{TARGET_IMAGE_PATH}'")
        print("  -> Place a clear frontal photo named 'target_face.jpg' next to this script.")
        sys.exit(1)
    print(f"[OK] Target image found: '{TARGET_IMAGE_PATH}'")


# ──────────────────────────────────────────────
# DRAW BOUNDING BOX WITH CORNER ACCENTS
# ──────────────────────────────────────────────
def draw_box(frame, x, y, w, h, label, color, confidence=None):
    top, left, bottom, right = y, x, y + h, x + w

    # Subtle glow
    overlay = frame.copy()
    cv2.rectangle(overlay, (left-3, top-3), (right+3, bottom+3), color, 2)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Main box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # L-shaped corner accents
    corner_len, thickness = 18, 3
    for (cx, cy, dx, dy) in [
        (left,  top,     1,  1),
        (right, top,    -1,  1),
        (left,  bottom,  1, -1),
        (right, bottom, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

    # Label
    conf_text = f"  {confidence*100:.0f}%" if confidence is not None else ""
    text = f" {label}{conf_text} "
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    label_y = top - 10 if top > 30 else bottom + 10 + th
    cv2.rectangle(frame, (left, label_y - th - 4), (left + tw, label_y + 4), color, cv2.FILLED)
    cv2.putText(frame, text, (left, label_y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# DETECT FACES (OpenCV — fast, runs every frame)
# ──────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces  # list of (x, y, w, h)


# ──────────────────────────────────────────────
# RECOGNIZE FACE (DeepFace)
# ──────────────────────────────────────────────
def is_target(face_img):
    """Returns (match: bool, confidence: float)"""
    try:
        result = DeepFace.verify(
            img1_path=face_img,
            img2_path=TARGET_IMAGE_PATH,
            model_name=MODEL_NAME,
            enforce_detection=False,
            silent=True
        )
        distance   = result["distance"]
        threshold  = result["threshold"]
        match      = result["verified"]
        confidence = max(0.0, 1.0 - (distance / threshold))
        return match, round(confidence, 2)
    except Exception:
        return False, 0.0


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    validate_target()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAMERA_INDEX}")
        sys.exit(1)

    print("[OK] Camera opened. Press 'Q' to quit, 'S' to screenshot.")
    print("[INFO] First recognition may take a few seconds (model loading)...")

    frame_count  = 0
    # Cache last result per face region to avoid lag
    last_results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame_count += 1
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_key = (x // 20, y // 20)  # bucket position for caching

            # Run DeepFace every 5th frame; use cached result otherwise
            if frame_count % 5 == 0 or face_key not in last_results:
                face_crop = frame[y:y+h, x:x+w]
                matched, confidence = is_target(face_crop)
                last_results[face_key] = (matched, confidence)
            else:
                matched, confidence = last_results[face_key]

            if matched:
                label, color = "TARGET", COLOR_RECOGNIZED
            else:
                label, color = "Unknown", COLOR_UNKNOWN

            draw_box(frame, x, y, w, h, label, color,
                     confidence if matched else None)

        # HUD
        cv2.putText(frame,
                    f"Faces: {len(faces)}  |  Model: {MODEL_NAME}  |  Q=quit  S=screenshot",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Face Recognition Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[OK] Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")


if __name__ == "__main__":
    main()