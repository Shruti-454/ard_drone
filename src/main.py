"""
main.py  –  ESP32-S3 Drone Camera Viewer
=========================================
Features added on top of base viewer:
  • ByteTrack-style multi-person tracking with stable IDs
  • Face embedding memory for re-identification after occlusion/loss
  • Click-to-lock target selection
  • Drone directional command output (printed to terminal + overlaid on frame)
  • Panning search pattern when target is lost
  • Full terminal HUD: live controls reminder on every state transition
"""

import cv2
import urllib.request
import urllib.error
import http.client
import socket
import numpy as np
import time
import json
import os
import threading
from datetime import datetime

from tracker import (
    ByteTracker, FaceEmbedder, DroneCommander, DroneCommand,
    TargetLockManager, TrackResult
)

# ── Try importing config, fall back gracefully ────────────────────────────────
try:
    from config import ESP32_IP
except ImportError:
    ESP32_IP = "http://192.168.1.8"   # fallback

BASE_URL       = ESP32_IP.rstrip("/")
SNAPSHOT_DIR   = "snapshots"
RECORDING_DIR  = "recordings"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN      = (30,  230,  30)
RED        = (30,   30, 220)
YELLOW     = (0,  220,  255)
CYAN       = (220, 200,   0)
ORANGE     = (0,  140,  255)
WHITE      = (255, 255, 255)
BLACK      = (0,   0,    0)
GREY       = (160, 160, 160)

# Distinct per-ID colours (cycles for ID > len)
ID_PALETTE = [
    (255, 180,  50), (50, 255, 180), (180,  50, 255),
    (50,  220, 255), (255,  50, 180), (180, 255,  50),
    (255, 255,  50), (50, 150, 255), (255, 100, 100),
    (100, 255, 100),
]

def id_color(tid: int) -> tuple:
    return ID_PALETTE[tid % len(ID_PALETTE)]

# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

CONTROLS_TEXT = """
╔══════════════════════════════════════════════════════════╗
║           ESP32-S3 DRONE CAMERA  –  CONTROLS            ║
╠══════════════════════════════════════════════════════════╣
║  CLICK  on a face in the video window to lock target     ║
║  0-9    Lock the face labelled #N (shown in frame)       ║
║  u      Unlock target / return to DETECT mode            ║
║  f      Toggle face-detection + tracking overlay         ║
║  z      Toggle QR-code reader                            ║
║  m      Toggle motion detection                          ║
║  t      Toggle telemetry overlay                         ║
║  o      Rotate frame 90° CW                              ║
║  g      Toggle rule-of-thirds grid                       ║
║  c      Toggle centre crosshair                          ║
║  s      Save snapshot                                    ║
║  r      Toggle video recording                           ║
║  1/2    Resolution: SVGA / UXGA                          ║
║  l      Toggle LED  │  L  Flash LED                      ║
║  q      Quit                                             ║
╚══════════════════════════════════════════════════════════╝
"""

def print_controls():
    print(CONTROLS_TEXT)

def print_state_banner(state: str, target_id=None, cmd: DroneCommand = None):
    ts = datetime.now().strftime("%H:%M:%S")
    if state == "detect":
        print(f"[{ts}]  STATE: DETECT  –  CLICK a face or press 0-9 to lock target")
    elif state == "tracked":
        cmd_str = str(cmd) if cmd else ""
        print(f"[{ts}]  STATE: TRACKED  id={target_id}  │  {cmd_str}")
    elif state == "search":
        print(f"[{ts}]  STATE: SEARCH   id={target_id}  –  panning camera …")

_last_state_print   = ""
_last_cmd_print     = ""
_last_print_time    = 0.0

def maybe_print_state(state, target_id, cmd: DroneCommand):
    """Rate-limited terminal state output (max 4 Hz)."""
    global _last_state_print, _last_cmd_print, _last_print_time
    now = time.time()
    cmd_str = str(cmd) if cmd else ""
    if (state != _last_state_print or cmd_str != _last_cmd_print) and (now - _last_print_time > 0.25):
        print_state_banner(state, target_id, cmd)
        _last_state_print = state
        _last_cmd_print   = cmd_str
        _last_print_time  = now

# ─────────────────────────────────────────────────────────────────────────────
# ESP32 HTTP Client
# ─────────────────────────────────────────────────────────────────────────────

class ESP32Client:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def send_command(self, endpoint: str) -> dict:
        try:
            resp = urllib.request.urlopen(f"{self.base_url}{endpoint}", timeout=5)
            return json.loads(resp.read().decode())
        except Exception as e:
            return {"success": False, "error": str(e)}

    def toggle_led(self, state: str):      return self.send_command(f"/led?state={state}")
    def flash_led(self, count: int = 5):   return self.send_command(f"/flash?count={count}")
    def set_resolution(self, val: str):    return self.send_command(f"/res?val={val}")
    def get_telemetry(self) -> dict:       return self.send_command("/telemetry")

    def get_snapshot(self) -> bytes | None:
        try:
            return urllib.request.urlopen(f"{self.base_url}/snapshot", timeout=5).read()
        except Exception:
            return None

    def get_stream(self):
        return urllib.request.urlopen(self.base_url + "/", timeout=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Stream Buffer
# ─────────────────────────────────────────────────────────────────────────────

class StreamBuffer:
    def __init__(self):
        self.buffer = b""

    def feed(self, data: bytes):
        self.buffer += data

    def get_frame(self):
        a = self.buffer.find(b"\xff\xd8")
        b = self.buffer.find(b"\xff\xd9")
        if a != -1 and b != -1:
            if a < b:
                jpg = self.buffer[a:b+2]
                self.buffer = self.buffer[b+2:]
                return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                self.buffer = self.buffer[a:]
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Frame Analyzer  (QR + motion – face done by ByteTracker)
# ─────────────────────────────────────────────────────────────────────────────

class FrameAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.qr_detector = cv2.QRCodeDetector()
        self.prev_gray = None
        self.motion_threshold = 5000

    def detect_faces_raw(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(50,50))
        return [tuple(int(v) for v in f) for f in faces] if len(faces) else []

    def read_qr(self, frame):
        data, _, _ = self.qr_detector.detectAndDecode(frame)
        return data if data else None

    def detect_motion(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21,21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, None
        delta  = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.dilate(cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        self.prev_gray = gray
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > self.motion_threshold for c in cnts)
        return motion, thresh if motion else None

    def draw_motion_contours(self, frame, thresh):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > self.motion_threshold:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w,y+h), ORANGE, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self, out_dir):
        self.out_dir  = out_dir
        self.recording = False
        self.writer    = None
        self.filename  = None

    def start(self, frame):
        if self.recording: return
        self.filename = os.path.join(self.out_dir,
            f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
        h, w = frame.shape[:2]
        self.writer   = cv2.VideoWriter(self.filename,
                            cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w,h))
        self.recording = True
        print(f"[REC] Started: {self.filename}")

    def write_frame(self, frame):
        if self.recording and self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        if self.filename:
            print(f"[REC] Saved: {self.filename}")
        self.filename = None

# ─────────────────────────────────────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self):
        self.prev_time = 0
        self.smooth_fps = 0.0

    def update(self):
        now = time.time()
        dt  = max(now - self.prev_time, 1e-3)
        fps = 1.0 / dt
        self.prev_time  = now
        self.smooth_fps = self.smooth_fps * 0.9 + fps * 0.1
        return self.smooth_fps

# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Overlay
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryOverlay:
    def __init__(self, client):
        self.client    = client
        self.data      = {}
        self.last_fetch = 0

    def update(self):
        if time.time() - self.last_fetch < 2: return
        self.last_fetch = time.time()
        try: self.data = self.client.get_telemetry()
        except Exception: pass

    def draw(self, frame):
        if not self.data: return
        items = [
            (f"HEAP  {self.data.get('heap','--')}",      YELLOW),
            (f"Up    {self.data.get('uptime','--')}s",   YELLOW),
            (f"RSSI  {self.data.get('rssi','--')} dBm",  YELLOW),
            (f"Res   {self.data.get('resolution','--')}", YELLOW),
            (f"Temp  {self.data.get('temperature','--')}°C", YELLOW),
        ]
        y, lh = 48, 18
        _panel(frame, 8, y-4, 200, len(items)*lh+6, 0.5)
        cy = y + 2
        for text, color in items:
            _text_sm(frame, text, 14, cy+10, color)
            cy += lh

# ─────────────────────────────────────────────────────────────────────────────
# Minimal HUD helpers (inline, no class overhead)
# ─────────────────────────────────────────────────────────────────────────────
def draw_drone_cmd(frame, cmd: DroneCommand):
    if cmd is None:
        return

    h, w = frame.shape[:2]

    direction_arrows = {
        "left": "◄ LEFT",
        "right": "RIGHT ►",
        "forward": "▲ FWD",
        "back": "▼ BACK",
        "center": "● CENTER",
        "pan_left": "↺ PAN LEFT",
        "pan_right": "↻ PAN RIGHT",
        "searching": "⟳ SEARCHING",
    }

    label = direction_arrows.get(cmd.direction, cmd.direction.upper())

    mag_pct = f" {int(cmd.magnitude * 100)}%"
    full = label + (mag_pct if cmd.magnitude > 0 else "")

    tw, th = cv2.getTextSize(full, FONT_SM, 0.65, 2)[0]

    x = w // 2 - tw // 2
    y = h - 18

    _panel(frame, x - 8, y - th - 8, tw + 16, th + 12, 0.65)

    colors = {
        "left": "#left",
        "right": "#right",
        "forward": "#fwd",
        "back": "#back",
        "center": "#ctr",
        "pan_left": "#pan",
        "pan_right": "#pan",
        "searching": "#src",
    }

    col_map = {
        "#left": CYAN,
        "#right": CYAN,
        "#fwd": GREEN,
        "#back": ORANGE,
        "#ctr": GREEN,
        "#pan": YELLOW,
        "#src": WHITE,
    }

    col = col_map.get(colors.get(cmd.direction, "#src"), WHITE)

    cv2.putText(
        frame,
        full,
        (x + 1, y + 1),
        FONT_SM,
        0.65,
        BLACK,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        full,
        (x, y),
        FONT_SM,
        0.65,
        col,
        2,
        cv2.LINE_AA,
    )
# ─────────────────────────────────────────────────────────────────────────────
# Per-track drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_tracks(frame, tracks: list, lock_mgr: TargetLockManager):
    for t in tracks:
        if not t.confirmed: continue
        x, y, w, h = t.bbox
        is_target = (t.id == lock_mgr.target_id)
        color     = RED if is_target else id_color(t.id)
        thick     = 3 if is_target else 2
        draw_corner_box(frame, x, y, w, h, color, thick)

        # ID label
        label = f"#{t.id}" + (" [TARGET]" if is_target else "")
        lw, lh = cv2.getTextSize(label, FONT_SM, 0.5, 1)[0]
        _panel(frame, x, y-lh-10, lw+8, lh+6, 0.6)
        cv2.putText(frame, label, (x+3, y-4), FONT_SM, 0.5, color, 2, cv2.LINE_AA)

        # Crosshair on target
        if is_target:
            cx, cy = x+w//2, y+h//2
            cv2.drawMarker(frame, (cx,cy), RED,
                           cv2.MARKER_CROSS, markerSize=22, thickness=2)

    # Detect mode: show index hints
    if lock_mgr.state == "detect":
        idx = 0
        for t in tracks:
            if not t.confirmed: continue
            x, y, w, h = t.bbox
            hint = f"[{idx}]"
            cv2.putText(frame, hint, (x, y+h+14), FONT_SM, 0.5, GREEN, 1, cv2.LINE_AA)
            idx += 1
            if idx >= 10: break

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER  (main class)
# ─────────────────────────────────────────────────────────────────────────────

class Viewer:
    WINDOW = "ESP32-S3 Drone Camera"
    DETECT_EVERY = 3

    def __init__(self):
        self.client    = ESP32Client(BASE_URL)
        print("Setting default resolution to SVGA …")
        self.client.set_resolution("SVGA")

        self.buf       = StreamBuffer()
        self.analyzer  = FrameAnalyzer()
        self.recorder  = Recorder(RECORDING_DIR)
        self.fps_ctr   = FPSCounter()
        self.telemetry = TelemetryOverlay(self.client)

        # Tracking stack
        self.embedder  = FaceEmbedder()
        self.tracker   = ByteTracker(self.embedder)
        self.lock_mgr  = TargetLockManager()
        self.commander: DroneCommander | None = None   # created after first frame

        # State flags
        self.enable_face     = True
        self.enable_qr       = False
        self.enable_motion   = False
        self.show_telemetry  = False
        self.led_on          = False
        self.running         = True
        self.rotation_angle  = 0
        self.show_grid       = False
        self.show_crosshair  = False

        # Recording state
        self._rec_start_time = 0.0
        self._stream_recording = False

        # Threading
        self.latest_frame  = None
        self.new_frame_rdy = False
        self.frame_lock    = threading.Lock()
        self._cap_thread   = threading.Thread(
                                target=self._capture_loop, daemon=True)

        # Animation counters
        self._flash_tick = 0
        self._frame_idx  = 0
        self._tracks: list[TrackResult] = []

        # Mouse state
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_clicked = False

    # ── Capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        print(f"Connecting to {BASE_URL} …")
        while self.running:
            stream = None
            try:
                stream = self.client.get_stream()
                print("Stream connected.\n")
                while self.running:
                    try:
                        data = stream.read(65536)
                        if not data: break
                        self.buf.feed(data)
                        while True:
                            frame = self.buf.get_frame()
                            if frame is None: break
                            with self.frame_lock:
                                self.latest_frame  = frame
                                self.new_frame_rdy = True
                    except (urllib.error.URLError, ConnectionError,
                            http.client.IncompleteRead,
                            http.client.RemoteDisconnected, socket.timeout):
                        print("Connection lost, reconnecting …")
                        break
                    except Exception as e:
                        print(f"Stream error: {e}")
                        break
            except Exception as e:
                print(f"Connect failed: {e}  (retry in 2s …)")
            finally:
                if stream:
                    try: stream.close()
                    except Exception: pass
            time.sleep(2)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_x = x
            self._mouse_y = y
            self._mouse_clicked = True

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        print_controls()
        self._cap_thread.start()

        cv2.namedWindow(self.WINDOW)
        cv2.setMouseCallback(self.WINDOW, self._mouse_cb)

        while self.running:
            with self.frame_lock:
                if self.new_frame_rdy and self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    self.new_frame_rdy = False
                else:
                    frame = None

            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                self._handle_key(key, None)
                continue

            fps = self.fps_ctr.update()
            self._frame_idx += 1

            # Rotation
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h_fr, w_fr = frame.shape[:2]

            # Lazily create DroneCommander once we know frame size
            if self.commander is None:
                self.commander = DroneCommander(w_fr, h_fr)
            else:
                self.commander.update_frame_size(w_fr, h_fr)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)

            # ── Face detection + ByteTracking ──────────────────────────────
            if self.enable_face:
                if self._frame_idx % self.DETECT_EVERY == 0:
                    dets = self.analyzer.detect_faces_raw(frame)
                else:
                    dets = []     # tracker predicts between detections
                    # still feed empty list so Kalman steps forward
                self._tracks = self.tracker.update(dets if dets else [], gray)

            # ── Mouse click → lock ──────────────────────────────────────────
            if self._mouse_clicked:
                self._mouse_clicked = False
                locked = self.lock_mgr.lock_by_click(
                    self._mouse_x, self._mouse_y, self._tracks)
                if locked is None:
                    print("[CLICK] No confirmed face at that position.")

            # ── Lock manager update ─────────────────────────────────────────
            # (updates state machine, detects loss)
            _dummy_cmd = self.lock_mgr.update(self._tracks)

            # ── Drone command ───────────────────────────────────────────────
            if self.lock_mgr.state == "tracked":
                bbox_now = self.tracker.get_bbox_for_id(self.lock_mgr.target_id)
                drone_cmd = self.commander.compute(bbox_now, "tracked")
                drone_cmd.target_id = self.lock_mgr.target_id
            elif self.lock_mgr.state == "search":
                drone_cmd = self.commander.compute(None, "search")
                drone_cmd.target_id = self.lock_mgr.target_id
            else:
                drone_cmd = DroneCommand(direction="searching", magnitude=0.0)

            # Terminal output
            maybe_print_state(self.lock_mgr.state, self.lock_mgr.target_id, drone_cmd)

            # ── Draw ────────────────────────────────────────────────────────
            if self.enable_face:
                draw_tracks(frame, self._tracks, self.lock_mgr)

            if self.lock_mgr.state == "search":
                self._flash_tick += 1
                draw_search_overlay(frame, self._flash_tick)

            if self.enable_qr:
                qr = self.analyzer.read_qr(frame)
                if qr:
                    _text(frame, f"QR: {qr}", 10, h_fr - 14, (255,0,255), 0.55, 2)

            if self.enable_motion:
                motion, thresh = self.analyzer.detect_motion(frame)
                if motion and thresh is not None:
                    self.analyzer.draw_motion_contours(frame, thresh)
                    _text(frame, "MOTION!", w_fr-120, 28, ORANGE, 0.6, 2)

            if self.show_telemetry:
                self.telemetry.update()
                self.telemetry.draw(frame)

            if self.show_grid:
                for i in (1,2):
                    cv2.line(frame,(w_fr*i//3,0),(w_fr*i//3,h_fr),GREY,1)
                    cv2.line(frame,(0,h_fr*i//3),(w_fr,h_fr*i//3),GREY,1)

            if self.show_crosshair:
                cx, cy = w_fr//2, h_fr//2
                cv2.line(frame,(cx-20,cy),(cx+20,cy),WHITE,1)
                cv2.line(frame,(cx,cy-20),(cx,cy+20),WHITE,1)
                cv2.circle(frame,(cx,cy),4,WHITE,1)

            # Drone command banner at bottom
            draw_drone_cmd(frame, drone_cmd)

            # Top bar (state, fps, badges)
            badges = []
            if self.rotation_angle:    badges.append(f"ROT:{self.rotation_angle}°")
            if self.enable_face:       badges.append("TRACK")
            if self.enable_qr:         badges.append("QR")
            if self.enable_motion:     badges.append("MOTION")
            if self.show_telemetry:    badges.append("TELE")
            if self.recorder.recording: badges.append("●REC")
            draw_top_bar(frame, fps, badges, self.lock_mgr.state)

            # Recording
            if not self._stream_recording and self.recorder.recording:
                self.recorder.start(frame)
                self._stream_recording = True
                self._rec_start_time   = time.time()
            elif self._stream_recording and self.recorder.recording:
                self.recorder.write_frame(frame)
            elif self._stream_recording and not self.recorder.recording:
                self._stream_recording = False
                self._rec_start_time   = 0.0

            # Recording timer badge
            if self.recorder.recording and self._rec_start_time:
                el = time.time() - self._rec_start_time
                rt = f"REC {int(el//60):02d}:{int(el%60):02d}"
                rw, rh_ = cv2.getTextSize(rt, FONT_SM, 0.55, 2)[0]
                rx = w_fr - rw - 20
                ry = h_fr - 10
                _panel(frame, rx-6, ry-rh_-8, rw+12, rh_+10, 0.7)
                cv2.putText(frame, rt, (rx,ry), FONT_SM, 0.55, RED, 2, cv2.LINE_AA)

            cv2.imshow(self.WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key, frame)

        self.recorder.stop()
        cv2.destroyAllWindows()
        print("Viewer closed.")

    # ── Key handler ───────────────────────────────────────────────────────────

    def _handle_key(self, key: int, frame):
        if key == 255 or key == -1: return

        if key == ord("q"):
            self.running = False

        elif key == ord("u"):
            self.lock_mgr.unlock()

        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            confirmed = [t for t in self._tracks if t.confirmed]
            if idx < len(confirmed):
                t = confirmed[idx]
                self.lock_mgr.lock_by_id(t.id, self._tracks)
            else:
                print(f"[KEY] No confirmed face #{idx} visible (only {len(confirmed)} shown)")

        elif key == ord("f"):
            self.enable_face = not self.enable_face
            print(f"[FACE TRACKING] {'ON' if self.enable_face else 'OFF'}")

        elif key == ord("z"):
            self.enable_qr = not self.enable_qr
            print(f"[QR] {'ON' if self.enable_qr else 'OFF'}")

        elif key == ord("m"):
            self.enable_motion = not self.enable_motion
            print(f"[MOTION] {'ON' if self.enable_motion else 'OFF'}")

        elif key == ord("t"):
            self.show_telemetry = not self.show_telemetry
            print(f"[TELEMETRY] {'ON' if self.show_telemetry else 'OFF'}")

        elif key == ord("o"):
            self.rotation_angle = (self.rotation_angle + 90) % 360
            print(f"[ROTATE] {self.rotation_angle}°")

        elif key == ord("g"):
            self.show_grid = not self.show_grid
            print(f"[GRID] {'ON' if self.show_grid else 'OFF'}")

        elif key == ord("c"):
            self.show_crosshair = not self.show_crosshair
            print(f"[CROSSHAIR] {'ON' if self.show_crosshair else 'OFF'}")

        elif key == ord("s"):
            raw = self.client.get_snapshot()
            if raw:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
                with open(path, "wb") as f:
                    f.write(raw)
                print(f"[SNAP] Saved: {path}")
            else:
                print("[SNAP] Snapshot failed.")

        elif key == ord("r"):
            if self.recorder.recording:
                self.recorder.stop()
                self._stream_recording = False
            else:
                self.recorder.recording = True  # will .start() on next frame
                print("[REC] Recording will start on next frame.")

        elif key == ord("1"):
            self.client.set_resolution("SVGA")
            print("[RES] SVGA (800x600)")

        elif key == ord("2"):
            self.client.set_resolution("UXGA")
            print("[RES] UXGA (1600x1200)")

        elif key == ord("l"):
            self.led_on = not self.led_on
            self.client.toggle_led("on" if self.led_on else "off")
            print(f"[LED] {'ON' if self.led_on else 'OFF'}")

        elif key == ord("L"):
            self.client.flash_led(5)
            print("[LED] Flash ×5")

        elif key == ord("h"):
            print_controls()

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Viewer().run()