"""
tracker.py  –  ByteTrack-style multi-person tracker with face re-ID
====================================================================
Provides:
  • ByteTracker  – lightweight IoU + Kalman tracker that assigns stable IDs
  • FaceEmbedder – LBP histogram embeddings stored per track for re-ID
  • DroneCommander – converts target position to directional commands
  • TargetLockManager – click-to-lock + search-state machine
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def iou(boxA, boxB) -> float:
    """Intersection-over-Union between two (x,y,w,h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def center(box) -> tuple:
    x, y, w, h = box
    return (x + w//2, y + h//2)

def bbox_distance(a, b) -> float:
    cx1, cy1 = center(a)
    cx2, cy2 = center(b)
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

# ─────────────────────────────────────────────────────────────────────────────
# Kalman-based single track
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTrack:
    _id_counter = 0

    def __init__(self, bbox: tuple):
        KalmanTrack._id_counter += 1
        self.id = KalmanTrack._id_counter
        self.age = 0
        self.hits = 1
        self.miss = 0
        self.last_bbox = bbox          # (x,y,w,h)
        self.predicted_bbox = bbox

        # State: [cx, cy, w, h, vx, vy]
        self.kf = cv2.KalmanFilter(6, 4)
        x, y, w, h = bbox
        cx, cy = x + w//2, y + h//2

        self.kf.transitionMatrix = np.array([
            [1,0,0,0, 1,0],
            [0,1,0,0, 0,1],
            [0,0,1,0, 0,0],
            [0,0,0,1, 0,0],
            [0,0,0,0, 1,0],
            [0,0,0,0, 0,1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
        ], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self.kf.statePost           = np.array([[cx],[cy],[w],[h],[0],[0]], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        w = max(w, 20); h = max(h, 20)
        self.predicted_bbox = (int(cx-w/2), int(cy-h/2), int(w), int(h))
        self.age += 1
        return self.predicted_bbox

    def update(self, bbox: tuple):
        x, y, w, h = bbox
        cx, cy = x + w//2, y + h//2
        meas = np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        self.kf.correct(meas)
        self.last_bbox = bbox
        self.hits += 1
        self.miss = 0

    def mark_missed(self):
        self.miss += 1


# ─────────────────────────────────────────────────────────────────────────────
# Face Embedder  (LBP histogram – no DNN required)
# ─────────────────────────────────────────────────────────────────────────────

class FaceEmbedder:
    """
    Stores a rolling gallery of LBP histograms per track ID.
    Uses cosine similarity for re-identification.
    """
    GALLERY_SIZE = 8      # keep last N embeddings per ID
    REID_THRESH  = 0.72   # cosine similarity threshold to accept re-ID

    def __init__(self):
        # track_id -> list of np.ndarray embeddings
        self._gallery: dict[int, list] = defaultdict(list)

    @staticmethod
    def _lbp_hist(gray_patch: np.ndarray) -> np.ndarray:
        """Fast uniform LBP histogram on a 64x64 patch."""
        if gray_patch is None or gray_patch.size == 0:
            return np.zeros(256, dtype=np.float32)
        patch = cv2.resize(gray_patch, (64, 64))
        # Manual LBP
        rows, cols = patch.shape
        lbp = np.zeros_like(patch, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(patch, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (patch >= shifted).astype(np.uint8)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,255))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def add_embedding(self, track_id: int, gray_face: np.ndarray):
        emb = self._lbp_hist(gray_face)
        gallery = self._gallery[track_id]
        gallery.append(emb)
        if len(gallery) > self.GALLERY_SIZE:
            gallery.pop(0)

    def mean_embedding(self, track_id: int) -> Optional[np.ndarray]:
        if track_id not in self._gallery or not self._gallery[track_id]:
            return None
        return np.mean(self._gallery[track_id], axis=0)

    def best_match(self, probe_gray: np.ndarray, candidate_ids: list) -> Optional[int]:
        """Return the track_id with highest cosine similarity, or None."""
        probe = self._lbp_hist(probe_gray)
        best_id, best_sim = None, self.REID_THRESH
        for tid in candidate_ids:
            mean = self.mean_embedding(tid)
            if mean is None:
                continue
            sim = float(np.dot(probe, mean) / (np.linalg.norm(probe) * np.linalg.norm(mean) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_id  = tid
        return best_id

    def transfer(self, old_id: int, new_id: int):
        """Move embeddings from old_id to new_id (after re-ID merge)."""
        if old_id in self._gallery:
            self._gallery[new_id].extend(self._gallery.pop(old_id))

    def remove(self, track_id: int):
        self._gallery.pop(track_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# ByteTracker  (IoU matching + Kalman + face re-ID)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    id: int
    bbox: tuple       # (x,y,w,h)
    confirmed: bool   # only show after MIN_HITS
    is_target: bool = False

class ByteTracker:
    MIN_HITS   = 3    # frames before a track is "confirmed"
    MAX_MISS   = 30   # frames before a lost track is deleted
    IOU_THRESH = 0.30 # minimum IoU to associate detection→track

    def __init__(self, embedder: FaceEmbedder):
        self.tracks:   list[KalmanTrack] = []
        self.lost:     list[KalmanTrack] = []   # recently lost tracks (for re-ID)
        self.embedder = embedder
        self._dead_ids: set = set()             # IDs that have been deleted

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: list, gray_frame: np.ndarray) -> list[TrackResult]:
        """
        detections: list of (x,y,w,h) from face detector
        Returns list of TrackResult for all currently active tracks.
        """
        # 1. Predict all active tracks
        for t in self.tracks:
            t.predict()

        # 2. Match detections → active tracks (high IoU)
        unmatched_det, matched_tracks = self._match(detections, self.tracks)

        # 3. Try to match remaining detections → lost tracks via re-ID
        still_unmatched = []
        for det_idx in unmatched_det:
            det = detections[det_idx]
            x, y, w, h = det
            gray_patch  = gray_frame[max(0,y):y+h, max(0,x):x+w]
            reid_id     = self.embedder.best_match(gray_patch,
                                                   [t.id for t in self.lost])
            if reid_id is not None:
                # Recover the lost track
                lost_track = next((t for t in self.lost if t.id == reid_id), None)
                if lost_track:
                    lost_track.update(det)
                    lost_track.miss = 0
                    self.tracks.append(lost_track)
                    self.lost.remove(lost_track)
                    matched_tracks.add(lost_track.id)
            else:
                still_unmatched.append(det_idx)

        # 4. Create new tracks for still-unmatched detections
        for det_idx in still_unmatched:
            det = detections[det_idx]
            new_track = KalmanTrack(det)
            self.tracks.append(new_track)
            # Seed embedding immediately
            x, y, w, h = det
            patch = gray_frame[max(0,y):y+h, max(0,x):x+w]
            self.embedder.add_embedding(new_track.id, patch)

        # 5. Mark missed tracks, update embeddings on matched
        for t in self.tracks:
            if t.id not in matched_tracks:
                t.mark_missed()
            else:
                # Refresh embedding
                x, y, w, h = t.last_bbox
                patch = gray_frame[max(0,y):y+h, max(0,x):x+w]
                self.embedder.add_embedding(t.id, patch)

        # 6. Move very-missed tracks → lost, delete ancient lost tracks
        still_active = []
        for t in self.tracks:
            if t.miss > self.MAX_MISS:
                self.lost.append(t)
            else:
                still_active.append(t)
        self.tracks = still_active

        # Prune lost that have been gone too long
        self.lost = [t for t in self.lost if t.miss < self.MAX_MISS * 3]
        for t_lost in list(self.lost):
            if t_lost.miss >= self.MAX_MISS * 3:
                self.embedder.remove(t_lost.id)

        # 7. Return results
        results = []
        for t in self.tracks:
            results.append(TrackResult(
                id=t.id,
                bbox=t.last_bbox,
                confirmed=(t.hits >= self.MIN_HITS),
            ))
        return results

    def get_bbox_for_id(self, track_id: int) -> Optional[tuple]:
        for t in self.tracks:
            if t.id == track_id:
                return t.last_bbox
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _match(self, detections, tracks) -> tuple:
        """Greedy IoU matching. Returns (unmatched_det_indices, matched_track_ids)."""
        matched_track_ids = set()
        unmatched_det     = list(range(len(detections)))

        if not tracks or not detections:
            return unmatched_det, matched_track_ids

        # Build IoU cost matrix
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for ti, t in enumerate(tracks):
            for di, d in enumerate(detections):
                cost[ti, di] = iou(t.predicted_bbox, d)

        # Greedy assignment (best IoU first)
        while True:
            idx = np.unravel_index(np.argmax(cost), cost.shape)
            ti, di = idx
            if cost[ti, di] < self.IOU_THRESH:
                break
            t = tracks[ti]
            t.update(detections[di])
            matched_track_ids.add(t.id)
            cost[ti, :] = -1
            cost[:, di] = -1
            if di in unmatched_det:
                unmatched_det.remove(di)

        return unmatched_det, matched_track_ids


# ─────────────────────────────────────────────────────────────────────────────
# Drone Commander
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneCommand:
    direction: str        # "center"|"left"|"right"|"forward"|"back"|"pan_left"|"pan_right"|"searching"
    magnitude: float      # 0.0–1.0
    target_id: Optional[int] = None
    timestamp: float      = field(default_factory=time.time)

    def __str__(self):
        return f"CMD[{self.direction:>10}]  mag={self.magnitude:.2f}  tid={self.target_id}"


class DroneCommander:
    """
    Converts target bbox position relative to frame centre into drone commands.
    Dead-zone in centre → 'center' command.
    Target size ratio drives forward/back.
    """
    # Normalised dead-zone (fraction of frame)
    DEAD_ZONE_X   = 0.12
    DEAD_ZONE_Y   = 0.12
    # Desired face area ratio (fraction of frame area)
    TARGET_AREA_LO = 0.04   # too far → forward
    TARGET_AREA_HI = 0.25   # too close → back
    # Pan search speed when searching
    PAN_STEP = 1            # 1 = slow pan, higher = fast

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._pan_dir = "pan_right"
        self._pan_count = 0

    def update_frame_size(self, w: int, h: int):
        self.frame_w = w
        self.frame_h = h

    def compute(self, target_bbox: Optional[tuple],
                state: str) -> DroneCommand:
        """
        state: 'tracked' | 'search'
        target_bbox: (x,y,w,h) or None
        """
        if state == "search" or target_bbox is None:
            # Alternate pan direction every ~60 commands
            self._pan_count += 1
            if self._pan_count > 60:
                self._pan_count = 0
                self._pan_dir = "pan_right" if self._pan_dir == "pan_left" else "pan_left"
            return DroneCommand(direction=self._pan_dir, magnitude=0.5)

        x, y, w, h = target_bbox
        cx = x + w / 2
        cy = y + h / 2

        # Normalise to [-1, 1]
        nx = (cx - self.frame_w / 2) / (self.frame_w / 2)
        ny = (cy - self.frame_h / 2) / (self.frame_h / 2)

        # Area ratio
        area_ratio = (w * h) / (self.frame_w * self.frame_h)

        # Determine primary command
        abs_nx = abs(nx)
        abs_ny = abs(ny)

        if abs_nx > abs_ny and abs_nx > self.DEAD_ZONE_X:
            direction = "right" if nx > 0 else "left"
            magnitude = min(abs_nx, 1.0)
        elif abs_ny > self.DEAD_ZONE_Y:
            direction = "back" if ny < 0 else "forward"   # up in frame = back
            magnitude = min(abs_ny, 1.0)
        elif area_ratio < self.TARGET_AREA_LO:
            direction = "forward"
            magnitude = 0.4
        elif area_ratio > self.TARGET_AREA_HI:
            direction = "back"
            magnitude = 0.4
        else:
            direction = "center"
            magnitude = 0.0

        return DroneCommand(direction=direction, magnitude=magnitude)


# ─────────────────────────────────────────────────────────────────────────────
# Target Lock Manager  (state machine)
# ─────────────────────────────────────────────────────────────────────────────

class TargetLockManager:
    """
    States:  DETECT → TRACKED → SEARCH → TRACKED (re-acquired)
    Exposes:
      • lock_by_click(mx, my, tracks)  – called from mouse callback
      • lock_by_key(track_id)          – called from keypress
      • update(tracks)                 – called every frame
      • state, target_id, target_bbox
    """

    SEARCH_TIMEOUT = 10.0   # seconds before giving up and returning to DETECT

    def __init__(self):
        self.state      = "detect"
        self.target_id: Optional[int]   = None
        self.target_bbox: Optional[tuple] = None
        self._search_start: Optional[float] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def lock_by_click(self, mx: int, my: int,
                      tracks: list[TrackResult]) -> Optional[int]:
        """Click on a face to lock. Returns locked ID or None."""
        for t in tracks:
            if not t.confirmed:
                continue
            x, y, w, h = t.bbox
            if x <= mx <= x+w and y <= my <= y+h:
                self._lock(t.id, t.bbox)
                return t.id
        return None

    def lock_by_id(self, track_id: int,
                   tracks: list[TrackResult]) -> bool:
        """Lock a specific track ID. Returns True if found."""
        for t in tracks:
            if t.id == track_id:
                self._lock(t.id, t.bbox)
                return True
        return False

    def unlock(self):
        self.state      = "detect"
        self.target_id  = None
        self.target_bbox = None
        self._search_start = None
        print("[TARGET] Unlocked → DETECT mode")

    def update(self, tracks: list[TrackResult]) -> DroneCommand:
        """
        Call every frame. Returns the current DroneCommand.
        """
        from tracker import DroneCommander  # late import to avoid circulars
        # (DroneCommander is instantiated in Viewer, we only produce the command here)
        if self.state == "detect":
            return DroneCommand(direction="searching", magnitude=0.0)

        if self.state == "tracked":
            # Try to find our target in fresh tracks
            match = next((t for t in tracks if t.id == self.target_id), None)
            if match:
                self.target_bbox = match.bbox
                return DroneCommand(direction="center", magnitude=0.0,
                                    target_id=self.target_id)   # placeholder; Viewer computes
            else:
                # Lost!
                self.state = "search"
                self._search_start = time.time()
                print(f"[TARGET] ID {self.target_id} lost → SEARCH mode")
                return DroneCommand(direction="pan_right", magnitude=0.5)

        if self.state == "search":
            # Check re-ID: any track matching our stored ID?
            match = next((t for t in tracks if t.id == self.target_id), None)
            if match:
                self.state = "tracked"
                self.target_bbox = match.bbox
                self._search_start = None
                print(f"[TARGET] ID {self.target_id} re-acquired → TRACKED")
                return DroneCommand(direction="center", magnitude=0.0,
                                    target_id=self.target_id)
            # Timeout
            if self._search_start and time.time() - self._search_start > self.SEARCH_TIMEOUT:
                print(f"[TARGET] Search timeout → DETECT mode")
                self.unlock()
            return DroneCommand(direction="pan_right", magnitude=0.5)

        return DroneCommand(direction="searching", magnitude=0.0)

    # ── Private ───────────────────────────────────────────────────────────────

    def _lock(self, tid: int, bbox: tuple):
        self.target_id   = tid
        self.target_bbox = bbox
        self.state       = "tracked"
        self._search_start = None
        print(f"[TARGET] Locked → ID {tid}  bbox={bbox}  → TRACKED mode")