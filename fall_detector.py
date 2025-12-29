"""
Camera-Based Collapse & Non-Recovery Detection System
=====================================================

A safety-oriented, single-camera monitoring system designed to detect 
sudden posture collapse and failure to recover in corridor environments.

This system prioritizes safety by escalating risk under uncertainty 
(occlusion, ID loss) and requires explicit human acknowledgment for alerts.

Author: Nikiforos (Nik) Kontopoulos
License: MIT
"""

import cv2
import time
import math
import argparse
import logging
import winsound
import json
import threading
import queue
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List
import numpy as np
from ultralytics import YOLO

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fall_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION
# =========================
@dataclass
class FallDetectionConfig:
    # --- System Architecture ---
    SENSOR_MODE: str = "SINGLE_CAMERA"  # Architecture guard
    
    # --- Thresholds ---
    pose_conf_min: float = 0.4
    keypoint_conf_min: float = 0.5
    
    # Angles & Ratios
    upright_torso_max: float = 35.0
    horizontal_torso_min: float = 55.0
    standing_aspect_min: float = 1.4
    fallen_aspect_max: float = 1.1
    ground_contact_ratio_max: float = 0.15 
    
    # Velocity
    bbox_drop_ratio: float = 0.35
    stationary_velocity_max: float = 15.0
    max_frame_dt: float = 0.2  # Max time delta to consider valid (5 fps min)
    
    # --- Timing & Safety ---
    recovery_window_sec: float = 5.0
    alert_sec: float = 15.0
    
    # Occlusion
    occlusion_alert_sec: float = 10.0
    
    # Frame confirmation
    confirm_frames_required: int = 5
    
    # Alerts
    enable_audio_alert: bool = True
    alert_frequency_hz: int = 1000
    alert_ack_frequency_hz: int = 500  # Lower pitch for acknowledged
    alert_duration_ms: int = 500
    alert_interval_sec: float = 1.0

# =========================
# SAFE AUDIO ALERT SYSTEM
# =========================
class SafeAlertSystem:
    """Thread-safe alert system using a dedicated daemon thread."""
    def __init__(self, config: FallDetectionConfig):
        self.config = config
        self.state = "OFF" # OFF, ACTIVE, ACKNOWLEDGED
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.thread.start()

    def set_state(self, state: str):
        with self.lock:
            self.state = state

    def _alert_loop(self):
        while self.running:
            current_state = "OFF"
            with self.lock:
                current_state = self.state
            
            if self.config.enable_audio_alert:
                if current_state == "ACTIVE":
                    try:
                        winsound.Beep(self.config.alert_frequency_hz, self.config.alert_duration_ms)
                    except: pass
                    time.sleep(self.config.alert_interval_sec)
                elif current_state == "ACKNOWLEDGED":
                    # Quieter/slower beep for acknowledged but unresolved
                    try:
                        winsound.Beep(self.config.alert_ack_frequency_hz, 200)
                    except: pass
                    time.sleep(2.0)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def stop(self):
        self.running = False

# =========================
# EVENT LOGGER
# =========================
class EventLogger:
    """Logs critical events to JSONL for review."""
    def __init__(self, filename="fall_events.jsonl"):
        self.filename = filename

    def log_event(self, event_type: str, person_id: int, details: dict):
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event_type,
            "person_id": person_id,
            "details": details
        }
        with open(self.filename, "a") as f:
            f.write(json.dumps(entry) + "\n")

# =========================
# UTILITIES
# =========================
def calculate_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    angle_rad = math.atan2(abs(dx), dy) if dy != 0 else math.pi / 2
    return math.degrees(angle_rad)

def get_centroid(keypoints: np.ndarray, conf_threshold: float = 0.5) -> Optional[Tuple[float, float]]:
    valid_points = keypoints[keypoints[:, 2] > conf_threshold]
    if len(valid_points) == 0: return None
    return float(np.mean(valid_points[:, 0])), float(np.mean(valid_points[:, 1]))

def get_keypoint(keypoints: np.ndarray, idx: int, conf_threshold: float = 0.5) -> Optional[np.ndarray]:
    if idx < len(keypoints) and keypoints[idx, 2] > conf_threshold: return keypoints[idx]
    return None

class YOLOKeypoints:
    NOSE=0; LEFT_SHOULDER=5; RIGHT_SHOULDER=6; LEFT_HIP=11; RIGHT_HIP=12; LEFT_ANKLE=15; RIGHT_ANKLE=16

# =========================
# STATES
# =========================
class PersonState:
    UPRIGHT = "UPRIGHT"
    FALLING = "FALLING"
    HORIZONTAL_UNCONFIRMED = "HORIZONTAL_UNCONFIRMED"
    HORIZONTAL_INACTIVE = "HORIZONTAL_INACTIVE"
    ALERT = "ALERT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNSEEN = "UNSEEN"

@dataclass
class PersonMetrics:
    torso_angle: Optional[float] = None
    aspect_ratio: Optional[float] = None
    velocity: float = 0.0
    hip_ankle_ratio: Optional[float] = None
    is_horizontal: bool = False
    is_upright: bool = False

class Person:
    def __init__(self, pid: int, config: FallDetectionConfig, event_logger: EventLogger):
        self.id = pid
        self.config = config
        self.logger = event_logger
        self.state = PersonState.UPRIGHT
        
        # History
        self.centroid_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        self.bbox_height_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        # Timers
        self.collapse_time: Optional[float] = None
        self.last_seen_time: float = time.time()
        self.latched_alert = False
        self.acknowledged = False
        
        # Counters
        self.horizontal_frame_count = 0
        self.upright_frame_count = 0
        self.unseen_duration = 0.0

    def compute_metrics(self, keypoints: np.ndarray, bbox: np.ndarray) -> PersonMetrics:
        metrics = PersonMetrics()
        now = time.time()
        
        # 1. Torso Angle
        l_sh = get_keypoint(keypoints, YOLOKeypoints.LEFT_SHOULDER, self.config.keypoint_conf_min)
        r_sh = get_keypoint(keypoints, YOLOKeypoints.RIGHT_SHOULDER, self.config.keypoint_conf_min)
        l_hip = get_keypoint(keypoints, YOLOKeypoints.LEFT_HIP, self.config.keypoint_conf_min)
        r_hip = get_keypoint(keypoints, YOLOKeypoints.RIGHT_HIP, self.config.keypoint_conf_min)
        
        angles = []
        if l_sh is not None and l_hip is not None: angles.append(calculate_angle(l_hip[:2], l_sh[:2]))
        if r_sh is not None and r_hip is not None: angles.append(calculate_angle(r_hip[:2], r_sh[:2]))
        if angles: metrics.torso_angle = np.mean(angles)
        
        # 2. Aspect Ratio
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > 0: metrics.aspect_ratio = h / w

        # 3. Ground Contact
        l_ank = get_keypoint(keypoints, YOLOKeypoints.LEFT_ANKLE, self.config.keypoint_conf_min)
        r_ank = get_keypoint(keypoints, YOLOKeypoints.RIGHT_ANKLE, self.config.keypoint_conf_min)
        hips_y = [k[1] for k in [l_hip, r_hip] if k is not None]
        ankles_y = [k[1] for k in [l_ank, r_ank] if k is not None]
        
        if hips_y and ankles_y and h > 0:
            metrics.hip_ankle_ratio = (np.mean(ankles_y) - np.mean(hips_y)) / h
        
        # 4. Velocity (with dt clamping)
        c = get_centroid(keypoints)
        if c and self.centroid_history:
            dt = now - self.time_history[-1]
            if dt > 0:
                # Clamp dt to avoid massive velocity spikes on frame drops
                safe_dt = min(dt, self.config.max_frame_dt)
                dist = np.linalg.norm(np.array(c) - np.array(self.centroid_history[-1]))
                metrics.velocity = dist / safe_dt
                self.velocity_history.append(metrics.velocity)
        
        if c:
            self.centroid_history.append(c)
            self.time_history.append(now)
        self.bbox_height_history.append(h)
        
        # Scoring
        horiz_score = 0
        upright_score = 0
        
        if metrics.torso_angle is not None:
            if metrics.torso_angle > self.config.horizontal_torso_min: horiz_score += 2
            elif metrics.torso_angle < self.config.upright_torso_max: upright_score += 2
            
        if metrics.aspect_ratio is not None:
            if metrics.aspect_ratio < self.config.fallen_aspect_max: horiz_score += 1
            elif metrics.aspect_ratio > self.config.standing_aspect_min: upright_score += 1
            
        if metrics.hip_ankle_ratio is not None:
            if metrics.hip_ankle_ratio < self.config.ground_contact_ratio_max: horiz_score += 2
            elif metrics.hip_ankle_ratio > 0.3: upright_score += 1
        
        metrics.is_horizontal = (horiz_score >= 3)
        metrics.is_upright = (upright_score >= 2)
        return metrics

    def handle_missing(self) -> Optional[str]:
        now = time.time()
        self.unseen_duration = now - self.last_seen_time
        
        if self.state != PersonState.UNSEEN and self.unseen_duration > 1.0:
             if not self.latched_alert and not self.acknowledged:
                 pass

        if self.latched_alert:
            if self.acknowledged: return PersonState.ACKNOWLEDGED
            return PersonState.ALERT

        # Occlusion escalation (Guarded by SENSOR_MODE)
        if self.config.SENSOR_MODE == "SINGLE_CAMERA":
            if self.collapse_time:
                 # Escalation logic for single cam
                 if (now - self.collapse_time) > self.config.occlusion_alert_sec:
                     self.latched_alert = True
                     self.state = PersonState.ALERT
                     self.logger.log_event("ALERT_LATCHED", self.id, {"reason": "occlusion_after_collapse_single_cam", "duration": self.unseen_duration})
                     return PersonState.ALERT
        
        if self.unseen_duration > 1.0:
            return PersonState.UNSEEN
            
        return None

    def acknowledge_alert(self):
        """Operator acknowledges alert."""
        if self.latched_alert:
            self.acknowledged = True
            self.state = PersonState.ACKNOWLEDGED
            self.logger.log_event("ACKNOWLEDGED", self.id, {"type": "manual"})

    def update(self, keypoints: np.ndarray, bbox: np.ndarray, conf: float) -> Optional[str]:
        now = time.time()
        self.last_seen_time = now
        self.unseen_duration = 0.0
        
        if self.latched_alert:
            metrics = self.compute_metrics(keypoints, bbox)
            
            # If acknowledged, we still track recovery but sound is different
            if self.acknowledged:
                self.state = PersonState.ACKNOWLEDGED
            else:
                self.state = PersonState.ALERT
            
            # Check for recovery
            if metrics.is_upright and metrics.torso_angle < 20.0:
                 self.upright_frame_count += 1
                 if self.upright_frame_count > 15:
                     self.latched_alert = False
                     self.acknowledged = False
                     self.state = PersonState.UPRIGHT
                     self.collapse_time = None
                     self.logger.log_event("RECOVERY", self.id, {"reason": "confirmed_upright"})
            else:
                self.upright_frame_count = 0
            
            return self.state

        if conf < self.config.pose_conf_min:
             return self.handle_missing()

        metrics = self.compute_metrics(keypoints, bbox)
        
        # State Transitions
        if self.state == PersonState.UPRIGHT or self.state == PersonState.UNSEEN:
            if metrics.is_horizontal:
                self.horizontal_frame_count += 1
                self.upright_frame_count = 0
                
                recent_max = max(list(self.bbox_height_history)[:-1]) if len(self.bbox_height_history) > 1 else 0
                current_h = self.bbox_height_history[-1]
                is_rapid = (recent_max > 0 and (recent_max - current_h)/recent_max > self.config.bbox_drop_ratio)
                
                if self.horizontal_frame_count >= self.config.confirm_frames_required:
                    self.collapse_time = now
                    if is_rapid:
                        self.state = PersonState.FALLING
                        self.logger.log_event("FALL_DETECTED", self.id, {"type": "rapid"})
                    else:
                        self.state = PersonState.HORIZONTAL_UNCONFIRMED
                        self.logger.log_event("FALL_DETECTED", self.id, {"type": "gradual"})
            else:
                self.horizontal_frame_count = 0
                self.state = PersonState.UPRIGHT

        elif self.state == PersonState.FALLING:
             avg_vel = 0 if not self.velocity_history else np.mean(self.velocity_history)
             if avg_vel < self.config.stationary_velocity_max:
                 self.state = PersonState.HORIZONTAL_INACTIVE
                 self.logger.log_event("STATE_CHANGE", self.id, {"new_state": "INACTIVE"})
        
        elif self.state == PersonState.HORIZONTAL_UNCONFIRMED:
            if metrics.is_upright:
                self.upright_frame_count += 1
                if self.upright_frame_count >= self.config.confirm_frames_required:
                    self.state = PersonState.UPRIGHT
                    self.collapse_time = None
            elif self.collapse_time and (now - self.collapse_time) > self.config.recovery_window_sec:
                self.state = PersonState.HORIZONTAL_INACTIVE
                self.logger.log_event("STATE_CHANGE", self.id, {"new_state": "INACTIVE"})

        elif self.state == PersonState.HORIZONTAL_INACTIVE:
            if metrics.is_upright:
                self.upright_frame_count += 1
                if self.upright_frame_count >= self.config.confirm_frames_required:
                    self.state = PersonState.UPRIGHT
                    self.collapse_time = None
            
            if self.collapse_time and (now - self.collapse_time) > self.config.alert_sec:
                self.state = PersonState.ALERT
                self.latched_alert = True
                self.logger.log_event("ALERT_LATCHED", self.id, {"reason": "timeout"})
                return PersonState.ALERT

        return None

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    
    config = FallDetectionConfig(enable_audio_alert=not args.no_audio)
    event_logger = EventLogger()
    alert_system = SafeAlertSystem(config)
    
    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    
    people: Dict[int, Person] = {}
    
    print("Camera-Based Collapse Detection System Started...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            cv2.putText(frame, "MODE: CORRIDOR (STRICT)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, "MODE: CORRIDOR (STRICT)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            results = model.track(frame, persist=True, verbose=False)
            current_frame_pids = set()
            
            # Determine global alert state
            global_state = "OFF"
            
            if results[0].boxes.id is not None:
                for i, box in enumerate(results[0].boxes):
                    pid = int(box.id.item())
                    current_frame_pids.add(pid)
                    if pid not in people: people[pid] = Person(pid, config, event_logger)
                    
                    kpts = results[0].keypoints.data[i].cpu().numpy()
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    
                    people[pid].update(kpts, bbox, conf)
                    
                    # Draw
                    color = (0,255,0)
                    if people[pid].state == PersonState.FALLING: color = (0,165,255)
                    elif people[pid].state == PersonState.ALERT: color = (0,0,255)
                    elif people[pid].state == PersonState.ACKNOWLEDGED: color = (255,0,255) # Magenta
                    elif people[pid].state == PersonState.HORIZONTAL_INACTIVE: color = (0,0,200)
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    label = f"ID {pid}: {people[pid].state}"
                    cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Handle Missing
            for pid in list(people.keys()):
                if pid not in current_frame_pids:
                    state = people[pid].handle_missing()
                    
                    if state == PersonState.UNSEEN and people[pid].unseen_duration < 10.0:
                         last_pos = people[pid].centroid_history[-1] if people[pid].centroid_history else (0,0)
                         if last_pos:
                             txt = f"ID {pid}: UNSEEN ({people[pid].unseen_duration:.1f}s)"
                             cv2.putText(frame, txt, (int(last_pos[0]), int(last_pos[1])), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 2)

                    if people[pid].unseen_duration > 15.0 and not people[pid].latched_alert:
                        del people[pid]

            # Aggregate Alert State
            active_alerts = [p for p in people.values() if p.state == PersonState.ALERT]
            ack_alerts = [p for p in people.values() if p.state == PersonState.ACKNOWLEDGED]
            
            if active_alerts:
                global_state = "ACTIVE"
            elif ack_alerts:
                global_state = "ACKNOWLEDGED"
            
            alert_system.set_state(global_state)
            
            # Global UI
            if global_state == "ACTIVE":
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
                cv2.putText(frame, "EMERGENCY ALERT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 5)
                cv2.putText(frame, "PRESS 'SPACE' TO ACKNOWLEDGE", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            elif global_state == "ACKNOWLEDGED":
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (255,0,255), 10)
                cv2.putText(frame, "ALERT ACKNOWLEDGED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,0,255), 5)

            cv2.imshow("Medical Fall Detection v3.2", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord('r'): # Reset all
                for p in people.values(): 
                    if p.latched_alert:
                         p.latched_alert = False
                         p.acknowledged = False
                         p.state = PersonState.UPRIGHT
                         event_logger.log_event("RESET", p.id, {"type": "manual"})
            if key == 32: # SPACE to Acknowledge
                 for p in people.values():
                     if p.state == PersonState.ALERT:
                         p.acknowledge_alert()

    finally:
        alert_system.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
