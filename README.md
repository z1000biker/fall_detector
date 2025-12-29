# Camera-Based Collapse & Non-Recovery Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-enabled-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8--Pose-ultralytics-orange)
![Status](https://img.shields.io/badge/status-experimental--safety--critical-yellow)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Platform](https://img.shields.io/badge/platform-single--camera-lightgrey)
![Scope](https://img.shields.io/badge/scope-corridor--only-red)

> **A Safety-Oriented, Single-Camera Corridor Monitoring System**

## Overview

I’ve been building a single-camera, vision-based system designed to monitor corridor environments and detect sudden posture collapse and failure to recover using real-time human pose estimation.

I’m sharing this work publicly because, after years of developing closed medical software, this project represents a deliberate shift: toward transparent, explainable, safety-oriented systems whose limitations are explicit rather than hidden.

> [!IMPORTANT]
> **This repository does not present a medical device and makes no diagnostic claims.**
> Instead, it documents a conservative, assistive monitoring system designed to support human responders by escalating risk when a person loses functional posture and does not recover within an expected timeframe.

## What This System Is — and Is Not

### What it does
- ✅ **Detects rapid transitions** from upright to horizontal posture
- ✅ **Tracks persistent horizontal states** over time
- ✅ **Detects failure to recover**
- ✅ **Escalates risk** if a person becomes occluded or disappears after collapse
- ✅ **Supports multiple people** in a single camera view
- ✅ **Explicitly represents uncertainty** (UNSEEN states)
- ✅ **Requires human acknowledgment** of alerts
- ✅ **Logs all critical events** for post-incident review

### What it does not do
- ❌ **Diagnose injuries** or medical conditions
- ❌ **Determine cause** of a fall
- ❌ **Guarantee detection**
- ❌ **Replace human judgment** or response
- ❌ **Operate safely outside its declared environment**

The system is intentionally scoped to **corridor-like environments** where people are expected to be upright and walking.

---

## Core Design Philosophy

### 1. State, Not Frames
This system does not treat fall detection as a per-frame classification problem. Instead, it models human posture and behavior as a **temporal state machine**, where meaning emerges from transitions and persistence, not isolated predictions. A “fall” is not an outcome — it is a transition.

### 2. Uncertainty Is Risk
A foundational safety principle: **Uncertainty must increase risk, never reduce it.**
If a person becomes occluded, partially visible, or disappears after a collapse, the system escalates rather than assuming recovery. There is no silent failure mode.

### 3. Conservative Failure
**False positives are preferable to false reassurance.** The system is deliberately biased toward erring on the side of alerting when confidence is lost.

---

## System Architecture

### High-Level Pipeline
1. **Video Input**: Single RGB camera (Fixed position, Corridor FOV)
2. **Pose Estimation**: YOLOv8-Pose (Multi-person, ID tracking with distrust of persistence)
3. **Metric Extraction**: 
   - Torso orientation (L/R averaged)
   - Bounding box aspect ratio
   - Hip-to-ankle vertical compression
   - Centroid velocity (clamped)
4. **Temporal Reasoning**: Frame confirmation, Recovery windows, Occlusion timers
5. **State Machine**: Explicit states, Deterministic transitions
6. **Alerting & Logging**: Latched alerts, Acknowledgment, JSON logs

### Human State Model
The system tracks each person independently using the following states:

| State | Description |
|-------|-------------|
| `UPRIGHT` | Normal posture, expected movement |
| `FALLING` | Rapid posture transition detected (high urgency) |
| `HORIZONTAL_UNCONFIRMED` | Horizontal posture detected, recovery still possible |
| `HORIZONTAL_INACTIVE` | Horizontal posture persists beyond recovery window |
| `ALERT` | Collapse confirmed; requires human attention (Latched) |
| `ACKNOWLEDGED` | Alert seen by operator; monitoring continues silently |
| `UNSEEN` | Person temporarily occluded or lost; uncertainty is explicit |

---

## Key Detection Techniques

### 1. Dual-Torso Angle Estimation
Uses left and right shoulder-to-hip vectors and averages angles to reduce occlusion sensitivity. Robust to partial skeleton loss.

### 2. Bounding Box Aspect Ratio
Standing humans are tall and narrow; fallen humans are wider or square. Acts as a posture proxy when keypoints are degraded.

### 3. Ground Contact Heuristic (Hip-Ankle Compression)
Measures vertical compression of the lower body to filter kneeling vs collapsed posture. Strong signal in corridor environments.

> **Known Limitation**: Assistive devices (wheelchairs, walkers) alter this geometry and are explicitly documented as a limitation.

### 4. Velocity-Aware Collapse Detection
High velocity during collapse, low velocity after. Time-delta clamping prevents false spikes during frame drops. This avoids the common error: “Low motion means fall.”

---

## Occlusion & Identity Handling

### Identity Is Not Trusted
Tracking IDs are treated as fragile hints, not ground truth. If an ID disappears, **timers persist** and risk may escalate.

### UNSEEN State
When visibility is lost, the system enters `UNSEEN`. Duration is tracked, last known location is shown, and escalation continues if collapse was suspected.

### Single-Camera Guard
Occlusion escalation is explicitly guarded by `SENSOR_MODE = "SINGLE_CAMERA"`. This prevents unsafe reuse in multi-camera systems without architectural changes.

---

## Operations & Alerting

### Alert Lifecycle
1. **ALERT**: Visual + audio alarm (Latched)
2. **ACKNOWLEDGED**: Operator confirms awareness (Spacebar). Audio muted.
3. **RECOVERY**: Strong evidence of upright posture clears alert automatically.
4. **RESET**: Manual operator override (logged).

### Event Logging
All critical events are logged in JSONL format to `fall_events.jsonl` for post-incident review and accountability.

---

## Installation & Usage

### Requirements
- Python 3.9+
- OpenCV, NumPy, Ultralytics

```bash
pip install -r requirements.txt
# OR
pip install opencv-python numpy ultralytics
```

### Running
```bash
python fall_detector.py
```
- **SPACE**: Acknowledge Alert
- **R**: Reset System
- **ESC**: Exit

### Docker
```bash
docker build -t fall-detector .
# Requires passing display/webcam to container
```

---

## Known Limitations

This system cannot and should not claim to handle:
- Injury severity or medical diagnosis.
- Cause of collapse.
- Private environments (bedrooms, bathrooms).
- Non-corridor contexts without retraining/redesign.
- **Assistive Devices**: Wheelchairs/walkers may trigger false positives due to ground contact heuristics.

## Intended Use
This system is intended as an **assistive monitoring tool** in environments such as hospital corridors, care facility hallways, or staff-monitored walking areas. It is **not** to replace staff, alarms, or medical judgment.

---

**Author Contact**: Nik <sv1eex@hotmail.com>
