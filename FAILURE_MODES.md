# Failure Modes & Mitigations

---

## Occlusion After Collapse
**Risk:** Person disappears after falling  
**Mitigation:** UNSEEN state + escalation timer

---

## Tracking ID Switch
**Risk:** State reset due to new ID  
**Mitigation:** State persistence independent of ID

---

## Frame Drops / Lag
**Risk:** Velocity spikes or drops  
**Mitigation:** Time-delta clamping

---

## Kneeling False Positives
**Risk:** Kneel classified as fall  
**Mitigation:** Hip-to-ankle compression heuristic

---

## Assistive Devices
**Risk:** Misclassification  
**Mitigation:** Explicit documentation + corridor-only scope

---

## Operator Inattention
**Risk:** Alert ignored  
**Mitigation:** Latched alerts + acknowledgment state

---

## Misuse Outside Corridor
**Risk:** High false positives  
**Mitigation:** Explicit mode labeling + documentation
