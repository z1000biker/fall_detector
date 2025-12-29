# Safety Case

This document outlines the safety rationale for the system.

---

## Intended Use

Assistive monitoring of corridor environments to detect posture collapse and failure to recover.

---

## Non-Intended Use

- Medical diagnosis
- Injury assessment
- Cause-of-fall determination
- Private living spaces

---

## Primary Hazards

| Hazard | Description |
|------|------------|
| False reassurance | System assumes recovery when person is down |
| Silent failure | Loss of visibility suppresses alert |
| Alert fatigue | Excessive or oscillating alerts |
| Misuse | Deployment outside corridor context |

---

## Safety Controls

### Hazard: False Reassurance
- State persistence across ID loss
- Occlusion escalation
- No confidence-based suppression

### Hazard: Silent Failure
- Explicit UNSEEN state
- Timers continue during occlusion
- Conservative alerting

### Hazard: Alert Fatigue
- Alert latching
- Operator acknowledgment
- Rate-limited audio

### Hazard: Misuse
- Explicit corridor mode labeling
- Documented limitations
- Architecture guard (SENSOR_MODE)

---

## Residual Risk

Residual risk remains due to:
- Camera blind spots
- Severe occlusion
- Assistive devices
- Environmental misuse

These risks are documented and communicated.

---

## Conclusion

The system is designed to fail conservatively and visibly.
Uncertainty increases escalation rather than suppressing response.
