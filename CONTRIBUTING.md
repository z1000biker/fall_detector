# Contributing Guidelines

Thank you for your interest in contributing to this project.

This repository implements a **safety-oriented, camera-based collapse and non-recovery detection system**. Contributions are welcome, but **must respect the safety assumptions and design philosophy** described below.

---

## Guiding Principles

Before contributing, please understand the core principles of this system:

1. **Uncertainty increases risk**  
   Loss of visibility, tracking instability, or degraded pose confidence must never silently reduce alerts.

2. **State over classification**  
   Contributions should favor temporal reasoning and explicit state transitions rather than per-frame predictions.

3. **Conservative escalation**  
   False positives are preferable to false reassurance.

4. **Explainability over cleverness**  
   Any added logic should be inspectable, auditable, and explainable.

5. **No medical claims**  
   This project does not diagnose injuries or medical conditions.

---

## What Contributions Are Welcome

- Improvements to state logic or temporal reasoning
- Better handling of occlusion and uncertainty
- Performance and stability improvements
- Documentation improvements
- Visualization and debugging tools
- Logging, auditing, or replay tooling
- Test videos and edge-case scenarios

---

## What Contributions Are NOT Accepted

- Features that suppress alerts based solely on low confidence
- Claims of medical diagnosis or injury detection
- Black-box “confidence scores” used to silence alerts
- Features that assume multi-camera input without architectural changes
- Removal of explicit UNSEEN / uncertainty states

---

## Code Style & Expectations

- Keep logic deterministic where possible
- Avoid silent failure modes
- Prefer readable state machines over compact heuristics
- Comment **why** something exists, not just what it does
- Document new assumptions explicitly

---

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Test changes with:
   - Occlusion scenarios
   - Multiple people
   - Frame drops / lag
4. Submit a pull request with:
   - A clear description
   - Rationale for safety impact
   - Known limitations

---

## Final Note

This project is intentionally conservative.  
If a proposed change makes the system *feel* quieter, smoother, or less noisy, ask first whether it also makes it **less honest under uncertainty**.

When in doubt, escalate.

Thank you for contributing thoughtfully.
