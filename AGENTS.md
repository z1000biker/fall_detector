# AGENTS.md — AI & Autonomous Tooling Guidance

This document provides guidance for AI tools, coding agents, and automated systems interacting with this repository.

---

## Project Nature

This repository contains a **safety-critical, camera-based monitoring system** designed to detect posture collapse and failure to recover in corridor environments.

Although it is not a medical device, **unsafe changes can lead to silent failure modes** and must be avoided.

---

## Core Safety Constraints

Any AI agent modifying this codebase MUST respect the following constraints:

1. **Do NOT suppress alerts due to low confidence**
   - Low confidence increases risk.
   - Uncertainty must never silence the system.

2. **Do NOT remove or bypass state machines**
   - Per-frame classification is explicitly insufficient.
   - State transitions must remain explicit and auditable.

3. **Do NOT assume identity persistence**
   - Tracking IDs are unreliable.
   - ID loss must not reset collapse timers.

4. **Do NOT auto-resolve alerts without evidence**
   - Alerts must latch.
   - Resolution requires strong upright evidence or manual reset.

5. **Do NOT add medical claims**
   - No diagnosis, injury detection, or cause inference.

---

## Allowed Modifications

AI agents MAY:
- Refactor code for clarity
- Improve performance without changing semantics
- Add tests, logging, or replay tooling
- Improve documentation
- Add explicit safety checks
- Surface uncertainty more clearly

---

## Disallowed Modifications

AI agents MUST NOT:
- Introduce probabilistic “fall confidence” used to suppress alerts
- Replace state logic with end-to-end classifiers
- Assume multi-camera fusion without architectural changes
- Remove UNSEEN or occlusion escalation logic
- Optimize for silence or reduced alerting

---

## Design Intent Summary

This system is intentionally conservative.

If an AI agent is unsure whether a change reduces risk or hides uncertainty, the correct action is **to not apply the change** and instead request human review.

---

## Final Rule

When modifying this repository, AI agents should follow this heuristic:

> **If uncertainty increases, escalation should increase — never decrease.**
