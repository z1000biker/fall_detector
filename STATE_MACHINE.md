# State Machine

```mermaid
stateDiagram-v2
    [*] --> UPRIGHT

    UPRIGHT --> FALLING: Rapid posture + velocity
    UPRIGHT --> HORIZONTAL_UNCONFIRMED: Gradual collapse

    FALLING --> HORIZONTAL_INACTIVE: Motion stops

    HORIZONTAL_UNCONFIRMED --> UPRIGHT: Recovery
    HORIZONTAL_UNCONFIRMED --> HORIZONTAL_INACTIVE: Timeout

    HORIZONTAL_INACTIVE --> ALERT: Alert timeout
    HORIZONTAL_INACTIVE --> UPRIGHT: Recovery

    ALERT --> ACKNOWLEDGED: Operator acknowledges
    ALERT --> UPRIGHT: Confirmed recovery

    ACKNOWLEDGED --> UPRIGHT: Confirmed recovery

    UPRIGHT --> UNSEEN: Occlusion
    UNSEEN --> ALERT: Occlusion after collapse
    UNSEEN --> UPRIGHT: Reappearance
```

## Design Notes

1. **Alerts never auto-resolve silently**
2. **UNSEEN represents uncertainty explicitly**
3. **Occlusion after collapse escalates risk**
4. **Recovery requires strong evidence**
