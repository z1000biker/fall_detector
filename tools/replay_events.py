import json
import time
from datetime import datetime
import os

EVENT_FILE = "fall_events.jsonl"
SPEEDUP = 5.0  # Replay speed

def replay():
    if not os.path.exists(EVENT_FILE):
        print(f"Error: {EVENT_FILE} not found. Run the detector first to generate logs.")
        return

    print("Replaying events...\n")
    last_ts = None

    with open(EVENT_FILE) as f:
        for line in f:
            try:
                event = json.loads(line)
                ts = event["timestamp"]

                if last_ts:
                    dt = (ts - last_ts) / SPEEDUP
                    time.sleep(max(0, dt))

                last_ts = ts

                print(
                    f"[{event['iso_time']}] "
                    f"Person {event['person_id']} "
                    f"{event['event']} "
                    f"{event.get('details', {})}"
                )
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    replay()
