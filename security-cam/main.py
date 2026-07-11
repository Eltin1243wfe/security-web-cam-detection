"""
Stage 1+2+3: phone feed + YOLO boxes + anomaly rules on top.

Detection alone was proving unreliable to alert on directly — a chair
and a towel both threw one-frame false positives during testing, and
not every real person in frame is something worth being alerted about
(me getting water at 2am shouldn't page me). So this stage adds a
rules layer between "YOLO saw a person" and "this is worth caring
about": a consecutive-frame filter to kill single-frame noise, and an
armed/disarmed concept (with an optional time window) so detections
only become anomalies when they're actually supposed to.

No real alerting yet (sound/Telegram/Pushover) — that's stage 4. For
now, anomalies just print to the console so I can watch the logic
work before wiring up something that'll actually interrupt me.

Run: python main.py
Quit: press 'q' with the video window focused.
"""

from datetime import datetime

import cv2

import config
from src.camera import PhoneCamera
from src.detector import Detector
from src.rules import PresenceTracker, CooldownGate, is_armed

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def draw_status(frame, armed: bool):
    # Small persistent readout so I can see at a glance whether the
    # system thinks it's armed right now, without staring at the
    # terminal — mainly useful for sanity-checking the time window.
    text = "ARMED" if armed else "DISARMED"
    color = (0, 0, 255) if armed else (200, 200, 200)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def main():
    detector = Detector(
        model_name=config.MODEL_NAME,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        classes_of_interest=config.CLASSES_OF_INTEREST,
    )
    presence_tracker = PresenceTracker(config.CONSECUTIVE_FRAMES_THRESHOLD) 
    cooldown_gate = CooldownGate(config.ANOMALY_COOLDOWN_SECONDS)


    with PhoneCamera(config.CAMERA_SOURCE) as cam:
        while True:
            frame = cam.read_frame()

            # IP Webcam over WiFi drops frames occasionally — don't crash
            # the whole loop over a hiccup, just skip and try again.
            if frame is None:
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH,
                                        int(frame.shape[0] * config.FRAME_WIDTH / frame.shape[1])))

            detections = detector.detect(frame)
            person_detected = any(d["label"] == "person" for d in detections)

            # Only fires True on the frame where presence crosses the
            # consecutive-frame threshold — this is what filters out the
            # chair/towel-style single-frame flicker from becoming an event.
            event_started = presence_tracker.update(person_detected)
            armed = is_armed()

            if event_started and armed:
                if cooldown_gate.should_fire():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] ANOMALY: sustained person detected while armed")
            elif event_started:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] person detected (system disarmed, no anomaly)")

            frame = draw_detections(frame, detections)
            frame = draw_status(frame, armed)

            cv2.imshow("Security Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()