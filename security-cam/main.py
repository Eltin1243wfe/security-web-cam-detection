"""
Stage 6: local web dashboard replaces the desktop OpenCV window.
Detection runs on a background thread and writes into a small shared
SharedState object; Flask (running in the main thread) reads from that
same state to serve the live feed, armed/disarmed control, night
schedule, "It's OK" acknowledge, and the alert log in a browser instead
of a cv2.imshow() window.

Why the split: cv2.imshow() and a web server both want to "own" the
main loop in their own way — imshow needs waitKey() pumped constantly,
Flask's dev server blocks on app.run(). Running detection on its own
thread and Flask on the main thread is the simplest way to let both run
continuously without fighting each other.

Run: python main.py
Then open http://127.0.0.1:5000 in a browser. Ctrl+C in the terminal to
stop — there's no video window to press 'q' in anymore.
"""

import os
import threading
import time
from datetime import datetime

import cv2

import config
from src.camera import PhoneCamera
from src.detector import Detector
from src.rules import PresenceTracker, CooldownGate, is_armed
from src.alerts import play_alert, send_telegram_alert
from src.state import SharedState
from src.dashboard import create_app


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_status(frame, armed: bool, acknowledged: bool):
    text = "ARMED" if armed else "DISARMED"
    color = (0, 0, 255) if armed else (200, 200, 200)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if acknowledged:
        cv2.putText(frame, "ACKNOWLEDGED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def run_detection_loop(state: SharedState):
    """
    Runs forever on a background thread. Everything from stages 1-5
    lives here basically unchanged — the only new pieces are writing
    the annotated frame into shared state instead of cv2.imshow(), and
    checking/clearing the "It's OK" acknowledgment.
    """
    detector = Detector(
        model_name=config.MODEL_NAME,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        classes_of_interest=config.CLASSES_OF_INTEREST,
        min_box_height=config.MIN_PERSON_BOX_HEIGHT,
    )
    presence_tracker = PresenceTracker(
        config.CONSECUTIVE_FRAMES_THRESHOLD,
        max_center_drift=config.MAX_CENTER_DRIFT_PX,
    )
    cooldown_gate = CooldownGate(config.ANOMALY_COOLDOWN_SECONDS)

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)

    # Tracks whether a person was in frame last iteration — used purely
    # to detect the exact moment presence ends, so the "It's OK"
    # acknowledgment can clear itself automatically rather than
    # silently covering whoever walks in next.
    person_present_last_frame = False

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
            person_present = any(d["label"] == "person" for d in detections)

            if not person_present and person_present_last_frame:
                state.clear_acknowledged()
            person_present_last_frame = person_present

            # Only fires True on the frame where presence crosses the
            # consecutive-frame threshold AND stays spatially consistent.
            event_started = presence_tracker.update(detections)
            armed = is_armed(state)
            acknowledged = state.is_acknowledged()

            if event_started and armed and not acknowledged:
                if cooldown_gate.should_fire():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] ANOMALY: sustained person detected while armed")
                    play_alert()
                    send_telegram_alert(frame, caption=f"Person detected at {timestamp}")

                    snapshot_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join(config.SNAPSHOT_DIR, snapshot_name), frame)
                    state.add_alert(timestamp, snapshot_name)
            elif event_started and armed and acknowledged:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] person detected but acknowledged — no alert")

            frame = draw_detections(frame, detections)
            frame = draw_status(frame, armed, acknowledged)

            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                state.set_latest_jpeg(jpeg.tobytes())

            # Not strictly needed for throughput (detection itself is the
            # bottleneck), but keeps this thread from pegging a core if
            # detection ever gets fast enough that it wouldn't otherwise.
            time.sleep(0.01)


def main():
    state = SharedState(
        armed=config.SYSTEM_ARMED,
        use_time_window=config.USE_TIME_WINDOW,
        armed_start_hour=config.ARMED_START_HOUR,
        armed_end_hour=config.ARMED_END_HOUR,
    )

    detection_thread = threading.Thread(target=run_detection_loop, args=(state,), daemon=True)
    detection_thread.start()

    app = create_app(state)
    print(f"Dashboard running at http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, threaded=True)


if __name__ == "__main__":
    main()