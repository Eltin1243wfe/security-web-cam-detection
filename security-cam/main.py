"""
Stage 1+2: get the phone feed on screen with YOLO boxes drawn on it.

No anomaly logic, no alerting yet — that's next once I've confirmed
detection quality is actually good enough on real footage from my room.
No point building rules on top of a feed I haven't looked at yet.

Run: python main.py
Quit: press 'q' with the video window focused.
"""

import cv2

import config
from src.camera import PhoneCamera
from src.detector import Detector


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def main():
    detector = Detector(
        model_name=config.MODEL_NAME,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        classes_of_interest=config.CLASSES_OF_INTEREST,
    )

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
            frame = draw_detections(frame, detections)

            cv2.imshow("Security Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
