"""
Thin wrapper around ultralytics YOLO. Reason this exists instead of
calling model() directly in main.py: I want to filter by class and
confidence in one place, so main.py just gets back clean detections
and stays focused on the display/event loop.
"""

from ultralytics import YOLO


class Detector:
    def __init__(self, model_name: str, confidence_threshold: float, classes_of_interest: list[str]):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.classes_of_interest = set(classes_of_interest)

    def detect(self, frame):
        """
        Runs inference on a frame and returns a list of detections:
        [{"label": str, "confidence": float, "box": (x1, y1, x2, y2)}, ...]

        Returning plain dicts instead of raw ultralytics Result objects
        so nothing downstream needs to know this is YOLO specifically.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            confidence = float(box.conf[0])
            label = self.model.names[int(box.cls[0])]

            if confidence < self.confidence_threshold:
                continue
            if self.classes_of_interest and label not in self.classes_of_interest:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            })

        return detections
