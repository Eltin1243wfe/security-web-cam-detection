"""
Wraps the phone feed so the rest of the code doesn't care that the
source is Iriun Webcam specifically. If I switch to an RTSP camera or
a different device later, this is the only file that should need to
change.

Iriun exposes the phone as a normal v4l2 device (/dev/videoN) once the
app is open on the phone and the desktop client is connected, so this
takes a device index rather than a URL — same VideoCapture call either
way, OpenCV doesn't care which you hand it.
"""

import cv2


class PhoneCamera:
    def __init__(self, source: int):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ConnectionError(
                f"Couldn't open camera source {source}. "
                "Check that Iriun Webcam is open on the phone and the "
                "desktop client shows it connected. Run "
                "`v4l2-ctl --list-devices` if the index may have changed."
            )

    def read_frame(self):
        """Returns a single BGR frame, or None if the stream drops."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()