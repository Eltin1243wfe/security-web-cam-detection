"""
Sits between raw YOLO detections and any future alerting. Its only job
right now is deciding whether what got detected is actually worth caring
about — nothing here sends a notification yet, that's stage 4. For now
it just prints to the console when something crosses the line into
"anomaly", so I can watch the logic work before wiring up real alerts.

Problems this solves, all found during actual testing rather than
guessed upfront:
1. A single frame saying "person" isn't trustworthy — a chair and a
   towel both triggered one-frame false positives during testing.
2. Not every real person in frame is something I want to be alerted
   about (e.g. me getting water at night) — the system needs an
   armed/disarmed concept, same as any normal house alarm.
3. "A person exists somewhere in frame" isn't the same as "the same
   person has continuously been here" — if detection flickers between
   two unrelated spots (e.g. a real person, then briefly a shadow
   elsewhere), naively counting consecutive frames would treat that as
   one continuous presence when it isn't. Stage 5 adds a spatial check
   on top of the frame-count check for exactly this reason.
"""

from datetime import datetime
import math


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


class PresenceTracker:
    """
    Turns per-frame detections into a stable "someone is actually here"
    signal by requiring several consecutive frames before trusting it —
    and, as of stage 5, requiring those frames to be roughly the same
    person in roughly the same spot, not just "a person somewhere."

    One tracker instance = one continuous stretch of presence. Any gap
    (no person detected) or any large jump in position resets the count
    — so someone has to be visible, in a consistent spot, for the full
    threshold again before it counts as sustained. Deliberately strict
    for now; if real people keep tripping the drift check while just
    moving normally, the fix is raising max_center_drift, not removing
    the check.
    """

    def __init__(self, consecutive_frames_required: int, max_center_drift: float = float("inf")):
        self.consecutive_frames_required = consecutive_frames_required
        self.max_center_drift = max_center_drift
        self._consecutive_count = 0
        self._sustained = False
        self._last_center = None

    @staticmethod
    def _best_person_box(detections):
        # If more than one person's in frame, track whichever detection
        # YOLO is most confident about — good enough for a single
        # continuous "is someone here" signal. Tracking every person
        # independently is a real upgrade but a bigger one than this
        # stage needs.
        person_boxes = [d for d in detections if d["label"] == "person"]
        if not person_boxes:
            return None
        return max(person_boxes, key=lambda d: d["confidence"])["box"]

    def update(self, detections: list) -> bool:
        """
        Call once per frame with that frame's detections (already
        filtered by class/confidence/size upstream). Returns True on
        exactly the frame where presence becomes sustained — not on
        every frame after — so callers can treat True as "a new event
        just started."
        """
        person_box = self._best_person_box(detections)

        if person_box is None:
            self._consecutive_count = 0
            self._sustained = False
            self._last_center = None
            return False

        current_center = _box_center(person_box)

        if self._last_center is not None:
            drift = _distance(self._last_center, current_center)
            if drift > self.max_center_drift:
                # Jumped too far to plausibly be the same continuous
                # presence — treat this as a fresh start, not a
                # continuation of what came before.
                self._consecutive_count = 0
                self._sustained = False

        self._last_center = current_center
        self._consecutive_count += 1

        if self._consecutive_count >= self.consecutive_frames_required and not self._sustained:
            self._sustained = True
            return True

        return False


class CooldownGate:
    """
    Rate-limits how often an anomaly is allowed to actually fire, separate
    from whether presence is "sustained" or not. PresenceTracker resets
    the instant a frame drops, so a person standing in frame can trip
    "event started" repeatedly within seconds if YOLO loses and re-finds
    them — this is what stops that from becoming a spam of alerts once
    stage 4 wires up real notifications.
    """

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._last_fired_at: datetime | None = None

    def should_fire(self, now: datetime | None = None) -> bool:
        """
        Call when PresenceTracker.update() returns True. Returns whether
        enough time has passed since the last fire to allow another one.
        Does NOT fire on its own — caller decides what "firing" means
        (print, sound, notification) and should only do it if this
        returns True.
        """
        now = now or datetime.now()

        if self._last_fired_at is not None:
            elapsed = (now - self._last_fired_at).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        self._last_fired_at = now
        return True


def is_armed(state, now: datetime | None = None) -> bool:
    """
    Whether the system should currently treat sustained presence as an
    anomaly worth alerting on. Reads from the live SharedState rather
    than static config — as of stage 6, armed status and the schedule
    are toggleable from the dashboard at runtime, so config.py only
    supplies the *initial* values state gets seeded with at startup.

    state.is_armed_override_enabled() is the manual override — if
    that's off, nothing else matters. If it's on and the schedule is
    enabled, arming is further restricted to the configured hour window
    (e.g. midnight-6am, when nobody should reasonably be up).
    """
    if not state.is_armed_override_enabled():
        return False

    use_time_window, start, end = state.get_schedule()
    if not use_time_window:
        return True

    now = now or datetime.now()

    if start < end:
        return start <= now.hour < end
    return now.hour >= start or now.hour < end