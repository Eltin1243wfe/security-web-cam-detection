"""
Sits between raw YOLO detections and any future alerting. Its only job
right now is deciding whether what got detected is actually worth caring
about — nothing here sends a notification yet, that's stage 4. For now
it just prints to the console when something crosses the line into
"anomaly", so I can watch the logic work before wiring up real alerts.

Two problems this solves, both found during actual testing rather than
guessed upfront:
1. A single frame saying "person" isn't trustworthy — a chair and a
   towel both triggered one-frame false positives during testing.
2. Not every real person in frame is something I want to be alerted
   about (e.g. me getting water at night) — the system needs an
   armed/disarmed concept, same as any normal house alarm.
"""

from datetime import datetime

import config


class PresenceTracker:
    """
    Turns per-frame detections into a stable "someone is actually here"
    signal by requiring several consecutive frames before trusting it.

    One tracker instance = one continuous stretch of presence. Any gap
    (a frame with nobody detected) resets the count — so someone has to
    be visible for the full threshold again before it counts as sustained.
    That's deliberately strict for now; if it turns out real people keep
    dropping a frame here and there and getting reset unfairly, the fix
    is tolerance for brief gaps, not a lower threshold.
    """

    def __init__(self, consecutive_frames_required: int):
        self.consecutive_frames_required = consecutive_frames_required
        self._consecutive_count = 0
        self._sustained = False

    def update(self, person_detected: bool) -> bool:
        """
        Call once per frame with whether a person was detected in it.
        Returns True on exactly the frame where presence becomes
        sustained (crosses the threshold) — not on every frame after,
        so callers can treat True as "a new event just started."
        """
        if not person_detected:
            self._consecutive_count = 0
            self._sustained = False
            return False

        self._consecutive_count += 1

        if self._consecutive_count >= self.consecutive_frames_required and not self._sustained:
            self._sustained = True
            return True

        return False


def is_armed(now: datetime | None = None) -> bool:
    """
    Whether the system should currently treat sustained presence as an
    anomaly worth alerting on. SYSTEM_ARMED is the manual override — if
    that's off, nothing else matters. If it's on and USE_TIME_WINDOW is
    set, arming is further restricted to ARMED_START_HOUR-ARMED_END_HOUR
    (e.g. midnight-6am, when nobody should reasonably be up).
    """
    if not config.SYSTEM_ARMED:
        return False

    if not config.USE_TIME_WINDOW:
        return True

    now = now or datetime.now()
    start, end = config.ARMED_START_HOUR, config.ARMED_END_HOUR

    # Same comparison either way, just depends whether the window wraps
    # past midnight (e.g. 22 -> 6) or not (e.g. 1 -> 5).
    if start < end:
        return start <= now.hour < end
    return now.hour >= start or now.hour < end

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