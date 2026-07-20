"""
Single shared state object the detection thread writes to and the Flask
dashboard reads from (and occasionally writes to — the armed toggle,
the schedule, the "It's OK" button all go through here).

Everything's behind one lock. Update rate here is at most a handful of
times per second, so lock contention was never going to be the
bottleneck — one lock is a lot easier to reason about correctly than
trying to fine-tune which individual fields need their own.
"""

import threading


class SharedState:
    def __init__(self, armed: bool, use_time_window: bool, armed_start_hour: int, armed_end_hour: int):
        self._lock = threading.Lock()
        self._armed_override = armed
        self._use_time_window = use_time_window
        self._armed_start_hour = armed_start_hour
        self._armed_end_hour = armed_end_hour
        self._acknowledged = False  # "It's OK" — silences alerts for the CURRENT presence session only
        self._latest_jpeg = None
        self._alert_log = []
        self._max_log_entries = 50

    # --- armed state ------------------------------------------------------

    def is_armed_override_enabled(self) -> bool:
        with self._lock:
            return self._armed_override

    def set_armed_override(self, value: bool):
        with self._lock:
            self._armed_override = value

    def get_schedule(self):
        with self._lock:
            return self._use_time_window, self._armed_start_hour, self._armed_end_hour

    def set_schedule(self, use_time_window: bool, start_hour: int, end_hour: int):
        with self._lock:
            self._use_time_window = use_time_window
            self._armed_start_hour = start_hour
            self._armed_end_hour = end_hour

    # --- "It's OK" ----------------------------------------------------------

    def acknowledge(self):
        with self._lock:
            self._acknowledged = True

    def is_acknowledged(self) -> bool:
        with self._lock:
            return self._acknowledged

    def clear_acknowledged(self):
        # Called by the detection loop the moment presence actually ends
        # (person leaves frame) — this is what stops an acknowledgment
        # from silently covering a completely different visitor later.
        with self._lock:
            self._acknowledged = False

    # --- video frame --------------------------------------------------------

    def set_latest_jpeg(self, jpeg_bytes: bytes):
        with self._lock:
            self._latest_jpeg = jpeg_bytes

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    # --- alert log ------------------------------------------------------------

    def add_alert(self, timestamp: str, snapshot_filename: str):
        with self._lock:
            self._alert_log.insert(0, {"timestamp": timestamp, "snapshot_filename": snapshot_filename})
            self._alert_log = self._alert_log[: self._max_log_entries]

    def get_alert_log(self):
        with self._lock:
            return list(self._alert_log)