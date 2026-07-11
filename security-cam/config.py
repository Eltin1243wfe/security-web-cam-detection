"""
Single place for the knobs I'll actually want to turn while testing.

I'm keeping this as a plain Python module instead of YAML/JSON for now —
no parsing, no schema, just import and go. If this grows past ~30 lines
or I need per-environment configs, I'll switch to YAML then.
"""

# Iriun Webcam exposes the phone as a normal Linux video device once
# the app is open on the phone and the desktop client is connected —
# no URL needed, just the device index.
# Check yours with: v4l2-ctl --list-devices
CAMERA_SOURCE = 4

# YOLOv8n = nano. Fastest, least accurate. Good enough to prove the
# pipeline works on a laptop CPU. Swap to yolov8s/m once detection
# quality actually matters more than iteration speed.
MODEL_NAME = "yolov8n.pt"

# Detections below this get thrown out. 0.5 is a reasonable starting
# point — lower it if YOLO's missing obvious people, raise it if the
# feed is full of junk boxes on shadows/furniture.
CONFIDENCE_THRESHOLD = 0.5

# Only draw/log these classes. Full COCO list has 80 classes and most
# of them (toaster, kite, etc.) are noise for a security use case.
CLASSES_OF_INTEREST = ["person"]

# Resize incoming frames to this width before running inference.
# Smaller = faster detection, at the cost of missing small/far objects.
# 640 is the YOLO training resolution so it's a sane default.
FRAME_WIDTH = 800

# --- Stage 3: anomaly rules ---------------------------------------------

# Manual kill switch. False = system never treats detections as anomalies,
# full stop, regardless of time. No dashboard to flip this yet (that's
# stage 6), so for now it's a hand-edit here — e.g. set False if I've got
# guests over and don't want to think about the schedule at all.
SYSTEM_ARMED = True

# If True, SYSTEM_ARMED alone isn't enough to arm — detections only count
# as anomalies during the hours below too. If False, SYSTEM_ARMED is the
# only gate and the system is armed any time it's on.
USE_TIME_WINDOW = True

# Window is [start, end) in 24h format. Handles wrapping past midnight
# fine (e.g. 22 -> 6) — see is_armed() in src/rules.py for the logic.
ARMED_START_HOUR = 0   # midnight
ARMED_END_HOUR = 6     # 6am

# How many consecutive frames a person needs to appear in before it
# counts as a real event instead of noise — this is the fix for the
# chair/towel one-frame flicker I saw during testing. At ~15fps this is
# roughly 1 second of continuous presence. Raise it if still getting
# false positives, lower it if real presence feels slow to register.
CONSECUTIVE_FRAMES_THRESHOLD = 15
