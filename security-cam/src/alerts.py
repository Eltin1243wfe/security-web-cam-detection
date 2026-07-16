"""
Stage 4: sound + Telegram push alerts. Both are best-effort — a broken
audio device or a dropped network call should never take down the
detection loop itself, so both are wrapped defensively and fail quietly
with a console line instead of raising.

Sound: shells out to paplay/aplay rather than a compiled audio library.
First version used simpleaudio, which crashed the whole process
(SIGSEGV) — almost certainly PulseAudio and simpleaudio's direct ALSA
access fighting over the same device. Shelling out sidesteps that.

Telegram: sends the actual frame as a photo, not just text, since
"someone's in the room" is a lot more useful with a picture attached.
Runs on a background thread — network calls can take a second or two,
and the detection loop shouldn't stall waiting on a slow connection
every time someone walks into frame.
"""

import shutil
import subprocess
import threading

import cv2
import requests

import config

_PLAYER = shutil.which("paplay") or shutil.which("aplay")


def play_alert():
    """
    Fires the local alert sound. Non-blocking — Popen starts the player
    and returns immediately without waiting for it to finish.
    """
    if _PLAYER is None:
        print("[alerts] no audio player found (paplay/aplay) — skipping sound")
        return

    try:
        subprocess.Popen(
            [_PLAYER, config.ALERT_SOUND_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[alerts] couldn't play sound: {e}")


def _send_telegram_photo(frame, caption: str):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[alerts] Telegram not configured (missing token/chat id) — skipping")
        return

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        print("[alerts] couldn't encode frame for Telegram")
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        response = requests.post(
            url,
            data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": ("snapshot.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=10,
        )
        if not response.ok:
            print(f"[alerts] Telegram send failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[alerts] Telegram send error: {e}")


def send_telegram_alert(frame, caption: str = "Anomaly detected"):
    threading.Thread(target=_send_telegram_photo, args=(frame, caption), daemon=True).start()