"""
Stage 4, part 1: local sound alerts. Telegram/Pushover come after this —
wanted something working end-to-end before adding an external service
and its account setup into the mix.

First version used simpleaudio, which talks to ALSA directly via a
compiled C extension — that crashed the whole process (SIGSEGV) the
moment it tried to play, almost certainly because PulseAudio already
owns the audio device on a normal Ubuntu desktop and doesn't like
something else grabbing ALSA underneath it. Shelling out to `paplay`
sidesteps that entirely: it's the standard PulseAudio playback command,
already installed on basically every Ubuntu desktop, and there's no
compiled extension in my own process that can take the whole thing down.
"""

import shutil
import subprocess

import config

# Checked once at import time rather than on every alert — no reason to
# re-search PATH for the same binary a hundred times a session.
_PLAYER = shutil.which("paplay") or shutil.which("aplay")


def play_alert():
    """
    Fires the alert sound. Non-blocking — Popen starts the player and
    returns immediately without waiting for it to finish, so the
    detection loop keeps running while the sound plays.

    Wrapped in a try/except on purpose: a missing/broken audio setup
    shouldn't take down detection. Worst case I miss a beep, not the
    whole system — which is exactly the failure mode I just hit with
    simpleaudio crashing the entire process.
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