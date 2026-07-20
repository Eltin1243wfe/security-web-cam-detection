# Room Security & Anomaly Alerting

CV system that watches a room through my phone's camera, detects people/objects
with YOLO, and (eventually) flags anomalous stuff — someone in frame during
off-hours, an object left behind, motion that looks like a fall — and alerts me.

## Where this is at

- [x] Phone camera feed into Python
- [x] YOLO detection running live, boxes drawn on screen
- [x] Anomaly/event rules layer
- [x] Alerting (sound, then Telegram/Pushover)
- [ ] False-positive hardening
- [ ] One-command run + local web dashboard
- [ ] Productize for a real buyer

Right now this just shows a live window with detection boxes on it. That's it.
Everything else comes after I've actually watched the raw detections for a
while and know what I'm dealing with.

## Setup

**1. Phone side**

Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
on an Android phone. Open it, scroll down, hit "Start server". It'll show you
a local IP like `http://192.168.1.100:8080`.

(No iPhone yet — iOS equivalents exist but I haven't tested one. If I end up
on iPhone, ping me to swap this out.)

**2. Laptop side**

Phone and laptop need to be on the same WiFi network.

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open `config.py` and set `STREAM_URL` to the address IP Webcam gave you,
with `/video` on the end, e.g.:

```python
STREAM_URL = "http://192.168.1.100:8080/video"
```

**3. Run it**

```bash
python main.py
```

First run downloads the YOLOv8n weights automatically (~6MB). A window
should pop up with the live feed and green boxes around detected people.
Press `q` to quit.

## Project layout

```
main.py           - entry point, ties camera + detector + display together
config.py          - stream URL, model choice, thresholds — the stuff I tweak
src/camera.py      - wraps the phone MJPEG stream
src/detector.py    - wraps YOLO, filters by class/confidence
```

## Notes to self

- Using YOLOv8n (nano) for now — fast enough for live CPU inference, not
  the most accurate. Bump to yolov8s/m in config.py once I care more about
  missed detections than framerate.
- Only detecting "person" right now (see `CLASSES_OF_INTEREST` in config.py).
  Easy to widen once the anomaly logic needs specific objects (bags, etc.).
- No cloud, no training — pretrained weights only, per the plan.
