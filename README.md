# Phone-detect-by-Python

A real-time webcam app that detects mobile phones with YOLOv8 and OpenCV. It
plays an audio alert when a phone is seen during a study session or inside a
configured no-phone zone.

## Setup

Use Python 3.9 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app with your default webcam:

```bash
python phone_detector.py
```

The first run downloads the default `yolov8n.pt` model. The study timer starts
with a 25-minute study phase followed by a 5-minute break, repeating until the
app exits. Press `q` in the camera window to quit.

## Options

No-phone zones are rectangles in normalized frame coordinates, from `0` (left
or top) to `1` (right or bottom). Supply `--zone` more than once to add zones:

```bash
python phone_detector.py --zone 0.1,0.2,0.5,0.9 --zone 0.6,0.2,0.9,0.9
```

Phones trigger an alert in a no-phone zone at any time and anywhere in the
frame during a study phase. For zone-only alerts, use `--no-study-session`.
Study and break lengths, camera index, detection confidence, and alert cooldown
can also be configured:

```bash
python phone_detector.py --study-minutes 30 --break-minutes 5 --camera 0
```

Use `--alert-sound path/to/sound.wav` to play a sound file instead of the
platform's default alert. On systems without an available audio player, the app
falls back to the terminal bell.

Run the focused tests with:

```bash
python -m unittest discover -s tests
```
