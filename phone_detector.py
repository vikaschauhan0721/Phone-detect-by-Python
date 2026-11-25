"""Detect mobile phones from a webcam and alert during study sessions."""

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path


PHONE_CLASS_ID = 67


@dataclass(frozen=True)
class NoPhoneZone:
    """A no-phone rectangle expressed as normalized frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
            raise ValueError("zone coordinates must be finite values between 0 and 1")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("zone coordinates must be ordered as x1,y1,x2,y2")

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def parse_zone(value):
    try:
        coordinates = tuple(float(part) for part in value.split(","))
        if len(coordinates) != 4:
            raise ValueError
        return NoPhoneZone(*coordinates)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "zone must be four ordered coordinates between 0 and 1: x1,y1,x2,y2"
        ) from error


def study_session_active(now, started_at, study_minutes, break_minutes):
    """Return whether a repeating study/break cycle is currently in its study phase."""
    if study_minutes <= 0 or break_minutes < 0:
        raise ValueError("study minutes must be positive and break minutes non-negative")
    study_seconds = study_minutes * 60
    cycle_seconds = (study_minutes + break_minutes) * 60
    return (now - started_at) % cycle_seconds < study_seconds


def should_alert(phone_detected, phone_in_zone, study_active):
    return phone_in_zone or (phone_detected and study_active)


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def non_negative_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return number


def positive_float(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return number


def play_alert(sound_path=None):
    """Play a supplied sound or the platform's default alert sound."""
    if sys.platform == "win32":
        import winsound

        if sound_path:
            winsound.PlaySound(
                str(sound_path), winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        else:
            winsound.MessageBeep(winsound.MB_ICONWARNING)
        return

    player = shutil.which("afplay") if sys.platform == "darwin" else None
    if sys.platform.startswith("linux"):
        player = shutil.which("paplay") or shutil.which("aplay")

    if player:
        sound = None
        if sound_path:
            sound = str(sound_path)
        elif sys.platform == "darwin":
            sound = "/System/Library/Sounds/Glass.aiff"
        elif player.endswith("paplay"):
            sound = "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"
        else:
            sound = "/usr/share/sounds/alsa/Front_Center.wav"

        temporary_sound = None
        try:
            if sound and os.path.isfile(sound):
                result = subprocess.run(
                    [player, sound],
                    check=False,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    return
            if not sound_path and sys.platform.startswith("linux"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                    temporary_sound = file.name
                sample_rate = 44100
                samples = (
                    struct.pack(
                        "<h",
                        int(12000 * math.sin(2 * math.pi * 880 * index / sample_rate)),
                    )
                    for index in range(sample_rate // 3)
                )
                with wave.open(temporary_sound, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(sample_rate)
                    audio.writeframes(b"".join(samples))
                subprocess.run(
                    [player, temporary_sound],
                    check=False,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            if temporary_sound:
                try:
                    os.unlink(temporary_sound)
                except OSError:
                    pass

    print("\a", end="", flush=True)


def detect_phones(model, frame, confidence):
    results = model(frame, conf=confidence, classes=[PHONE_CLASS_ID], verbose=False)
    phones = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0].item()) == PHONE_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                phones.append(
                    (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(box.conf[0].item()),
                    )
                )
    return phones


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect phones from a webcam and sound an alert."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument(
        "--weights", default="yolov8n.pt", help="YOLOv8 COCO model weights"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.45, help="minimum detection confidence"
    )
    parser.add_argument(
        "--zone",
        action="append",
        type=parse_zone,
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="normalized no-phone rectangle; may be supplied more than once",
    )
    parser.add_argument(
        "--study-minutes", type=positive_int, default=25, help="study phase length"
    )
    parser.add_argument(
        "--break-minutes",
        type=non_negative_int,
        default=5,
        help="break phase length",
    )
    parser.add_argument(
        "--no-study-session",
        action="store_false",
        dest="study_session",
        help="alert only when a phone is inside a no-phone zone",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=positive_float,
        default=3.0,
        help="minimum seconds between audio alerts",
    )
    parser.add_argument(
        "--alert-sound",
        type=Path,
        help="optional audio file to play instead of the platform default",
    )
    return parser


def run_app(args):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Install the application dependencies with: python -m pip install -r requirements.txt"
        ) from error

    model = YOLO(args.weights)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Could not open webcam {args.camera}")

    started_at = time.monotonic()
    last_alert = float("-inf")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.", file=sys.stderr)
                break

            height, width = frame.shape[:2]
            phones = detect_phones(model, frame, args.confidence)
            phone_in_zone = False

            for zone in args.zone:
                top_left = (int(zone.x1 * width), int(zone.y1 * height))
                bottom_right = (int(zone.x2 * width), int(zone.y2 * height))
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "NO-PHONE ZONE",
                    (top_left[0], max(20, top_left[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            for x1, y1, x2, y2, confidence in phones:
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                in_zone = any(zone.contains(center_x, center_y) for zone in args.zone)
                phone_in_zone = phone_in_zone or in_zone
                color = (0, 0, 255) if in_zone else (0, 200, 0)
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )
                cv2.putText(
                    frame,
                    f"phone {confidence:.2f}",
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            now = time.monotonic()
            study_active = args.study_session and study_session_active(
                now, started_at, args.study_minutes, args.break_minutes
            )
            status = "STUDY" if study_active else (
                "BREAK" if args.study_session else "STUDY MODE OFF"
            )
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0) if study_active else (255, 200, 0),
                2,
            )

            if should_alert(bool(phones), phone_in_zone, study_active):
                if now - last_alert >= args.alert_cooldown:
                    threading.Thread(
                        target=play_alert,
                        args=(args.alert_sound,),
                        daemon=True,
                    ).start()
                    last_alert = now

            cv2.imshow("Phone detector (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.alert_sound and not args.alert_sound.is_file():
        parser.error("--alert-sound must refer to an existing audio file")
    run_app(args)


if __name__ == "__main__":
    main()
