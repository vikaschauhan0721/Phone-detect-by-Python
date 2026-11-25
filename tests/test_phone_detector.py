import argparse
import unittest
from types import SimpleNamespace

from phone_detector import (
    NoPhoneZone,
    detect_phones,
    parse_zone,
    should_alert,
    study_session_active,
)


class NoPhoneZoneTests(unittest.TestCase):
    def test_parse_normalized_zone(self):
        zone = parse_zone("0.1,0.2,0.8,0.9")

        self.assertEqual(zone, NoPhoneZone(0.1, 0.2, 0.8, 0.9))
        self.assertTrue(zone.contains(0.5, 0.5))
        self.assertFalse(zone.contains(0.9, 0.5))

    def test_parse_rejects_invalid_coordinates(self):
        for value in ("0,0,1", "0.8,0.2,0.1,0.9", "0,0,1.1,1", "nan,0,1,1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_zone(value)


class AlertRuleTests(unittest.TestCase):
    def test_alerts_for_phone_in_zone_or_during_study(self):
        self.assertTrue(should_alert(False, True, False))
        self.assertTrue(should_alert(True, False, True))
        self.assertFalse(should_alert(True, False, False))
        self.assertFalse(should_alert(False, False, True))

    def test_study_and_break_phases_repeat(self):
        start = 100.0
        self.assertTrue(study_session_active(100.0, start, 25, 5))
        self.assertTrue(study_session_active(1599.0, start, 25, 5))
        self.assertFalse(study_session_active(1600.0, start, 25, 5))
        self.assertFalse(study_session_active(1899.0, start, 25, 5))
        self.assertTrue(study_session_active(1900.0, start, 25, 5))

    def test_study_schedule_requires_valid_durations(self):
        with self.assertRaises(ValueError):
            study_session_active(0, 0, 0, 5)
        with self.assertRaises(ValueError):
            study_session_active(0, 0, 25, -1)


class PhoneDetectionTests(unittest.TestCase):
    def test_keeps_only_phone_detections(self):
        def tensor(value):
            return SimpleNamespace(item=lambda: value)

        def box(class_id, coordinates, confidence):
            return SimpleNamespace(
                cls=[tensor(class_id)],
                xyxy=[SimpleNamespace(tolist=lambda: coordinates)],
                conf=[tensor(confidence)],
            )

        class FakeModel:
            def __call__(self, frame, **options):
                self.options = options
                return [
                    SimpleNamespace(
                        boxes=[
                            box(67, [1, 2, 30, 40], 0.9),
                            box(0, [5, 6, 20, 25], 0.8),
                        ]
                    )
                ]

        model = FakeModel()
        phones = detect_phones(model, object(), 0.4)

        self.assertEqual(phones, [(1.0, 2.0, 30.0, 40.0, 0.9)])
        self.assertEqual(model.options["classes"], [67])
        self.assertEqual(model.options["conf"], 0.4)


if __name__ == "__main__":
    unittest.main()
