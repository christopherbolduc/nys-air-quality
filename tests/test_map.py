from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nys_aq.daily import _svg_map

NS = {"svg": "http://www.w3.org/2000/svg"}
END = datetime(2026, 10, 31, 11, tzinfo=timezone.utc)


def reading(loc_id=1, **changes):
    result = {"locationsId": loc_id, "sensorsId": loc_id, "parameter_name": "pm25",
              "units": "µg/m³", "value": 10.0, "datetime_utc": "2026-10-31T10:00:00Z",
              "stale": False}
    result.update(changes)
    return result


class ConcentrationMapTests(unittest.TestCase):
    def render(self, rows, *, count=1, units="µg/m³", stale_hours=12):
        with tempfile.TemporaryDirectory() as tmp:
            path = _svg_map(
                Path(tmp), report_date="2026-10-30", sample_label="Test locations",
                ny_multipolygon=[[[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]],
                ny_locations=[{"id": i, "name": "A & B <Station>",
                               "coordinates": {"longitude": 0.5 + i / 10, "latitude": 1.0}}
                              for i in range(1, count + 1)],
                rows=rows, primary_param="pm25", units=units,
                collection_started_at=END, collection_ended_at=END,
                stale_hours=stale_hours,
            )
            root = ET.parse(path).getroot()
            text = " ".join(root.itertext())
            return root, text

    def test_concentration_label_timing_tooltips_and_no_health_categories(self):
        root, text = self.render([reading()])
        self.assertIn("PM2.5 concentration (latest)", text)
        self.assertIn("Report label: 2026-10-30", text)
        self.assertIn("Collected (UTC): 2026-10-31T11:00:00Z", text)
        self.assertIn("observed 2026-10-31T10:00:00Z", text)
        self.assertIn("10 µg/m³", text)
        self.assertIn("A & B <Station>", text)
        self.assertIn("Not an AQI or a daily average", text)
        self.assertNotIn("Good (", text)
        self.assertNotIn("Unhealthy", text)
        self.assertEqual(len(root.findall("svg:circle", NS)), 1)

    def test_invalid_latest_value_never_falls_back_to_older_value(self):
        for value in (None, -1, float("nan"), float("inf"), True, "10"):
            with self.subTest(value=value):
                root, text = self.render([reading(), reading(value=value, datetime_utc="2026-10-31T10:30:00Z")])
                self.assertIn("Fresh: 0", text)
                self.assertIn("No fresh valid readings", text)
                self.assertEqual(root.find("svg:circle", NS).get("fill"), "lightgray")

    def test_incompatible_units_and_bad_times_are_unavailable(self):
        for changes in ({"units": "mg/m³"}, {"units": None},
                        {"datetime_utc": "invalid"}, {"datetime_utc": "2026-10-31T10:00:00"},
                        {"datetime_utc": "2026-10-31T11:00:01Z"}):
            with self.subTest(changes=changes):
                _, text = self.render([reading(**changes)])
                self.assertIn("unavailable: 1", text)
                self.assertIn("No fresh valid readings", text)
        _, text = self.render([reading()], units="")
        self.assertIn("units unavailable", text)
        self.assertIn("No fresh valid readings", text)

    def test_stale_and_missing_points_cannot_affect_fresh_scale(self):
        root, text = self.render([reading(value=5), reading(2, value=500, stale=True,
                                  datetime_utc="2026-10-30T10:00:00Z")], count=3)
        self.assertIn("Fresh: 1 • stale: 1 • unavailable: 1", text)
        self.assertIn("5.00", text)
        self.assertNotIn("500.00", text)
        circles = root.findall("svg:circle", NS)
        self.assertEqual(circles[1].get("fill"), "rgb(180,180,180)")
        self.assertEqual(circles[2].get("fill"), "lightgray")

    def test_empty_or_all_stale_has_no_scale(self):
        for rows in ([], [reading(stale=True, datetime_utc="2026-10-30T10:00:00Z")]):
            with self.subTest(rows=rows):
                root, text = self.render(rows)
                self.assertIn("No fresh valid readings", text)
                # The only rectangles are the background and main panel.
                self.assertEqual(len(root.findall("svg:rect", NS)), 2)

    def test_freshness_uses_collection_end_and_configured_boundary(self):
        rows = [reading(1, datetime_utc="2026-10-31T09:00:00Z", stale=True),
                reading(2, datetime_utc="2026-10-31T08:59:59Z", stale=False)]
        _, text = self.render(rows, count=2, stale_hours=2)
        self.assertIn("Fresh: 1 • stale: 1", text)

    def test_relative_scale_percentiles_and_monotonic_color(self):
        root, text = self.render([reading(1, value=0), reading(2, value=100)], count=2)
        self.assertIn("5.00", text)
        self.assertIn("95.00", text)
        self.assertIn("compare numeric values across dates", text)
        circles = root.findall("svg:circle", NS)
        self.assertEqual(circles[0].get("fill"), "rgb(190,219,240)")
        self.assertEqual(circles[1].get("fill"), "rgb(8,70,125)")
        self.assertEqual(circles[0].get("r"), circles[1].get("r"))

    def test_observation_times_sorted_as_instants_and_conflicts_not_arbitrary(self):
        root, text = self.render([reading(value=4, datetime_utc="2026-10-31T10:00:00Z"),
                                  reading(value=9, datetime_utc="2026-10-31T06:30:00-04:00")])
        self.assertIn("9 µg/m³", text)
        self.assertIn("observed 2026-10-31T10:30:00Z", text)
        _, text = self.render([reading(value=4), reading(value=9)])
        self.assertIn("No fresh valid readings", text)


if __name__ == "__main__":
    unittest.main()
