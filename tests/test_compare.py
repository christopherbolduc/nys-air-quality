from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nys_aq.compare import compare_snapshots, inspect_snapshot
from nys_aq.snapshot import build_snapshot, utc_text, write_snapshot

END = datetime(2026, 10, 30, 11, tzinfo=timezone.utc)


def measurement(loc_id, value, end=END, sensor_id=None, **changes):
    row = {"locationsId": loc_id, "sensorsId": sensor_id or loc_id,
           "datetime_utc": utc_text(end - timedelta(hours=1)), "parameter_name": "pm25",
           "units": "µg/m³", "value": value}
    row.update(changes)
    return row


def snapshot(rows, *, end=END, stale_hours=12, loc_ids=None):
    loc_ids = loc_ids if loc_ids is not None else sorted({r["locationsId"] for r in rows})
    locations = [{"id": i, "name": f"Station {i}", "coordinates": {"latitude": 42, "longitude": -75}}
                 for i in loc_ids]
    return build_snapshot(report_date=str((end - timedelta(days=1)).date()),
                          started_at=end - timedelta(minutes=2), ended_at=end,
                          stale_hours=stale_hours, ny_locations=locations,
                          sampled_locations=locations,
                          requests=[{"location_id": i, "outcome": "measurements"} for i in loc_ids], rows=rows)


class ComparisonTests(unittest.TestCase):
    def pair(self, before, after, **kwargs):
        return compare_snapshots(snapshot(before), snapshot(after, end=END + timedelta(days=1)), **kwargs)

    def test_broad_increase_is_numeric_traceable_and_inputs_unchanged(self):
        end = END + timedelta(days=1)
        before = snapshot([measurement(i, v) for i, v in enumerate([5, 10, 15], 1)])
        after = snapshot([measurement(i, v, end) for i, v in enumerate([10, 20, 30], 1)], end=end)
        original = copy.deepcopy((before, after))
        result = compare_snapshots(before, after, min_matched_locations=3)
        self.assertEqual((before, after), original)
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["summary"]["median_sensor_delta"], 10)
        self.assertEqual(result["summary"]["increased_sensors"], 3)
        self.assertEqual(result["summary"]["baseline_median"], 10)
        self.assertEqual(result["summary"]["current_median"], 20)
        for pair in result["pairs"]:
            self.assertEqual(pair["percent_change"], 100)
            self.assertEqual(pair["observation_gap_hours"], 24)
            for side, data in (("baseline", before), ("current", after)):
                index = int(pair[f"{side}_pointer"].rsplit("/", 1)[1])
                self.assertEqual(data["measurements"][index]["value"], pair[f"{side}_value"])
        self.assertIsNone(result["policy"]["event_thresholds"])
        json.dumps(result, allow_nan=False)

    def test_stable_and_single_outlier_do_not_become_regional_claims(self):
        end = END + timedelta(days=1)
        before = [measurement(i, 10) for i in range(1, 6)]
        stable = self.pair(before, [measurement(i, 10, end) for i in range(1, 6)])
        self.assertEqual(stable["summary"]["unchanged_sensors"], 5)
        self.assertEqual(stable["summary"]["median_sensor_delta"], 0)
        outlier = self.pair(before, [measurement(i, 100 if i == 1 else 10, end) for i in range(1, 6)])
        self.assertEqual(outlier["summary"]["increased_sensors"], 1)
        self.assertEqual(outlier["summary"]["median_sensor_delta"], 0)
        self.assertEqual(outlier["pairs"][0]["delta"], 90)
        self.assertNotIn("cause", outlier)

    def test_stale_invalid_empty_and_zero_are_distinct(self):
        end = END + timedelta(days=1)
        for after in ([], [measurement(1, None, end)],
                      [measurement(1, 0, end, datetime_utc=utc_text(end - timedelta(hours=13)))]):
            with self.subTest(after=after):
                result = self.pair([measurement(1, 10)], after)
                self.assertEqual(result["status"], "insufficient_data")
                self.assertIsNone(result["summary"])
        result = self.pair([measurement(1, 0)], [measurement(1, 5, end)])
        self.assertEqual(result["pairs"][0]["delta"], 5)
        self.assertIsNone(result["pairs"][0]["percent_change"])

    def test_sample_shift_and_units_never_become_artificial_changes(self):
        end = END + timedelta(days=1)
        result = self.pair([measurement(1, 5), measurement(2, 100)],
                           [measurement(1, 5, end), measurement(3, 1, end)])
        self.assertEqual(result["summary"]["median_sensor_delta"], 0)
        self.assertEqual(result["coverage"]["baseline_only"], [{"location_id": 2, "sensor_id": 2}])
        self.assertEqual(result["coverage"]["current_only"], [{"location_id": 3, "sensor_id": 3}])
        result = self.pair([measurement(1, 10)], [measurement(1, 0.02, end, units="mg/m³")])
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["coverage"]["current_other_units"], {"mg/m³": 1})

    def test_repeat_conflicting_repeat_regression_and_gap_excluded(self):
        end = END + timedelta(hours=1)  # Same observation can still be fresh.
        baseline = snapshot([measurement(1, 10)])
        for value, observed, reason in (
            (10, END - timedelta(hours=1), "repeated_observation"),
            (11, END - timedelta(hours=1), "conflicting_repeated_observation"),
            (11, END - timedelta(hours=2), "observation_time_regressed"),
        ):
            with self.subTest(reason=reason):
                current = snapshot([measurement(1, value, end, datetime_utc=utc_text(observed))], end=end)
                result = compare_snapshots(baseline, current)
                self.assertEqual(result["status"], "insufficient_data")
                self.assertEqual(result["exclusions"]["pairs"][0]["reason"], reason)
        end = END + timedelta(days=3)
        current = snapshot([measurement(1, 20, end)], end=end)
        result = compare_snapshots(baseline, current)
        self.assertEqual(result["collection_gap_hours"], 72)
        self.assertEqual(result["intervening_report_labels_not_examined"], 2)
        self.assertFalse(result["intervening_history_checked"])
        self.assertEqual(result["exclusions"]["pairs"][0]["reason"], "observation_gap_exceeded")
        result = compare_snapshots(baseline, current, max_observation_gap_hours=72)
        self.assertEqual(result["status"], "compared")

    def test_latest_invalid_value_and_duplicate_rows_do_not_fall_back(self):
        end = END + timedelta(days=1)
        for rows in ([measurement(1, 10, end), measurement(1, -1, end, datetime_utc=utc_text(end))],
                     [measurement(1, 10, end), measurement(1, 10, end)]):
            result = self.pair([measurement(1, 10)], rows)
            self.assertEqual(result["status"], "insufficient_data")
            self.assertTrue(result["exclusions"]["current"])

    def test_minimum_is_distinct_locations_not_sensor_count(self):
        end = END + timedelta(days=1)
        result = self.pair([measurement(1, 10, sensor_id=1), measurement(1, 20, sensor_id=2)],
                           [measurement(1, 12, end, sensor_id=1), measurement(1, 22, end, sensor_id=2)],
                           min_matched_locations=2)
        self.assertEqual(result["coverage"]["matched_sensors"], 2)
        self.assertEqual(result["coverage"]["matched_locations"], 1)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(len(result["pairs"]), 2)  # Evidence is still available.

    def test_latest_unit_change_never_falls_back_to_old_units(self):
        end = END + timedelta(days=1)
        for units in (None, "mg/m³"):
            result = self.pair([measurement(1, 10)],
                               [measurement(1, 10, end), measurement(1, 0.02, end,
                                units=units, datetime_utc=utc_text(end))])
            self.assertEqual(result["status"], "insufficient_data")
            self.assertIn("superseded_observation", [e["reason"] for e in result["exclusions"]["current"]])

    def test_percentage_overflow_remains_json_safe(self):
        end = END + timedelta(days=1)
        result = self.pair([measurement(1, 1e-300)], [measurement(1, 1e300, end)])
        self.assertIsNone(result["pairs"][0]["percent_change"])
        self.assertEqual(result["pairs"][0]["percent_change_unavailable_reason"], "numeric_overflow")
        json.dumps(result, allow_nan=False)

    def test_changed_location_and_failed_request_cannot_be_used(self):
        end = END + timedelta(days=1)
        baseline = snapshot([measurement(1, 10)])
        current = snapshot([measurement(1, 20, end)], end=end)
        current["locations"][0]["coordinates"]["latitude"] = 43
        result = compare_snapshots(baseline, current)
        self.assertEqual(result["exclusions"]["pairs"][0]["reason"], "location_coordinates_changed")
        current["locations"][0]["request"]["outcome"] = "failed"
        result = compare_snapshots(baseline, current)
        self.assertEqual(result["exclusions"]["current"][0]["reason"], "location_not_successfully_sampled")

    def test_revalidate_corrupt_flags_values_and_timestamps(self):
        for changes, reason in (({"value": float("nan")}, "invalid_value"),
                                ({"observed_at_utc": utc_text(END + timedelta(hours=1))}, "future_observation"),
                                ({"quality_flags": ["unknown_units"]}, "source_quality_exclusion")):
            result = snapshot([measurement(1, 10)])
            result["measurements"][0].update(changes)
            view = inspect_snapshot(result)
            self.assertEqual(view["status"], "insufficient_data")
            self.assertEqual(view["exclusions"][0]["reason"], reason)

    def test_bad_headers_configuration_and_order_are_errors(self):
        baseline = snapshot([measurement(1, 10)])
        end = END + timedelta(days=1)
        current = snapshot([measurement(1, 20, end)], end=end)
        for kwargs in ({"max_observation_gap_hours": 0}, {"max_observation_gap_hours": float("inf")},
                       {"min_matched_locations": 0}, {"min_matched_locations": True}, {"units": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_snapshots(baseline, current, **kwargs)
        for pair in ((current, baseline), (baseline, baseline)):
            with self.assertRaises(ValueError):
                compare_snapshots(*pair)
        for mutate in (
            lambda s: s.update(schema_version=True),
            lambda s: s.update(measurements={}),
            lambda s: s["collection"].update(freshness_reference_utc="invalid"),
            lambda s: s["collection"].update(stale_hours=24),
        ):
            bad = copy.deepcopy(current)
            mutate(bad)
            with self.assertRaises(ValueError):
                compare_snapshots(baseline, bad)

    def test_inspection_separates_units_and_uses_observation_period(self):
        view = inspect_snapshot(snapshot([measurement(1, 5), measurement(2, 15),
                                          measurement(3, 0.01, units="mg/m³")]))
        self.assertEqual(view["by_unit"]["µg/m³"]["median"], 10)
        self.assertEqual(view["by_unit"]["mg/m³"]["sensors"], 1)
        self.assertEqual(view["by_unit"]["µg/m³"]["observation_start_utc"], utc_text(END - timedelta(hours=1)))

    def test_cli_reads_without_changing_files_and_rejects_missing_input(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_snapshots.py"
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "before.json", Path(tmp) / "after.json"
            end = END + timedelta(days=1)
            write_snapshot(a, snapshot([measurement(1, 10)]))
            write_snapshot(b, snapshot([measurement(1, 20, end)], end=end))
            original = (a.read_bytes(), b.read_bytes())
            # Source-tree tests must not depend on an editable package install.
            env = dict(os.environ, PYTHONPATH=str(script.parents[1] / "src"))
            process = subprocess.run([sys.executable, str(script), str(a), str(b)], env=env,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["pairs"][0]["delta"], 10)
            self.assertEqual((a.read_bytes(), b.read_bytes()), original)
            self.assertEqual(len(list(Path(tmp).iterdir())), 2)
            inspect = subprocess.run([sys.executable, str(script), str(a)], env=env, capture_output=True, text=True)
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            self.assertEqual(json.loads(inspect.stdout)["status"], "available")
            missing = subprocess.run([sys.executable, str(script), str(a.parent / "missing.json")], env=env, capture_output=True)
            self.assertEqual(missing.returncode, 2)


if __name__ == "__main__":
    unittest.main()
