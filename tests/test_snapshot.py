from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nys_aq.daily import DailyConfig, fetch_latest_for_locations, prune_old_artifacts, run_daily
from nys_aq.snapshot import build_snapshot, read_snapshot, utc_text, write_snapshot

END = datetime(2026, 10, 31, 11, tzinfo=timezone.utc)
START = END - timedelta(minutes=2)


def location(loc_id=1):
    return {"id": loc_id, "name": f"Station {loc_id}",
            "coordinates": {"longitude": 1.0, "latitude": 1.0}}


def row(**changes):
    result = {"locationsId": 1, "sensorsId": 7, "datetime_utc": "2026-10-31T10:00:00Z",
              "parameter_name": "pm25", "units": "µg/m³", "value": 8.5, "stale": False}
    result.update(changes)
    return result


def snapshot(rows=None, **changes):
    kwargs = dict(report_date="2026-10-30", started_at=START, ended_at=END, stale_hours=12,
                  ny_locations=[location()], sampled_locations=[location()],
                  requests=[{"location_id": 1, "outcome": "measurements",
                             "started_at_utc": utc_text(START), "ended_at_utc": utc_text(END)}],
                  rows=[row()] if rows is None else rows)
    kwargs.update(changes)
    return build_snapshot(**kwargs)


def config(root):
    return DailyConfig(bbox="test", sample_size=100, stale_hours=12,
                       run_date=date(2026, 10, 30), repo_root=root,
                       ny_boundary_geojson=root / "data" / "nys_boundary.geojson",
                       retention_days=91, _api_key="secret-test-key")


class SnapshotTests(unittest.TestCase):
    def test_round_trip_and_input_unchanged(self):
        rows = [row(), row(sensorsId=8, parameter_name="o3", units="ppm", value=0.02)]
        original = copy.deepcopy(rows)
        result = snapshot(rows)
        self.assertEqual(rows, original)
        self.assertEqual(result["quality_summary"]["usable"], 2)
        self.assertEqual(result["collection"]["freshness_reference_utc"], utc_text(END))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_snapshot(Path(tmp) / "snapshot.json", result)
            self.assertEqual(read_snapshot(path, expected_collection_start=utc_text(START)), result)
            with self.assertRaisesRegex(ValueError, "expected collection"):
                read_snapshot(path, expected_collection_start="another-run")

    def test_invalid_readings_are_preserved_and_excluded(self):
        cases = [({"value": None}, "missing_value"), ({"value": float("nan")}, "nonfinite_value"),
                 ({"value": float("inf")}, "nonfinite_value"), ({"value": -1}, "negative_value"),
                 ({"value": "8.5"}, "invalid_value"), ({"value": True}, "invalid_value"),
                 ({"datetime_utc": None}, "missing_timestamp"),
                 ({"datetime_utc": "not-a-time"}, "invalid_timestamp"),
                 ({"datetime_utc": "2026-10-31T10:00:00"}, "invalid_timestamp"),
                 ({"datetime_utc": "2026-10-31T11:00:01Z"}, "future_timestamp"),
                 ({"parameter_name": None}, "unknown_parameter"),
                 ({"units": ""}, "unknown_units"), ({"locationsId": 99}, "invalid_location_id"),
                 ({"sensorsId": None}, "invalid_sensor_id")]
        for changes, flag in cases:
            with self.subTest(changes=changes):
                result = snapshot([row(**changes)])
                measurement = result["measurements"][0]
                self.assertIn(flag, measurement["quality_flags"])
                self.assertFalse(measurement["usable"])
                self.assertEqual(result["quality_summary"]["invalid"], 1)
                json.dumps(result, allow_nan=False)

    def test_freshness_boundary_and_timezone_normalization(self):
        rows = [row(sensorsId=7, datetime_utc="2026-10-30T23:00:00Z"),
                row(sensorsId=8, datetime_utc="2026-10-30T22:59:59Z"),
                row(sensorsId=9, datetime_utc="2026-10-31T06:00:00-04:00")]
        measurements = snapshot(rows)["measurements"]
        self.assertEqual([m["stale"] for m in measurements], [False, True, False])
        self.assertEqual([m["usable"] for m in measurements], [True, False, True])
        self.assertEqual(measurements[2]["observed_at_utc"], "2026-10-31T10:00:00Z")

    def test_all_duplicate_copies_excluded_and_conflicts_flagged(self):
        for second, conflict in ((row(), False), (row(value=100), True)):
            with self.subTest(conflict=conflict):
                result = snapshot([row(), second])
                self.assertEqual(result["quality_summary"]["duplicate_rows"], 2)
                self.assertEqual(result["quality_summary"]["usable"], 0)
                for m in result["measurements"]:
                    self.assertEqual("conflicting_duplicate" in m["quality_flags"], conflict)
        # Different units are retained separately, never silently converted.
        self.assertEqual(snapshot([row(), row(units="mg/m³")])["quality_summary"]["duplicate_rows"], 0)

    def test_empty_failed_and_unsampled_are_distinct(self):
        result = snapshot([], ny_locations=[location(i) for i in (1, 2, 3)],
                          sampled_locations=[location(1), location(2)],
                          requests=[{"location_id": 1, "outcome": "empty"},
                                    {"location_id": 2, "outcome": "failed", "error_type": "Timeout"}])
        self.assertEqual([loc["request"]["outcome"] for loc in result["locations"]],
                         ["empty", "failed", "unsampled"])
        self.assertEqual(result["quality_summary"]["returned"], 0)
        self.assertEqual(result["quality_summary"]["usable"], 0)
        self.assertEqual(result["quality_summary"]["fetch_failures"], 1)
        with self.assertRaisesRegex(ValueError, "Missing request"):
            snapshot(requests=[])

    def test_atomic_failure_retains_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            write_snapshot(path, snapshot())
            before = path.read_bytes()
            with patch("nys_aq.snapshot.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    write_snapshot(path, snapshot([row(value=20)]))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises(ValueError):
                write_snapshot(path, {"schema_version": 1, "bad": float("nan")})
            self.assertEqual(path.read_bytes(), before)

    def test_rerun_replaces_snapshot_and_versions_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            write_snapshot(path, snapshot())
            write_snapshot(path, snapshot([row(value=20)]))
            self.assertEqual(read_snapshot(path)["measurements"][0]["value"], 20)
            path.write_text('{"schema_version": 2}')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_snapshot(path)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                write_snapshot(path, {"schema_version": 2})

    def test_retention_removes_snapshot_with_folder_and_keeps_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            cutoff = cfg.run_date - timedelta(days=cfg.retention_days)
            old = root / "reports" / str(cutoff - timedelta(days=1)) / "snapshot.json"
            retained = root / "reports" / str(cutoff) / "snapshot.json"
            latest = root / "reports" / "latest" / "map.svg"
            for path in (old, retained, latest):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            prune_old_artifacts(cfg)
            self.assertFalse(old.exists())
            self.assertTrue(retained.exists())
            self.assertTrue(latest.exists())


class CollectionIntegrationTests(unittest.TestCase):
    def test_fetch_records_failure_empty_and_success_without_extra_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            with patch("nys_aq.daily._get_with_backoff", side_effect=[
                RuntimeError("message containing secret-test-key"), {"results": []},
                {"results": [{"value": 10}]},
            ]) as get, patch("time.sleep"):
                ok, errors, elapsed = fetch_latest_for_locations(config(Path(tmp)), [1, 2, 3],
                                                                 request_records=records)
            self.assertEqual(get.call_count, 3)
            self.assertEqual(len(ok), 2)
            self.assertEqual(len(errors), 1)
            self.assertEqual([r["outcome"] for r in records], ["failed", "empty", "measurements"])
            self.assertNotIn("secret-test-key", json.dumps(records))
            self.assertTrue(all(r["started_at_utc"] <= r["ended_at_utc"] for r in records))

    def test_full_run_isolated_and_snapshot_failure_stops_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            cfg.ny_boundary_geojson.parent.mkdir(parents=True)
            cfg.ny_boundary_geojson.write_text(json.dumps({"geometry": {
                "type": "MultiPolygon", "coordinates": [[[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]],
            }}))

            def fake_fetch(cfg, ids, *, request_records):
                request_records.append({"location_id": 1, "outcome": "measurements"})
                return [{"results": [{"locationsId": 1, "sensorsId": 7,
                                       "datetime": {"utc": "2026-10-31T10:00:00Z"},
                                       "value": 8.5}]}], [], 0.01

            def fake_enrich(cfg, *, rows, location_ids):
                for item in rows:
                    item.update(parameter_name="pm25", units="µg/m³")
                return 0, 0.0

            with patch("nys_aq.daily.load_config", return_value=cfg) as load, \
                 patch("nys_aq.daily.fetch_locations", return_value=(1, [location()])), \
                 patch("nys_aq.daily.fetch_latest_for_locations", side_effect=fake_fetch), \
                 patch("nys_aq.daily.enrich_rows_with_sensor_meta", side_effect=fake_enrich), \
                 redirect_stdout(io.StringIO()):
                run_daily(repo_root=root)
                self.assertEqual(load.call_args.kwargs["repo_root"], root)
                directory = root / "reports" / "2026-10-30"
                result = read_snapshot(directory / "snapshot.json")
                self.assertEqual(result["measurements"][0]["value"], 8.5)
                self.assertTrue((directory / "map.svg").exists())
                self.assertTrue((root / "notes" / "2026-10-30.md").exists())
                self.assertTrue((root / "data" / "daily.csv").exists())
                with patch("nys_aq.snapshot.write_snapshot", side_effect=OSError("disk failure")):
                    with self.assertRaises(OSError):
                        run_daily(repo_root=root)


if __name__ == "__main__":
    unittest.main()
