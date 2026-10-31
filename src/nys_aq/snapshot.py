"""Versioned evidence for later investigation; independent of report calculations."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Collection times must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_observation_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def _safe(value: object) -> object:
    """Preserve API evidence while producing strict, portable JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return value


def _is_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def build_snapshot(
    *,
    report_date: str,
    started_at: datetime,
    ended_at: datetime,
    stale_hours: int,
    ny_locations: list[dict],
    sampled_locations: list[dict],
    requests: list[dict],
    rows: list[dict],
) -> dict:
    """Build a snapshot without mutating the rows used by existing reports.

    Duplicate identity is location/sensor/parameter/units/observation time. All
    copies are excluded from initial comparisons, including conflicting copies.
    A future observation has no tolerance: any time after collection end is flagged.
    """
    if ended_at < started_at or stale_hours < 0:
        raise ValueError("Invalid collection interval or freshness threshold")
    start, end = utc_text(started_at), utc_text(ended_at)
    sampled = {loc["id"] for loc in sampled_locations}
    outcomes = {r["location_id"]: r for r in requests}
    locations = []
    for loc in ny_locations:
        loc_id = loc["id"]
        selected = loc_id in sampled
        if selected and loc_id not in outcomes:
            raise ValueError(f"Missing request outcome for selected location {loc_id}")
        locations.append({
            "location_id": loc_id,
            "name": loc.get("name"),
            "coordinates": _safe(loc.get("coordinates") or {}),
            "sampled": selected,
            "request": outcomes.get(loc_id) if selected else {"outcome": "unsampled"},
        })

    measurements = []
    cutoff = ended_at - timedelta(hours=stale_hours)
    for row in rows:
        flags = []
        value = row.get("value")
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        finite = numeric and math.isfinite(value)
        if value is None:
            flags.append("missing_value")
        elif not numeric:
            flags.append("invalid_value")
        elif not finite:
            flags.append("nonfinite_value")
        elif value < 0:
            flags.append("negative_value")

        raw_time = row.get("datetime_utc")
        observed = parse_observation_time(raw_time)
        if raw_time is None or raw_time == "":
            flags.append("missing_timestamp")
        elif observed is None:
            flags.append("invalid_timestamp")
        elif observed > ended_at:
            flags.append("future_timestamp")

        parameter, units = row.get("parameter_name"), row.get("units")
        if not isinstance(parameter, str) or parameter.strip().lower() in ("", "unknown"):
            flags.append("unknown_parameter")
        if not isinstance(units, str) or units.strip().lower() in ("", "unknown"):
            flags.append("unknown_units")
        loc_id, sensor_id = row.get("locationsId"), row.get("sensorsId")
        if not _is_id(loc_id) or loc_id not in sampled:
            flags.append("invalid_location_id")
        if not _is_id(sensor_id):
            flags.append("invalid_sensor_id")
        measurements.append({
            "location_id": _safe(loc_id), "sensor_id": _safe(sensor_id),
            "observed_at_utc": utc_text(observed) if observed else None,
            "parameter": _safe(parameter), "units": _safe(units),
            "value": value if finite else None,
            "original_value": _safe(value), "original_timestamp": _safe(raw_time),
            "quality_flags": flags,
            "stale": observed < cutoff if observed else None,
        })

    groups: dict[tuple, list[dict]] = {}
    for m in measurements:
        # Malformed identities cannot establish that two rows are duplicates.
        if any(f in m["quality_flags"] for f in (
            "invalid_location_id", "invalid_sensor_id", "missing_timestamp",
            "invalid_timestamp", "unknown_parameter", "unknown_units",
        )):
            continue
        key = (m["location_id"], m["sensor_id"], m["parameter"], m["units"], m["observed_at_utc"])
        groups.setdefault(key, []).append(m)
    for group in groups.values():
        if len(group) > 1:
            conflict = len({json.dumps(m["original_value"], sort_keys=True) for m in group}) > 1
            for m in group:
                m["quality_flags"].append("duplicate")
                if conflict:
                    m["quality_flags"].append("conflicting_duplicate")

    for m in measurements:
        m["usable"] = not m["quality_flags"] and m["stale"] is False
    flag_counts = Counter(f for m in measurements for f in m["quality_flags"])
    invalid = sum(any(f not in ("duplicate", "conflicting_duplicate") for f in m["quality_flags"])
                  for m in measurements)
    return {
        "schema_version": 1,
        "collection": {
            "report_date": report_date, "report_timezone": "America/New_York",
            "source": "OpenAQ /latest", "started_at_utc": start, "ended_at_utc": end,
            "freshness_reference_utc": end, "stale_hours": stale_hours,
        },
        "locations": locations,
        "measurements": measurements,
        "quality_summary": {
            "returned": len(measurements), "usable": sum(m["usable"] for m in measurements),
            "stale": sum(m["stale"] is True for m in measurements),
            "invalid": invalid, "duplicate_rows": flag_counts["duplicate"],
            "fetch_failures": sum(r["outcome"] == "failed" for r in requests),
            "metadata_gaps": sum(any(f in m["quality_flags"] for f in ("unknown_parameter", "unknown_units"))
                                 for m in measurements),
            "flags": dict(sorted(flag_counts.items())),
        },
    }


def write_snapshot(path: Path, snapshot: dict) -> Path:
    """Replace only after strict serialization; leave prior evidence on failure."""
    if type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("Unsupported snapshot schema version")
    content = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    json.loads(content)  # Validate the complete serialized document before replacement.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".snapshot-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def read_snapshot(path: Path, *, expected_collection_start: str | None = None) -> dict:
    """Reject unsupported formats and, when specified, evidence from another run."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 1):
        raise ValueError("Unsupported snapshot schema version")
    if expected_collection_start is not None:
        if snapshot["collection"]["started_at_utc"] != expected_collection_start:
            raise ValueError("Snapshot does not belong to the expected collection")
    return snapshot
