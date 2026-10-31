"""Read-only, deterministic tools for comparing concentration snapshots.

Results describe sampled sensors, never statewide exposure, AQI, or event causes.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from statistics import median

from nys_aq.snapshot import parse_observation_time, utc_text


def _context(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("Expected a version 1 snapshot")
    collection = snapshot.get("collection")
    if not isinstance(collection, dict):
        raise ValueError("Missing collection metadata")
    start = parse_observation_time(collection.get("started_at_utc"))
    end = parse_observation_time(collection.get("ended_at_utc"))
    reference = parse_observation_time(collection.get("freshness_reference_utc"))
    hours = collection.get("stale_hours")
    if (start is None or end is None or start > end or reference != end
            or not isinstance(hours, (int, float)) or isinstance(hours, bool)
            or not math.isfinite(hours) or hours < 0):
        raise ValueError("Invalid collection timing or freshness configuration")
    if collection.get("source") != "OpenAQ /latest" or not isinstance(collection.get("report_date"), str):
        raise ValueError("Unrecognized collection source or missing report label")
    label = date.fromisoformat(collection["report_date"])
    if not isinstance(snapshot.get("measurements"), list) or not isinstance(snapshot.get("locations"), list):
        raise ValueError("Snapshot measurements and locations must be lists")
    return {"start": start, "end": end, "hours": hours, "label": label, "evidence": {
        "report_label": collection["report_date"], "collection_started_at_utc": utc_text(start),
        "collection_ended_at_utc": utc_text(end), "stale_hours": hours,
    }}


def _positive_id(value: object) -> bool:
    return type(value) is int and value > 0


def _index(snapshot: dict, parameter: str) -> tuple[dict, list[dict], dict]:
    """Select newest per sensor before checking value/units; keep exclusions."""
    context = _context(snapshot)
    locations = {}
    for loc in snapshot["locations"]:
        if not isinstance(loc, dict) or not _positive_id(loc.get("location_id")):
            raise ValueError("Malformed snapshot location")
        if loc["location_id"] in locations:
            raise ValueError("Duplicate location metadata")
        locations[loc["location_id"]] = loc
    groups: dict[tuple, list[tuple[int, dict, object]]] = {}
    exclusions = []

    def exclude(index: int, reason: str) -> None:
        exclusions.append({"pointer": f"/measurements/{index}", "reason": reason})

    for i, row in enumerate(snapshot["measurements"]):
        if not isinstance(row, dict):
            raise ValueError("Malformed measurement row")
        if row.get("parameter") != parameter:
            continue
        if not _positive_id(row.get("location_id")) or not _positive_id(row.get("sensor_id")):
            exclude(i, "invalid_identity")
            continue
        observed = parse_observation_time(row.get("observed_at_utc"))
        if observed is None:
            exclude(i, "invalid_observation_time")
            continue
        key = (row["location_id"], row["sensor_id"])
        groups.setdefault(key, []).append((i, row, observed))

    eligible = {}
    for key, group in sorted(groups.items()):
        newest = max(t for _, _, t in group)
        latest = [(i, row, t) for i, row, t in group if t == newest]
        for i, _, t in group:
            if t != newest:
                exclude(i, "superseded_observation")
        if len(latest) != 1:
            for i, _, _ in latest:
                exclude(i, "duplicate_latest_observation")
            continue
        i, row, observed = latest[0]
        value = row.get("value")
        units = row.get("units")
        loc = locations.get(key[0], {})
        outcome = loc.get("request") or {}
        if loc.get("sampled") is not True or not isinstance(outcome, dict) or outcome.get("outcome") != "measurements":
            reason = "location_not_successfully_sampled"
        elif not isinstance(units, str) or units.strip().lower() in ("", "unknown"):
            reason = "unknown_units"
        elif (not isinstance(value, (int, float)) or isinstance(value, bool)
              or not math.isfinite(value) or value < 0):
            reason = "invalid_value"
        elif observed > context["end"]:
            reason = "future_observation"
        elif (context["end"] - observed).total_seconds() > context["hours"] * 3600:
            reason = "stale_observation"
        elif row.get("quality_flags") != [] or row.get("usable") is not True or row.get("stale") is not False:
            reason = "source_quality_exclusion"
        else:
            eligible[(*key, units)] = {"row": row, "time": observed, "pointer": f"/measurements/{i}", "location": loc}
            continue
        exclude(i, reason)
    return eligible, sorted(exclusions, key=lambda e: int(e["pointer"].rsplit("/", 1)[1])), context


def inspect_snapshot(snapshot: dict, *, parameter: str = "pm25") -> dict:
    """Summarize available evidence separately for each original unit."""
    eligible, exclusions, context = _index(snapshot, parameter)
    by_unit = {}
    for units in sorted({key[2] for key in eligible}):
        members = [entry for key, entry in eligible.items() if key[2] == units]
        values = [entry["row"]["value"] for entry in members]
        by_unit[units] = {
            "sensors": len(members), "locations": len({e["row"]["location_id"] for e in members}),
            "min": min(values), "median": median(values), "max": max(values),
            "observation_start_utc": utc_text(min(e["time"] for e in members)),
            "observation_end_utc": utc_text(max(e["time"] for e in members)),
            "pointers": [e["pointer"] for e in members],
        }
    return {
        "status": "available" if eligible else "insufficient_data",
        "snapshot": context["evidence"], "parameter": parameter,
        "by_unit": by_unit, "exclusions": exclusions,
        "exclusion_counts": dict(sorted(Counter(e["reason"] for e in exclusions).items())),
        "scope": "Latest eligible sampled-sensor readings; not a daily or statewide average.",
    }


def compare_snapshots(
    baseline: dict, current: dict, *, parameter: str = "pm25", units: str = "µg/m³",
    max_observation_gap_hours: float = 36, min_matched_locations: int = 1,
) -> dict:
    """Compare exactly two specified collections, without assigning event severity.

    The default 36-hour observation gap is an engineering guard for approximately
    daily collections, not a health threshold. Override it explicitly for other
    intervals. One location suffices for arithmetic, not a regional conclusion.
    """
    if (isinstance(max_observation_gap_hours, bool)
            or not isinstance(max_observation_gap_hours, (int, float))
            or not math.isfinite(max_observation_gap_hours) or max_observation_gap_hours <= 0
            or type(min_matched_locations) is not int or min_matched_locations < 1):
        raise ValueError("Invalid comparison gap or minimum matched locations")
    if not isinstance(units, str) or units.strip().lower() in ("", "unknown"):
        raise ValueError("Comparison units must be explicit")
    before, before_exclusions, bc = _index(baseline, parameter)
    after, after_exclusions, cc = _index(current, parameter)
    if cc["start"] < bc["end"] or cc["end"] <= bc["end"]:
        raise ValueError("Current collection must follow baseline without overlap")
    if bc["hours"] != cc["hours"]:
        raise ValueError("Snapshots must use the same freshness threshold")
    before_keys = {key for key in before if key[2] == units}
    after_keys = {key for key in after if key[2] == units}
    pairs, pair_exclusions = [], []
    for key in sorted(before_keys & after_keys):
        b, c = before[key], after[key]
        gap = (c["time"] - b["time"]).total_seconds() / 3600
        reason = None
        if gap == 0:
            reason = "repeated_observation" if c["row"]["value"] == b["row"]["value"] else "conflicting_repeated_observation"
        elif gap < 0:
            reason = "observation_time_regressed"
        elif gap > max_observation_gap_hours:
            reason = "observation_gap_exceeded"
        elif b["location"].get("coordinates") != c["location"].get("coordinates"):
            reason = "location_coordinates_changed"
        if reason:
            pair_exclusions.append({"location_id": key[0], "sensor_id": key[1], "reason": reason,
                                    "baseline_pointer": b["pointer"], "current_pointer": c["pointer"]})
            continue
        bval, cval = b["row"]["value"], c["row"]["value"]
        delta = cval - bval
        percent = delta / bval * 100 if bval != 0 else None
        percent_reason = "zero_baseline" if bval == 0 else None
        if percent is not None and not math.isfinite(percent):
            percent, percent_reason = None, "numeric_overflow"
        pairs.append({
            "location_id": key[0], "sensor_id": key[1], "units": units,
            "location_name": c["location"].get("name"), "coordinates": c["location"].get("coordinates"),
            "baseline_value": bval, "current_value": cval, "delta": delta,
            "percent_change": percent, "percent_change_unavailable_reason": percent_reason,
            "baseline_observed_at_utc": utc_text(b["time"]), "current_observed_at_utc": utc_text(c["time"]),
            "observation_gap_hours": gap,
            "baseline_pointer": b["pointer"], "current_pointer": c["pointer"],
        })
    matched_locations = len({p["location_id"] for p in pairs})
    summary = None
    if matched_locations >= min_matched_locations:
        summary = {
            "matched_sensors": len(pairs), "matched_locations": matched_locations,
            "baseline_median": median(p["baseline_value"] for p in pairs),
            "current_median": median(p["current_value"] for p in pairs),
            "median_sensor_delta": median(p["delta"] for p in pairs),
            "increased_sensors": sum(p["delta"] > 0 for p in pairs),
            "decreased_sensors": sum(p["delta"] < 0 for p in pairs),
            "unchanged_sensors": sum(p["delta"] == 0 for p in pairs),
        }
    return {
        "status": "compared" if summary is not None else "insufficient_data",
        "baseline": bc["evidence"], "current": cc["evidence"], "parameter": parameter, "units": units,
        "policy": {"max_observation_gap_hours": max_observation_gap_hours,
                   "min_matched_locations": min_matched_locations, "event_thresholds": None},
        "collection_gap_hours": (cc["end"] - bc["end"]).total_seconds() / 3600,
        "intervening_report_labels_not_examined": max(0, (cc["label"] - bc["label"]).days - 1),
        "intervening_history_checked": False,
        "coverage": {
            "baseline_eligible_sensors": len(before_keys), "current_eligible_sensors": len(after_keys),
            "matched_sensors": len(pairs), "matched_locations": matched_locations,
            "baseline_only": [{"location_id": k[0], "sensor_id": k[1]} for k in sorted(before_keys - after_keys)],
            "current_only": [{"location_id": k[0], "sensor_id": k[1]} for k in sorted(after_keys - before_keys)],
            "baseline_other_units": dict(sorted(Counter(k[2] for k in before if k[2] != units).items())),
            "current_other_units": dict(sorted(Counter(k[2] for k in after if k[2] != units).items())),
        },
        "summary": summary, "pairs": pairs,
        "exclusions": {"baseline": before_exclusions, "current": after_exclusions, "pairs": pair_exclusions},
        "scope": "Matched sampled sensors only. No daily, statewide, health-category, or causal conclusion is assigned.",
    }
