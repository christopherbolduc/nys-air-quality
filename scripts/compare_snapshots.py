#!/usr/bin/env python3
"""Inspect one saved snapshot or compare two; no network, credentials, or writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nys_aq.compare import compare_snapshots, inspect_snapshot
from nys_aq.snapshot import read_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Snapshot JSON to inspect, or comparison baseline")
    parser.add_argument("current", nargs="?", type=Path, help="Later snapshot JSON for comparison")
    parser.add_argument("--parameter", default="pm25")
    parser.add_argument("--units", default="µg/m³", help="Exact units for comparison; no conversion")
    parser.add_argument("--max-gap-hours", type=float, default=36)
    parser.add_argument("--min-locations", type=int, default=1)
    args = parser.parse_args()
    try:
        baseline = read_snapshot(args.baseline)
        if args.current is None:
            result = inspect_snapshot(baseline, parameter=args.parameter)
        else:
            result = compare_snapshots(
                baseline, read_snapshot(args.current), parameter=args.parameter, units=args.units,
                max_observation_gap_hours=args.max_gap_hours, min_matched_locations=args.min_locations,
            )
        result["input_paths"] = {"baseline": str(args.baseline.resolve())}
        if args.current is not None:
            result["input_paths"]["current"] = str(args.current.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
