#!/usr/bin/env python3
"""
CLI wrapper for the daily run.

Example:
  python scripts/run_daily.py
  python scripts/run_daily.py --date 2026-01-28
"""

from __future__ import annotations

import argparse
from datetime import date

from nys_aq.daily import run_daily


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=_parse_date, default=None, help="Run date (YYYY-MM-DD). Default: yesterday (ET).")
    args = parser.parse_args()

    run_daily(run_date=args.date)


if __name__ == "__main__":
    main()