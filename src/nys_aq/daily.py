
"""
Daily run logic for NYS air quality snapshots.

Designed to be called from both:
- scripts/run_daily.py (CLI)
- notebooks (import and call run_daily)
"""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DailyConfig:
    bbox: str
    sample_size: int
    stale_hours: int
    run_date: date
    repo_root: Path
    ny_boundary_geojson: Path
    retention_days: int
    _api_key: str

    @property
    def api_key(self) -> str:
        return self._api_key

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "DailyConfig("
            f"bbox={self.bbox!r}, "
            f"sample_size={self.sample_size!r}, "
            f"stale_hours={self.stale_hours!r}, "
            f"run_date={self.run_date!r}, "
            f"repo_root={str(self.repo_root)!r}, "
            f"ny_boundary_geojson={str(self.ny_boundary_geojson)!r}, "
            f"retention_days={self.retention_days!r}, "
            "_api_key='***'"
            ")"
        )


def _load_dotenv(env_path: Path) -> None:
    """
    Load key/value pairs from a .env file into os.environ, without overwriting existing vars.
    """
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _default_run_date(tz: ZoneInfo) -> date:
    """
    Default to 'yesterday' in the given timezone.
    """
    return datetime.now(tz).date() - timedelta(days=1)


def load_config(
    *,
    repo_root: Path | None = None,
    run_date: date | None = None,
    tz_name: str = "America/New_York",
) -> DailyConfig:
    """
    Build configuration from repo files + environment variables.

    Env vars (via .env or process env):
    - OPENAQ_API_KEY (required)
    - BBOX (optional; performance hint)
    - SAMPLE_SIZE (optional)
    - STALE_HOURS (optional)
    - RETENTION_DAYS (optional; delete notes/reports older than this many days; default 91)
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()

    _load_dotenv(root / ".env")

    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY is not set. Put it in .env or set it in the environment.")

    bbox = os.getenv("BBOX", "-79.8,40.45,-71.85,45.1")
    sample_size = int(os.getenv("SAMPLE_SIZE", "100"))
    stale_hours = int(os.getenv("STALE_HOURS", "12"))
    retention_days = int(os.getenv("RETENTION_DAYS", "91"))

    tz = ZoneInfo(tz_name)
    effective_date = run_date or _default_run_date(tz)

    ny_geojson = root / "data" / "nys_boundary.geojson"
    if not ny_geojson.exists():
        raise FileNotFoundError(f"Missing NY boundary file: {ny_geojson}")

    return DailyConfig(
        bbox=bbox,
        sample_size=sample_size,
        stale_hours=stale_hours,
        run_date=effective_date,
        repo_root=root,
        ny_boundary_geojson=ny_geojson,
        retention_days=retention_days,
        _api_key=api_key,
    )


def _parse_ymd(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def prune_old_artifacts(cfg: DailyConfig) -> dict[str, int]:
    """
    Delete note/report artifacts older than cfg.retention_days.
    Only affects:
      - notes/YYYY-MM-DD.md
      - reports/YYYY-MM-DD/...
    Never touches:
      - data/daily.csv
      - reports/latest/
    """
    notes_dir = cfg.repo_root / "notes"
    reports_root = cfg.repo_root / "reports"
    cutoff = cfg.run_date - timedelta(days=cfg.retention_days)

    deleted_notes = 0
    deleted_report_dirs = 0

    if notes_dir.exists():
        for p in notes_dir.glob("*.md"):
            d = _parse_ymd(p.stem)
            if d is not None and d < cutoff:
                p.unlink(missing_ok=True)
                deleted_notes += 1

    if reports_root.exists():
        for p in reports_root.iterdir():
            if not p.is_dir():
                continue
            if p.name == "latest":
                continue
            d = _parse_ymd(p.name)
            if d is not None and d < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                deleted_report_dirs += 1

    return {"deleted_notes": deleted_notes, "deleted_report_dirs": deleted_report_dirs}


def fetch_locations(cfg: DailyConfig) -> tuple[int, list[dict]]:
    """
    Fetch a location catalog from OpenAQ for the configured bbox (US only).

    Returns:
        (latency_ms, locations)
    """
    import time

    import requests

    base = "https://api.openaq.org/v3"

    t0 = time.time()
    resp = requests.get(
        f"{base}/locations",
        params={"bbox": cfg.bbox, "iso": "US", "limit": 1000},
        headers={"X-API-Key": cfg.api_key},
        timeout=30,
    )
    latency_ms = int((time.time() - t0) * 1000)

    resp.raise_for_status()
    payload = resp.json()
    return latency_ms, payload.get("results", []) or []


def load_ny_multipolygon(cfg: DailyConfig) -> list:
    """
    Load NY boundary GeoJSON and return MultiPolygon coordinates.
    """
    import json

    feature = json.loads(cfg.ny_boundary_geojson.read_text(encoding="utf-8"))
    geom = feature.get("geometry") or {}
    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    if geom_type != "MultiPolygon" or not isinstance(coords, list):
        raise ValueError(f"Expected MultiPolygon GeoJSON at {cfg.ny_boundary_geojson}")

    return coords


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon:
        return False

    if not _point_in_ring(lon, lat, polygon[0]):
        return False

    for hole in polygon[1:]:
        if _point_in_ring(lon, lat, hole):
            return False

    return True


def point_in_multipolygon(lon: float, lat: float, multipolygon: list) -> bool:
    for polygon in multipolygon:
        if _point_in_polygon(lon, lat, polygon):
            return True
    return False


def filter_locations_in_ny(multipolygon: list, locations: list[dict]) -> list[dict]:
    """
    Filter OpenAQ locations to those whose coordinates fall inside the NY boundary.
    """
    out: list[dict] = []
    for loc in locations:
        c = loc.get("coordinates") or {}
        try:
            lon = float(c.get("longitude"))
            lat = float(c.get("latitude"))
        except (TypeError, ValueError):
            continue

        if point_in_multipolygon(lon, lat, multipolygon):
            out.append(loc)

    return out


def stable_sample_locations(locations: list[dict], sample_size: int, salt: str) -> list[dict]:
    """
    Deterministically sample locations by sorting on a stable hash of (location id + salt).
    """
    import hashlib

    def rank(loc: dict) -> int:
        loc_id = str(loc.get("id", ""))
        raw = (loc_id + "|" + salt).encode("utf-8")
        return int(hashlib.sha256(raw).hexdigest(), 16)

    ordered = sorted(locations, key=rank)
    return ordered[: min(sample_size, len(ordered))]


def _get_with_backoff(session, url: str, *, max_attempts: int = 8, timeout_s: int = 30) -> dict:
    """
    GET JSON with basic handling for OpenAQ rate limiting (429).
    """
    import time

    delay_s = 1.0
    for _ in range(max_attempts):
        resp = session.get(url, timeout=timeout_s)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay_s = max(delay_s, float(retry_after))
                except ValueError:
                    pass
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.5, 30.0)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Rate-limited too long: {url}")


def fetch_latest_for_locations(
    cfg: DailyConfig, location_ids: list[int], *, request_records: list[dict] | None = None,
) -> tuple[list[dict], list[tuple[int, str]], float]:
    """
    Fetch latest measurements for each location id.
    Returns (payloads_ok, errors, elapsed_seconds).
    """
    import time

    import requests

    base = "https://api.openaq.org/v3"
    session = requests.Session()
    session.headers.update({"X-API-Key": cfg.api_key})

    ok: list[dict] = []
    errors: list[tuple[int, str]] = []

    t0 = time.time()
    for loc_id in location_ids:
        url = f"{base}/locations/{loc_id}/latest"
        record = {"location_id": loc_id, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            payload = _get_with_backoff(session, url)
            ok.append(payload)
            record["outcome"] = "measurements" if payload.get("results") else "empty"
        except Exception as e:
            errors.append((loc_id, str(e)))
            # Store only the exception class, not URLs or credential-bearing messages.
            record.update(outcome="failed", error_type=type(e).__name__)
        record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        if request_records is not None:
            request_records.append(record)

        # Keep under common minute-level limits; reliable > fast.
        time.sleep(1.05)

    return ok, errors, (time.time() - t0)


def normalize_latest(cfg: DailyConfig, latest_payloads: list[dict]) -> list[dict]:
    """
    Flatten OpenAQ latest payloads into one row per measurement.
    """
    now_utc = datetime.now(timezone.utc)
    stale_cutoff = now_utc - timedelta(hours=cfg.stale_hours)

    rows: list[dict] = []
    for payload in latest_payloads:
        for m in payload.get("results", []) or []:
            dt_utc = (m.get("datetime") or {}).get("utc")
            coords = m.get("coordinates") or {}

            dt_obj: datetime | None = None
            if isinstance(dt_utc, str):
                try:
                    dt_obj = datetime.fromisoformat(dt_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
                except ValueError:
                    dt_obj = None

            rows.append(
                {
                    "locationsId": m.get("locationsId"),
                    "sensorsId": m.get("sensorsId"),
                    "datetime_utc": dt_utc,
                    "value": m.get("value"),
                    "latitude": coords.get("latitude"),
                    "longitude": coords.get("longitude"),
                    "stale": (dt_obj is None) or (dt_obj < stale_cutoff),
                }
            )

    return rows


def load_sensor_cache(cache_path: Path) -> dict[int, dict]:
    """
    Load cached sensor metadata: sensor_id -> {parameter_name, units}.
    """
    import json

    if not cache_path.exists():
        return {}

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    raw = payload.get("sensor_meta") or {}
    if not isinstance(raw, dict):
        return {}

    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = dict(v)
        except Exception:
            continue
    return out


def save_sensor_cache(cache_path: Path, sensor_meta: dict[int, dict]) -> None:
    """
    Persist sensor metadata cache in a stable JSON format.
    """
    import json

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sensor_meta": {str(k): v for k, v in sorted(sensor_meta.items(), key=lambda kv: kv[0])},
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def enrich_rows_with_sensor_meta(
    cfg: DailyConfig,
    *,
    rows: list[dict],
    location_ids: list[int],
    throttle_s: float = 0.25,
) -> tuple[int, float]:
    """
    Attach parameter_name and units onto rows using a cached sensor map.
    Only calls OpenAQ for sensors that aren't already cached.

    Returns:
        (new_cache_entries_added, elapsed_seconds)
    """
    import time

    import requests

    cache_path = cfg.repo_root / "data" / "sensor_meta_cache.json"
    sensor_meta = load_sensor_cache(cache_path)

    needed = {r.get("sensorsId") for r in rows if isinstance(r.get("sensorsId"), int)}
    needed_int = {int(x) for x in needed if isinstance(x, int)}
    missing = needed_int.difference(sensor_meta.keys())

    if not missing:
        for r in rows:
            sid = r.get("sensorsId")
            meta = sensor_meta.get(sid) if isinstance(sid, int) else None
            if meta:
                r.update(meta)
        return 0, 0.0

    session = requests.Session()
    session.headers.update({"X-API-Key": cfg.api_key})
    base = "https://api.openaq.org/v3"

    added = 0
    t0 = time.time()

    for loc_id in location_ids:
        if not missing:
            break

        url = f"{base}/locations/{loc_id}/sensors"
        try:
            payload = _get_with_backoff(session, url)
        except Exception:
            time.sleep(throttle_s)
            continue

        for s in payload.get("results", []) or []:
            sid = s.get("id")
            if not isinstance(sid, int) or sid not in missing:
                continue

            param = s.get("parameter") or {}
            sensor_meta[sid] = {
                "parameter_name": param.get("name"),
                "units": param.get("units"),
            }
            missing.remove(sid)
            added += 1

            if not missing:
                break

        time.sleep(throttle_s)

    elapsed = time.time() - t0
    if added:
        save_sensor_cache(cache_path, sensor_meta)

    for r in rows:
        sid = r.get("sensorsId")
        meta = sensor_meta.get(sid) if isinstance(sid, int) else None
        if meta:
            r.update(meta)

    return added, elapsed


def pretty_parameter_name(param: str) -> str:
    p = (param or "").strip().lower()
    mapping = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "pm1": "PM1",
        "pm4": "PM4",
        "o3": "O₃",
        "no2": "NO₂",
        "so2": "SO₂",
        "co": "CO",
        "co2": "CO₂",
        "nox": "NOₓ",
        "no": "NO",
        "bc": "Black carbon",
        "ufp": "Ultrafine particles",
    }
    return mapping.get(p, param)


def _svg_parameter_coverage(
    report_dir: Path,
    *,
    report_date: str,
    top_params: list[tuple[str, int]],
    title_suffix: str = "Pollutant coverage",
) -> Path:
    """
    Horizontal bar chart: measurement counts by parameter for the daily sample.
    Includes an in-chart dynamic dictionary (bottom-right).
    """
    from datetime import date as _date
    from xml.sax.saxutils import escape

    chart_path = report_dir / "parameter_coverage.svg"

    # Format date: "D Month, YYYY" (no leading zero)
    try:
        d = _date.fromisoformat(report_date)
        pretty_date = f"{d.day} {d.strftime('%B')}, {d.year}"
    except ValueError:
        pretty_date = report_date

    items = [(str(p), int(c)) for p, c in top_params]
    items = items[:10]

    total = sum(c for _, c in items) or 1
    max_v = max((c for _, c in items), default=1)

    width, height = 1040, 460
    pad_left, pad_right, pad_top, pad_bottom = 180, 120, 94, 58
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    bar_h = 22
    gap = 12
    needed_h = len(items) * (bar_h + gap) - gap
    if needed_h > plot_h:
        height = pad_top + needed_h + pad_bottom
        plot_h = needed_h

    y0 = pad_top
    y1 = pad_top + plot_h

    # Ticks/grid
    tick_count = 5
    tick_step = max(1, int(max_v / (tick_count - 1)))
    tick_max = tick_step * (tick_count - 1)

    bg = "#F7F5EF"
    panel = "#FFFFFF"
    text_dark = "#1F2933"
    text_muted = "#607080"
    grid = "#D7DDE3"
    bar_fill = "#0F4C5C"
    bar_fill_soft = "#2A6F97"
    accent = "#C96E4B"

    dictionary = {
        "pm25": "PM2.5: fine particles ≤2.5 µm",
        "pm10": "PM10: particles ≤10 µm",
        "pm1": "PM1: particles ≤1 µm",
        "pm4": "PM4: particles ≤4 µm",
        "o3": "O3: ozone",
        "no2": "NO2: nitrogen dioxide",
        "so2": "SO2: sulfur dioxide",
        "co": "CO: carbon monoxide",
        "bc": "BC: black carbon",
        "co2": "CO2: carbon dioxide",
        "no": "NO: nitric oxide",
        "nox": "NOx: nitrogen oxides",
        "ch4": "CH4: methane",
        "ufp": "UFP: ultrafine particles",
    }

    present = [p.strip().lower() for p, _ in items]
    present_set = set(present)

    priority = [
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co",
        "bc",
        "pm1",
        "pm4",
        "co2",
        "no",
        "nox",
        "ch4",
        "ufp",
    ]

    ordered_keys = [k for k in priority if k in present_set]
    legend_lines = [dictionary[k] for k in ordered_keys if k in dictionary]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')
    parts.append(
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" rx="18" ry="18" '
        f'fill="{panel}" stroke="#E3E8EE" stroke-width="1"/>'
    )

    parts.append(
        f'<text x="36" y="48" font-family="Public Sans, Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="22" font-weight="600" fill="{text_dark}">'
        f'{escape(title_suffix)} — {escape(pretty_date)}'
        f"</text>"
    )
    parts.append(
        f'<text x="36" y="72" font-family="Public Sans, Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="12.5" font-weight="400" fill="{text_muted}">'
        f'Counts returned by OpenAQ / latest across the NY daily sample. Top {len(items)} parameters shown.'
        f"</text>"
    )
    parts.append(
        f'<text x="{width - 36}" y="47" font-family="Public Sans, Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="12" '
        f'text-anchor="end" fill="{text_dark}">Total measurements: {total}</text>'
    )

    # Axes
    parts.append(f'<line x1="{pad_left}" y1="{y0}" x2="{pad_left}" y2="{y1}" stroke="#6B7A86" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_left}" y1="{y1}" x2="{pad_left + plot_w}" y2="{y1}" stroke="#6B7A86" stroke-width="1"/>')

    # Grid + tick labels
    for i in range(tick_count):
        v = i * tick_step
        x = pad_left + (v / tick_max) * plot_w if tick_max else pad_left
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="{grid}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{y1 + 18}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="11" '
            f'text-anchor="middle" fill="{text_muted}">{v}</text>'
        )

    parts.append(
        f'<text x="{pad_left + plot_w/2:.2f}" y="{height - 18}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" '
        f'font-size="12" text-anchor="middle" fill="{text_muted}">measurement count</text>'
    )

    # Bars + labels
    max_label_x = width - pad_right
    for idx, (param, count) in enumerate(items):
        y = pad_top + idx * (bar_h + gap)
        bar_w = (count / max_v) * plot_w if max_v else 0.0

        parts.append(
            f'<text x="{pad_left - 12}" y="{y + bar_h - 5}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" '
            f'font-size="12" text-anchor="end" fill="{text_dark}">{escape(param)}</text>'
        )

        parts.append(
            f'<rect x="{pad_left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="6" ry="6" '
            f'fill="{bar_fill}" stroke="{bar_fill_soft}" stroke-width="0.8"/>'
        )

        pct = int(round((count / total) * 100))
        label = f"{count} ({pct}%)"

        outside_x = pad_left + bar_w + 8
        approx_text_w = 7 * len(label)
        if outside_x + approx_text_w <= max_label_x:
            parts.append(
                f'<text x="{outside_x:.2f}" y="{y + bar_h - 5}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="12" fill="{text_dark}">'
                f"{escape(label)}</text>"
            )
        else:
            inside_x = max(pad_left + 6, pad_left + bar_w - 6)
            parts.append(
                f'<text x="{inside_x:.2f}" y="{y + bar_h - 5}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="12" '
                f'fill="white" text-anchor="end">{escape(label)}</text>'
            )

    # In-chart dynamic dictionary (bottom-right)
    if legend_lines:
        line_h = 14
        title_h = 14
        pad_box = 12

        # Size the box conservatively
        box_w = 308
        box_h = pad_box * 2 + title_h + len(legend_lines) * line_h

        # Place the dictionary box inside the plot at bottom-right.
        box_x = pad_left + plot_w - box_w - 12
        box_y = y1 - box_h - 12

        # Background
        parts.append(
            f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_w}" height="{box_h}" rx="12" ry="12" '
            f'fill="#FBFCFD" stroke="#D8E0E7" stroke-width="1"/>'
        )

        tx = box_x + pad_box
        ty = box_y + pad_box + 11

        parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" font-size="11" fill="{text_dark}">'
            f"Dictionary"
            f"</text>"
        )
        for idx, line in enumerate(legend_lines, start=1):
            parts.append(
                f'<text x="{tx:.2f}" y="{ty + idx * line_h:.2f}" font-family="Inter, Aptos, Segoe UI, Arial, sans-serif" '
                f'font-size="11" fill="{text_dark}">{escape(line)}</text>'
            )

    parts.append("</svg>")
    chart_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return chart_path


def _svg_map(
    report_dir: Path,
    *,
    report_date: str,
    sample_label: str,
    ny_multipolygon: list,
    ny_locations: list[dict],
    rows: list[dict],
    primary_param: str,
    units: str,
    collection_started_at: datetime | None = None,
    collection_ended_at: datetime | None = None,
    stale_hours: int = 12,
) -> Path:
    """Draw concentrations on a snapshot-relative scale, never AQI categories."""
    import math
    from xml.sax.saxutils import escape

    from nys_aq.snapshot import parse_observation_time, utc_text

    # Select the newest timestamp per location before validating its value. An
    # invalid latest value must not silently fall back to an older clean value.
    candidates: dict[int, list[tuple[datetime, dict]]] = {}
    for row in rows:
        if row.get("parameter_name") != primary_param:
            continue
        loc_id = row.get("locationsId")
        observed = parse_observation_time(row.get("datetime_utc"))
        if not isinstance(loc_id, int) or isinstance(loc_id, bool) or observed is None:
            continue
        candidates.setdefault(loc_id, []).append((observed, row))

    points = []
    for loc in ny_locations:
        coords = loc.get("coordinates") or {}
        try:
            lon, lat = float(coords.get("longitude")), float(coords.get("latitude"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lon) or not math.isfinite(lat):
            continue
        options = candidates.get(loc.get("id"), [])
        value, observed, status = None, None, "unavailable"
        if options:
            observed = max(t for t, _ in options)
            latest = [r for t, r in options if t == observed]
            valid = []
            for row in latest:
                raw = row.get("value")
                if (isinstance(raw, (int, float)) and not isinstance(raw, bool)
                        and math.isfinite(raw) and raw >= 0
                        and units and units.strip().lower() not in ("", "unknown")
                        and row.get("units") == units
                        and (collection_ended_at is None or observed <= collection_ended_at)):
                    valid.append(row)
            # Conflicting simultaneous readings or incompatible units must not
            # be combined into a single concentration point.
            if len(valid) == len(latest) and len({r["value"] for r in valid}) == 1:
                value = float(valid[0]["value"])
                stale = (observed < collection_ended_at - timedelta(hours=stale_hours)
                         if collection_ended_at is not None
                         else any(r.get("stale", True) for r in valid))
                status = "stale" if stale else "fresh"
        points.append((lon, lat, value, observed, status, loc))

    values = sorted(value for _, _, value, _, status, _ in points if status == "fresh")

    def percentile(p: float) -> float:
        k = (len(values) - 1) * p
        low = int(k)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (k - low)

    vmin, vmax = (percentile(0.05), percentile(0.95)) if values else (None, None)

    def color(t: float) -> str:
        # A sequential blue ramp avoids implying EPA health categories.
        t = max(0.0, min(1.0, t))
        return "rgb({},{},{})".format(*(round(a + (b - a) * t)
                                     for a, b in zip((190, 219, 240), (8, 70, 125))))

    lons = [float(lon) for poly in ny_multipolygon for lon, _ in poly[0]]
    lats = [float(lat) for poly in ny_multipolygon for _, lat in poly[0]]
    min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
    lon_scale = math.cos(math.radians(sum(lats) / len(lats)))
    width, height = 960, 600
    map_top, map_bottom, pad = 126, 154, 28
    lon_span = (max_lon - min_lon) * lon_scale or 1.0
    lat_span = max_lat - min_lat or 1.0
    drawable_w, drawable_h = width - 2 * pad, height - map_top - map_bottom
    scale = min(drawable_w / lon_span, drawable_h / lat_span)
    x0 = pad + (drawable_w - lon_span * scale) / 2
    y0 = map_top + (drawable_h - lat_span * scale) / 2

    def project(lon: float, lat: float) -> tuple[float, float]:
        return x0 + (lon - min_lon) * lon_scale * scale, y0 + (max_lat - lat) * scale

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-labelledby="map-title map-description">',
             '<title id="map-title">New York State latest concentration snapshot</title>',
             '<desc id="map-description">Colors show relative concentrations within this snapshot, not AQI. Observation times vary. Stale readings are gray; unavailable readings are light gray.</desc>',
             f'<rect width="{width}" height="{height}" fill="#F7F5EF"/>',
             f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" rx="18" fill="#FFFFFF" stroke="#E4E8EE"/>']

    def text(x: float, y: float, content: str, size: float = 12, **attrs) -> None:
        extra = " ".join(f'{key.replace("_", "-")}="{val}"' for key, val in attrs.items())
        parts.append(f'<text x="{x}" y="{y}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="{size}" fill="#344454" {extra}>{escape(content)}</text>')

    fresh_n = sum(p[4] == "fresh" for p in points)
    stale_n = sum(p[4] == "stale" for p in points)
    unavailable_n = len(points) - fresh_n - stale_n
    text(32, 43, f"New York State — {pretty_parameter_name(primary_param)} concentration (latest)", 22, font_weight="600")
    text(32, 65, f"Report label: {report_date} • {sample_label}")
    timing = "Collection time unavailable"
    if collection_started_at is not None and collection_ended_at is not None:
        timing = f"Collected (UTC): {utc_text(collection_started_at)} to {utc_text(collection_ended_at)}"
    text(32, 85, timing, 11)
    text(32, 105, f"Fresh: {fresh_n} • stale: {stale_n} • unavailable: {unavailable_n} • observation times vary", 11)

    for poly in ny_multipolygon:
        path = []
        for i, (lon, lat) in enumerate(poly[0]):
            x, y = project(float(lon), float(lat))
            path.append(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}")
        parts.append(f'<path d="{" ".join(path)} Z" fill="#EEF2EA" stroke="#1F2933" stroke-width="1.25" stroke-linejoin="round"/>')

    for lon, lat, value, observed, status, loc in points:
        x, y = project(lon, lat)
        fill = "lightgray" if status == "unavailable" else "rgb(180,180,180)"
        if status == "fresh":
            t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            fill = color(t)
        detail = f"{loc.get('name') or 'Location'} (ID {loc.get('id')}): {status}"
        if value is not None:
            detail += f"; {value:g} {units}"
        if observed is not None:
            detail += f"; observed {utc_text(observed)}"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{fill}" stroke="#24313A" stroke-width="0.75"><title>{escape(detail)}</title></circle>')

    text(32, 471, f"Concentration ({units or 'units unavailable'}) — relative scale", 13, font_weight="600")
    if values:
        for i in range(32):
            t = i / 31
            fill = color(t if vmax > vmin else 0.5)
            parts.append(f'<rect x="{32 + i * 9}" y="483" width="9" height="12" fill="{fill}"/>')
        text(32, 514, f"{vmin:.2f}")
        text(320, 514, f"{vmax:.2f}", text_anchor="end")
        text(344, 495, "Fresh readings: 5th–95th percentiles; values outside this range use end colors.", 11)
    else:
        text(32, 495, "No fresh valid readings available; no concentration scale is shown.")
    text(32, 542, "Colors are relative to this snapshot; compare numeric values across dates.", 11)
    text(32, 563, "Not an AQI or a daily average. Gray: stale; light gray: unavailable or invalid.", 11)
    parts.append("</svg>")
    map_path = report_dir / "map.svg"
    map_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return map_path


def _upsert_daily_csv(csv_path: Path, *, report_date: str, row: dict) -> None:
    fields = list(row.keys())
    existing: list[dict] = []

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("report_date"):
                    existing.append(r)

    kept = [r for r in existing if r.get("report_date") != report_date]
    kept.append(row)
    kept.sort(key=lambda r: r["report_date"])

    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in fields})

    tmp.replace(csv_path)


def write_outputs(
    cfg: DailyConfig,
    *,
    locations_latency_ms: int,
    ny_locations: list[dict],
    sampled_locations: list[dict],
    latest_elapsed_s: float,
    rows: list[dict],
    top_params: list[tuple[str, int]],
    latest_errors: list[tuple[int, str]] | None = None,
    collection_started_at: datetime | None = None,
    collection_ended_at: datetime | None = None,
) -> dict[str, Path]:
    """
    Write daily artifacts (idempotent per report date).
    """
    report_date = cfg.run_date.isoformat()
    sample_label = (
        "All NY locations (with coordinates)"
        if len(sampled_locations) == len(ny_locations)
        else f"Stable sample ({len(sampled_locations)} of {len(ny_locations)} NY locations)"
    )
    data_dir = cfg.repo_root / "data"
    notes_dir = cfg.repo_root / "notes"
    reports_dir = cfg.repo_root / "reports" / report_date

    data_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    stale_n = sum(1 for r in rows if r.get("stale"))
    stale_fraction = (stale_n / len(rows)) if rows else 0.0

    # Primary param = most common
    primary_param = top_params[0][0] if top_params else "unknown"
    map_param = os.getenv("MAP_PARAM", primary_param).strip().lower() or primary_param
    unit = ""
    for r in rows:
        if (r.get("parameter_name") or "unknown") == map_param and r.get("units"):
            unit = str(r.get("units"))
            break


    # CSV (upsert by report_date)
    csv_path = data_dir / "daily.csv"
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    csv_row = {
        "report_date": report_date,
        "report_tz": "America/New_York",
        "generated_at_utc": now_utc,
        "bbox": cfg.bbox,
        "ny_locations": str(len(ny_locations)),
        "sampled_locations": str(len(sampled_locations)),
        "measurements_total": str(len(rows)),
        "stale_hours": str(cfg.stale_hours),
        "stale_fraction": f"{stale_fraction:.4f}",
        "locations_latency_ms": str(locations_latency_ms),
        "latest_elapsed_s": f"{latest_elapsed_s:.2f}",
        "latest_error_count": str(len(latest_errors or [])),
        "top_parameters": ";".join([f"{p}:{c}" for p, c in top_params]),
        "primary_parameter": primary_param,
        "primary_units": unit,
    }
    _upsert_daily_csv(csv_path, report_date=report_date, row=csv_row)

    # Note (overwrite per day)
    note_path = notes_dir / f"{report_date}.md"
    lines: list[str] = []
    lines.append(f"# NYS air quality daily summary — {report_date}")
    lines.append("")
    lines.append("## Technical summary")
    lines.append(f"- Report date label (America/New_York): {report_date}")
    if collection_started_at is not None and collection_ended_at is not None:
        from nys_aq.snapshot import utc_text

        lines.append(f"- Collection started (UTC): {utc_text(collection_started_at)}")
        lines.append(f"- Collection ended (UTC): {utc_text(collection_ended_at)}")
    lines.append(f"- Locations in NY boundary: {len(ny_locations)}")
    lines.append(f"- Locations used: {sample_label}")
    lines.append(f"- Measurements normalized: {len(rows)}")
    lines.append(f"- Locations catalog latency: {locations_latency_ms} ms")
    lines.append(f"- Latest fetch duration: {latest_elapsed_s:.2f} s")
    if latest_errors:
        lines.append(f"- Latest fetch errors: {len(latest_errors)}")
    lines.append("")
    lines.append("### Data quality checks")
    lines.append(f"- Stale threshold: {cfg.stale_hours} hours")
    lines.append(f"- Stale fraction: {stale_fraction:.3f}")
    if latest_errors:
        lines.append("")
        lines.append("### Fetch warnings")
        for loc_id, msg in latest_errors[:5]:
            lines.append(f"- Location {loc_id}: {msg}")
    lines.append("")
    lines.append("### Parameter coverage (top)")
    for p, c in top_params:
        u = ""
        for rr in rows:
            if (rr.get("parameter_name") or "unknown") == p and rr.get("units"):
                u = str(rr.get("units"))
                break
        u_s = f" ({u})" if u else ""
        lines.append(f"- {p}{u_s}: {c}")
    lines.append("")
    lines.append("## Plain-language summary")
    lines.append(
        "This report checks a stable set of monitoring locations within New York State and records the latest readings "
        "available from OpenAQ. Some readings may be old; the quality section above tracks how many are considered stale."
    )
    lines.append(
        "The report date is an output label, not a full day of observations. "
        "Observation times vary by sensor. The map shows concentrations, not AQI or daily averages; "
        "its colors are relative to this snapshot and cannot be compared directly across dates."
    )
    if primary_param == "pm25":
        lines.append("")
        lines.append(
            "PM2.5 is a measure of very small airborne particles (2.5 micrometers or smaller) that can be breathed deep into the lungs."
        )

    lines.append("")
    lines.append("## Reports")

    coverage_path = reports_dir / "parameter_coverage.svg"
    map_path = reports_dir / "map.svg"

    cov_link = os.path.relpath(coverage_path, start=note_path.parent).replace(os.sep, "/")
    map_link = os.path.relpath(map_path, start=note_path.parent).replace(os.sep, "/")

    lines.append(f"- [Parameter coverage]({cov_link})")
    lines.append(f"- [Map]({map_link})")

    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # SVGs
    chart_path = _svg_parameter_coverage(
        reports_dir,
        report_date=report_date,
        top_params=top_params,
        title_suffix="Pollutant coverage",
    )
    map_path = _svg_map(
        reports_dir,
        report_date=report_date,
        sample_label=sample_label,
        ny_multipolygon=load_ny_multipolygon(cfg),
        ny_locations=ny_locations,
        rows=rows,
        primary_param=map_param,
        units=unit,
        collection_started_at=collection_started_at,
        collection_ended_at=collection_ended_at,
        stale_hours=cfg.stale_hours,
    )

    return {
        "daily_csv": csv_path,
        "note": note_path,
        "chart": chart_path,
        "map": map_path,
    }


def run_daily(
    *, run_date: date | None = None, tz_name: str = "America/New_York", repo_root: Path | None = None,
) -> DailyConfig:
    cfg = load_config(run_date=run_date, tz_name=tz_name, repo_root=repo_root)

    locations_latency_ms, locations = fetch_locations(cfg)
    ny_poly = load_ny_multipolygon(cfg)
    ny_locations = filter_locations_in_ny(ny_poly, locations)

    sampled = stable_sample_locations(ny_locations, cfg.sample_size, salt=cfg.bbox)
    location_ids = [loc["id"] for loc in sampled if isinstance(loc.get("id"), int)]

    request_records: list[dict] = []
    collection_started = datetime.now(timezone.utc)
    latest_ok, latest_errors, latest_elapsed_s = fetch_latest_for_locations(
        cfg, location_ids, request_records=request_records,
    )
    collection_ended = datetime.now(timezone.utc)
    if latest_errors:
        print("latest_fetch_errors:", len(latest_errors))
        for loc_id, msg in latest_errors[:5]:
            print(f"latest_fetch_error_{loc_id}:", msg)
    if not latest_ok:
        raise RuntimeError("No latest payloads returned; cannot build report")

    rows = normalize_latest(cfg, latest_ok)
    added, sensor_elapsed_s = enrich_rows_with_sensor_meta(cfg, rows=rows, location_ids=location_ids)

    stale_n = sum(1 for r in rows if r.get("stale"))
    print("date:", cfg.run_date.isoformat())
    print("bbox:", cfg.bbox)
    print("sample_size_config:", cfg.sample_size)
    print("stale_hours:", cfg.stale_hours)
    print("locations_returned:", len(locations))
    print("locations_latency_ms:", locations_latency_ms)
    print("ny_locations:", len(ny_locations))
    print("sample_size_actual:", len(sampled))
    print("latest_ok:", len(latest_ok))
    print("latest_elapsed_s:", round(latest_elapsed_s, 2))
    print("measurements_total:", len(rows))
    print("stale_fraction:", round(stale_n / len(rows), 3) if rows else 0.0)
    print("sensor_cache_new_entries:", added)
    print("sensor_lookup_elapsed_s:", round(sensor_elapsed_s, 2))

    pollutant_allowlist = {
        "pm25",
        "pm10",
        "so2",
        "no2",
        "co",
        "o3",
        "bc",
        "pm1",
        "pm4",
        "co2",
        "no",
        "nox",
        "ch4",
        "ufp",
    }

    pollutant_counts: dict[str, int] = {}
    for r in rows:
        p = (r.get("parameter_name") or "").strip().lower()
        if p in pollutant_allowlist:
            pollutant_counts[p] = pollutant_counts.get(p, 0) + 1

    top_params = sorted(pollutant_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    if not top_params:
        all_counts: dict[str, int] = {}
        for r in rows:
            p = (r.get("parameter_name") or "unknown").strip().lower()
            all_counts[p] = all_counts.get(p, 0) + 1
        top_params = sorted(all_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    print("top_parameters:", top_params)

    outputs = write_outputs(
        cfg,
        locations_latency_ms=locations_latency_ms,
        ny_locations=ny_locations,
        sampled_locations=sampled,
        latest_elapsed_s=latest_elapsed_s,
        rows=rows,
        top_params=top_params,
        latest_errors=latest_errors,
        collection_started_at=collection_started,
        collection_ended_at=collection_ended,
    )
    print("wrote_daily_csv:", outputs["daily_csv"])
    print("wrote_note:", outputs["note"])
    print("wrote_chart:", outputs["chart"])
    print("wrote_map:", outputs["map"])

    from nys_aq.snapshot import build_snapshot, write_snapshot

    snapshot = build_snapshot(
        report_date=cfg.run_date.isoformat(), started_at=collection_started,
        ended_at=collection_ended, stale_hours=cfg.stale_hours,
        ny_locations=ny_locations, sampled_locations=sampled,
        requests=request_records, rows=rows,
    )
    snapshot_path = cfg.repo_root / "reports" / cfg.run_date.isoformat() / "snapshot.json"
    # A storage failure fails the run, preventing CI from publishing an old
    # snapshot as new evidence. Existing report generation has already completed.
    write_snapshot(snapshot_path, snapshot)
    print("wrote_snapshot:", snapshot_path)

    pruned = prune_old_artifacts(cfg)
    if pruned["deleted_notes"] or pruned["deleted_report_dirs"]:
        print("pruned_notes:", pruned["deleted_notes"])
        print("pruned_report_dirs:", pruned["deleted_report_dirs"])
        print("retention_days:", cfg.retention_days)

    return cfg
