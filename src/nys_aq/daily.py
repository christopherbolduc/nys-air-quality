"""
Daily run logic for NYS air quality snapshots.

Designed to be called from both:
- scripts/run_daily.py (CLI)
- notebooks (import and call run_daily)
"""

from __future__ import annotations

import csv
import os
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
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()

    _load_dotenv(root / ".env")

    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAQ_API_KEY is not set. Put it in .env or set it in the environment.")

    bbox = os.getenv("BBOX", "-79.8,40.45,-71.85,45.1")
    sample_size = int(os.getenv("SAMPLE_SIZE", "100"))
    stale_hours = int(os.getenv("STALE_HOURS", "12"))

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
        _api_key=api_key,
    )


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


def fetch_latest_for_locations(cfg: DailyConfig, location_ids: list[int]) -> tuple[list[dict], list[tuple[int, str]], float]:
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
        try:
            ok.append(_get_with_backoff(session, url))
        except Exception as e:
            errors.append((loc_id, str(e)))

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

    width, height = 980, 420
    pad_left, pad_right, pad_top, pad_bottom = 230, 40, 70, 50
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    bar_h = 22
    gap = 10
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

    navy = "#0B1F3B"

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
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    parts.append(
        f'<text x="{pad_left}" y="26" font-family="Arial, sans-serif" font-size="18">'
        f'{escape(title_suffix)} — {escape(pretty_date)}'
        f"</text>"
    )
    parts.append(
        f'<text x="{pad_left}" y="48" font-family="Arial, sans-serif" font-size="12" fill="gray">'
        f'Counts returned by OpenAQ / latest across the NY daily sample (top {len(items)}). Total shown: {total}.'
        f"</text>"
    )

    # Axes
    parts.append(f'<line x1="{pad_left}" y1="{y0}" x2="{pad_left}" y2="{y1}" stroke="black"/>')
    parts.append(f'<line x1="{pad_left}" y1="{y1}" x2="{pad_left + plot_w}" y2="{y1}" stroke="black"/>')

    # Grid + tick labels
    for i in range(tick_count):
        v = i * tick_step
        x = pad_left + (v / tick_max) * plot_w if tick_max else pad_left
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="lightgray"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{y1 + 18}" font-family="Arial, sans-serif" font-size="11" '
            f'text-anchor="middle">{v}</text>'
        )

    parts.append(
        f'<text x="{pad_left + plot_w/2:.2f}" y="{height - 12}" font-family="Arial, sans-serif" '
        f'font-size="12" text-anchor="middle">measurement count</text>'
    )

    # Bars + labels
    max_label_x = width - pad_right
    for idx, (param, count) in enumerate(items):
        y = pad_top + idx * (bar_h + gap)
        bar_w = (count / max_v) * plot_w if max_v else 0.0

        parts.append(
            f'<text x="{pad_left - 10}" y="{y + bar_h - 5}" font-family="Arial, sans-serif" '
            f'font-size="12" text-anchor="end">{escape(param)}</text>'
        )

        parts.append(
            f'<rect x="{pad_left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" '
            f'fill="{navy}" stroke="black" stroke-width="0.5"/>'
        )

        pct = int(round((count / total) * 100))
        label = f"{count} ({pct}%)"

        outside_x = pad_left + bar_w + 8
        approx_text_w = 7 * len(label)
        if outside_x + approx_text_w <= max_label_x:
            parts.append(
                f'<text x="{outside_x:.2f}" y="{y + bar_h - 5}" font-family="Arial, sans-serif" font-size="12">'
                f"{escape(label)}</text>"
            )
        else:
            inside_x = max(pad_left + 6, pad_left + bar_w - 6)
            parts.append(
                f'<text x="{inside_x:.2f}" y="{y + bar_h - 5}" font-family="Arial, sans-serif" font-size="12" '
                f'fill="white" text-anchor="end">{escape(label)}</text>'
            )

    # In-chart dynamic dictionary (bottom-right)
    if legend_lines:
        line_h = 14
        title_h = 14
        pad_box = 10

        # Size the box conservatively
        box_w = 330
        box_h = pad_box * 2 + title_h + len(legend_lines) * line_h

        # Position bottom-right inside the plotting area
        box_x = pad_left + plot_w - box_w - 10
        box_y = y1 - box_h - 10

        # Background
        parts.append(
            f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_w}" height="{box_h}" '
            f'fill="white" stroke="black" stroke-width="0.6"/>'
        )

        tx = box_x + pad_box
        ty = box_y + pad_box + 11

        parts.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-family="Arial, sans-serif" font-size="11" fill="black">')
        parts.append(f'<tspan x="{tx:.2f}" dy="0">Dictionary</tspan>')
        for line in legend_lines:
            parts.append(f'<tspan x="{tx:.2f}" dy="{line_h}">{escape(line)}</tspan>')
        parts.append("</text>")

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
) -> Path:
    from xml.sax.saxutils import escape

    map_path = report_dir / "map.svg"

    # Latest value per location for the primary parameter
    loc_value: dict[int, tuple[float, str, bool]] = {}
    for r in rows:
        if (r.get("parameter_name") or "unknown") != primary_param:
            continue
        loc_id = r.get("locationsId")
        val = r.get("value")
        dt = r.get("datetime_utc") or ""
        stale = bool(r.get("stale"))
        if not isinstance(loc_id, int) or not isinstance(val, (int, float)):
            continue
        prev = loc_value.get(loc_id)
        if prev is None or dt > prev[1]:
            loc_value[loc_id] = (float(val), dt, stale)

    points: list[tuple[float, float, float | None, bool]] = []
    for loc in ny_locations:
        loc_id = loc.get("id")
        c = loc.get("coordinates") or {}
        try:
            lon = float(c.get("longitude"))
            lat = float(c.get("latitude"))
        except (TypeError, ValueError):
            continue
        v = loc_value.get(loc_id) if isinstance(loc_id, int) else None
        if v:
            points.append((lon, lat, v[0], v[2]))
        else:
            points.append((lon, lat, None, True))

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        k = (len(xs) - 1) * p
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if c == f:
            return xs[f]
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def color_ramp(t: float) -> str:
        t = clamp01(t)
        if t < 0.5:
            tt = t / 0.5
            r = int(lerp(52, 243, tt))
            g = int(lerp(152, 156, tt))
            b = int(lerp(219, 18, tt))
        else:
            tt = (t - 0.5) / 0.5
            r = int(lerp(243, 231, tt))
            g = int(lerp(156, 76, tt))
            b = int(lerp(18, 60, tt))
        return f"rgb({r},{g},{b})"
    
    def color_for_t(t: float) -> str:
        return color_ramp(clamp01(t))

    vals_fresh = [v for _, _, v, stale in points if isinstance(v, (int, float)) and (not stale)]
    vals_all = [v for _, _, v, _ in points if isinstance(v, (int, float))]
    scale_vals = vals_fresh if vals_fresh else vals_all
    vmin = percentile(scale_vals, 0.05)
    vmax = percentile(scale_vals, 0.95)
    if vmax <= vmin:
        vmax = vmin + 1.0

    def value_to_t(v: float) -> float:
        return clamp01((v - vmin) / (vmax - vmin))

    def color_for(v: float | None, stale: bool) -> str:
        if v is None:
            return "lightgray"
        if stale:
            return "rgb(180,180,180)"
        return color_ramp(value_to_t(v))

    def radius_for(v: float | None, stale: bool) -> float:
        if v is None or stale:
            return 3.5
        t = value_to_t(v)
        return 3.0 + t * 6.0  # 3 -> 9

    # Bounds from NY polygon
    lons: list[float] = []
    lats: list[float] = []
    for poly in ny_multipolygon:
        for lon, lat in poly[0]:
            lons.append(float(lon))
            lats.append(float(lat))
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    width, height = 900, 520
    pad = 20

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = pad + (lon - min_lon) / (max_lon - min_lon) * (width - 2 * pad)
        y = pad + (max_lat - lat) / (max_lat - min_lat) * (height - 2 * pad)
        return x, y

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    display_param = pretty_parameter_name(primary_param)

    fresh_n = sum(1 for _, _, v, stale in points if (v is not None) and (not stale))
    stale_n = sum(1 for _, _, v, stale in points if (v is not None) and stale)
    missing_n = sum(1 for _, _, v, _ in points if v is None)

    parts.append(
    f'<text x="{pad}" y="18" font-family="Arial, sans-serif" font-size="16">'
    f'New York State — {escape(display_param)} (latest)'
    f"</text>"
)
    parts.append(
    f'<text x="{pad}" y="36" font-family="Arial, sans-serif" font-size="12" fill="gray">'
    f'{escape(report_date)} • {escape(sample_label)} • fresh: {fresh_n} • stale: {stale_n} • missing: {missing_n}'
    f"</text>"
)

    for poly in ny_multipolygon:
        outer = poly[0]
        d = []
        for i, (lon, lat) in enumerate(outer):
            x, y = project(float(lon), float(lat))
            d.append(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}")
        d.append("Z")
        parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="black" stroke-width="1"/>')

    for lon, lat, v, stale in points:
        x, y = project(lon, lat)
        fill = color_for(v, stale)
        r = radius_for(v, stale)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="black" stroke-width="0.6"/>')

        # Legend box (bottom-left, above the footer line)
    legend_w = 260
    legend_h = 54
    legend_x = pad
    legend_y = height - 18 - legend_h - 10  # 10px above footer

    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" '
        f'fill="white" stroke="black" stroke-width="0.6"/>'
    )

    parts.append(
        f'<text x="{legend_x + 10}" y="{legend_y + 18}" font-family="Arial, sans-serif" font-size="12">'
        f'{escape(pretty_parameter_name(primary_param))} scale ({escape(units)})'
        f"</text>"
    )

    # Color ramp (left->right)
    ramp_x0 = legend_x + 10
    ramp_y0 = legend_y + 26
    ramp_w = legend_w - 20
    ramp_h = 10
    steps = 24
    step_w = ramp_w / steps

    for i in range(steps):
        t0 = i / steps
        x = ramp_x0 + i * step_w
        parts.append(
            f'<rect x="{x:.2f}" y="{ramp_y0:.2f}" width="{step_w + 0.6:.2f}" height="{ramp_h}" '
            f'fill="{color_for_t(t0)}" stroke="none"/>'
        )

    parts.append(
        f'<text x="{ramp_x0:.2f}" y="{legend_y + 50}" font-family="Arial, sans-serif" font-size="11" fill="gray">'
        f"{vmin:.2f}</text>"
    )
    parts.append(
        f'<text x="{(ramp_x0 + ramp_w):.2f}" y="{legend_y + 50}" font-family="Arial, sans-serif" font-size="11" fill="gray" '
        f'text-anchor="end">{vmax:.2f}</text>'
    )

    # Footer line (kept, now below legend)
    parts.append(
        f'<text x="{pad}" y="{height - 18}" font-family="Arial, sans-serif" font-size="12" fill="gray">'
        f"Scale uses p5–p95; stale points shown in gray"
        f"</text>"
    )
    parts.append("</svg>")

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
    lines.append(f"- Report date (America/New_York): {report_date}")
    lines.append(f"- Locations in NY boundary: {len(ny_locations)}")
    lines.append(f"- Locations used: {sample_label}")
    lines.append(f"- Measurements normalized: {len(rows)}")
    lines.append(f"- Locations catalog latency: {locations_latency_ms} ms")
    lines.append(f"- Latest fetch duration: {latest_elapsed_s:.2f} s")
    lines.append("")
    lines.append("### Data quality checks")
    lines.append(f"- Stale threshold: {cfg.stale_hours} hours")
    lines.append(f"- Stale fraction: {stale_fraction:.3f}")
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
    if primary_param == "pm25":
        lines.append("")
        lines.append(
            "PM2.5 is a measure of very small airborne particles (2.5 micrometers or smaller) that can be breathed deep into the lungs."
        )
    lines.append("")
    lines.append("## Reports")
    lines.append(f"- reports/{report_date}/parameter_coverage.svg")
    lines.append(f"- reports/{report_date}/map.svg")
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
    )

    return {
        "daily_csv": csv_path,
        "note": note_path,
        "chart": chart_path,
        "map": map_path,
    }


def run_daily(*, run_date: date | None = None, tz_name: str = "America/New_York") -> DailyConfig:
    cfg = load_config(run_date=run_date, tz_name=tz_name)

    locations_latency_ms, locations = fetch_locations(cfg)
    ny_poly = load_ny_multipolygon(cfg)
    ny_locations = filter_locations_in_ny(ny_poly, locations)

    sampled = stable_sample_locations(ny_locations, cfg.sample_size, salt=cfg.bbox)
    location_ids = [loc["id"] for loc in sampled if isinstance(loc.get("id"), int)]

    latest_ok, latest_errors, latest_elapsed_s = fetch_latest_for_locations(cfg, location_ids)
    if latest_errors:
        raise RuntimeError(f"Latest fetch errors: {latest_errors[:1]}")

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
        # Core criteria pollutants (OpenAQ focus)
        "pm25",
        "pm10",
        "so2",
        "no2",
        "co",
        "o3",
        "bc",
        # Limited set extras OpenAQ mentions
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

    # Fallback: if none of the allowlisted pollutants appear, show the top parameters overall
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
    )
    print("wrote_daily_csv:", outputs["daily_csv"])
    print("wrote_note:", outputs["note"])
    print("wrote_chart:", outputs["chart"])
    print("wrote_map:", outputs["map"])

    return cfg