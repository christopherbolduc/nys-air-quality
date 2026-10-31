[![Daily NYS Air Quality](https://github.com/christopherbolduc/nys-air-quality/actions/workflows/daily.yml/badge.svg)](https://github.com/christopherbolduc/nys-air-quality/actions/workflows/daily.yml)
![Last commit](https://img.shields.io/github/last-commit/christopherbolduc/nys-air-quality)

# NYS Air Quality (OpenAQ) — Daily Automation

This repository produces a daily, reproducible air-quality snapshot for New York State using the OpenAQ API. It filters OpenAQ locations to the NYS boundary (GeoJSON), fetches the latest readings for a stable set of NY locations, runs basic data quality checks, and writes a daily note and SVG reports.

Daily runs are automated via GitHub Actions and proposed as pull requests for human review and merge.

## Example output

**Map (latest values, colored by the selected pollutant parameter):**

![Map example](reports/latest/map.svg?)

**Pollutant coverage (counts + share of returned measurements, with an in-chart dictionary):**

![Coverage example](reports/latest/parameter_coverage.svg)

## What it generates

For each report date (yesterday in `America/New_York`):

The report date is an output label. Each run fetches the latest available readings
at collection time; it does not retrieve a full calendar day of observations.

- `notes/YYYY-MM-DD.md`  
  A daily write-up with:
  - technical summary (counts, timings, quality checks)
  - plain-language summary
  - links to generated report artifacts

- `reports/YYYY-MM-DD/map.svg`  
  NYS map of location points colored by the selected pollutant parameter (latest value). Includes a p5–p95 legend and freshness summary (fresh vs stale vs missing).

  New maps display concentrations using a blue scale based on fresh, valid
  readings. They show the UTC collection interval and distinguish fresh, stale,
  and unavailable/invalid readings. Point tooltips give values and observation
  times. Colors are relative to each snapshot, not health categories; compare
  numeric concentrations across dates. These maps do not calculate daily or
  NowCast AQI. Existing historical SVGs are preserved as originally generated.

- `reports/YYYY-MM-DD/parameter_coverage.svg`  
  Pollutant coverage chart showing which pollutant parameters were returned by `/latest` across the NY locations. Percentages represent the share of measurements in that snapshot (not severity).

- `data/daily.csv`  
  A CSV summary with one row per report date (idempotent upsert by date).

- `reports/YYYY-MM-DD/snapshot.json`
  A versioned, machine-readable snapshot of all fetched parameters, with sensor
  and location identities, units, original values, observation timestamps,
  collection timing, request outcomes, and data-quality flags. New snapshots are
  produced going forward; old report folders do not gain historical measurements.
  This is preparation for an air-quality investigator, not an implemented agent.

The automation opens a daily pull request so changes are reviewed and merged by a human.

## How it works (high level)

1. Fetch OpenAQ locations in a bbox covering NYS (performance hint).
2. Filter locations to those whose coordinates fall inside the NYS boundary GeoJSON.
3. Select a stable set of NY locations (deterministic; uses all available locations when the set is small).
4. Fetch `/locations/{id}/latest` with backoff to handle rate limits.
5. Normalize measurements into rows and label them using a cached sensor metadata map.
6. Compute data quality checks (e.g., stale data fraction).
7. Write daily artifacts (CSV + note + SVGs), overwriting per-day outputs.

## GitHub Actions automation

A scheduled workflow runs daily and:

- runs `python scripts/run_daily.py`
- commits generated artifacts
- opens a pull request for review/merge (human-in-the-loop)

The workflow uses a dated branch name (e.g., `automation/daily-YYYY-MM-DD`) and targets `main`.

### Secrets in GitHub

Add this repository secret:

- `OPENAQ_API_KEY`

Repo Settings → Secrets and variables → Actions → New repository secret.

### Generated artifacts

Daily notes and reports are generated in GitHub Actions and merged via pull request. The repository treats `notes/`, `reports/`, and `data/daily.csv` as build outputs produced by CI (not hand-edited).

## Artifact retention (automatic pruning)

To keep the repository readable over time, the daily job automatically removes older generated artifacts:

- `notes/YYYY-MM-DD.md`
- `reports/YYYY-MM-DD/` (daily SVG outputs)

The CSV summary (`data/daily.csv`) is retained indefinitely so longer-term trends can still be analyzed.

Measurement snapshots are deleted with their dated report folders. With the
current inclusive cutoff, `RETENTION_DAYS=91` can retain 92 date labels. Deletion
limits current files; previous versions remain in Git history.

Retention is controlled by `RETENTION_DAYS` (default: 91). This keeps roughly a quarter’s worth of daily snapshots visible for review while preventing the repo from accumulating unlimited artifacts.

## Local setup

### Requirements
- Python 3.11+
- An OpenAQ API key
- Dependencies: `requests`, `tzdata`, `pandas`

### Install
```bash
conda activate nys-aq
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Configure Secrets
Create `.env` (not committed) from `.env.example`:

```bash
cp .env.example .env
```
Set:
- `OPENAQ_API_KEY=...`

Optional configuration (via `.env` or environment variables):

- `MAP_PARAM` — OpenAQ parameter code used to color the map (defaults to the day’s most common pollutant parameter)

Optional:
- `RETENTION_DAYS` (default: 91) — delete notes/reports older than this many days

## Run locally
Default behavior generates a report for yesterday in `America/New_York`:
```bash
python scripts/run_daily.py
```
Run for a specifc date:
```bash
python scripts/run_daily.py --date YYYY-MM-DD
```

`--date` changes the output label and retention reference, not the observations
requested from OpenAQ. Rerunning a label replaces its snapshot with a new
collection. A snapshot-write failure fails the run; any previous snapshot remains
as older evidence, with its original collection timestamps. This prevents CI from
publishing a failed collection as a successful update. Outputs are not written as
one transaction across CSV, notes, maps, and snapshots.

For development, `run_daily(repo_root=...)` supports an isolated root containing
`data/nys_boundary.geojson`; live collection still requires the API key. Prefer
mocked requests and temporary directories for tests.

## Development

Inspect one saved measurement snapshot, or compare two specified collections:

```bash
python scripts/compare_snapshots.py reports/YYYY-MM-DD/snapshot.json
python scripts/compare_snapshots.py /path/to/baseline/snapshot.json /path/to/current/snapshot.json
```

The tool reads files and prints JSON; it makes no network requests and changes no
reports. It defaults to PM2.5 in exact `µg/m³` units, with no automatic conversion.
Use `--help` for the configurable observation-gap and matched-location limits.
Historical report directories created before snapshot storage have no input JSON;
comparisons need newly collected snapshots or clearly labeled synthetic fixtures.

Results include per-sensor values, deltas, percentages, observation times,
matched coverage, and source JSON pointers. Stale/invalid readings, repeats,
conflicts, changed locations, incompatible units, and excessive observation gaps
are excluded explicitly. Missing evidence produces `insufficient_data`, not a
zero concentration. Percent change is null for a zero baseline or numeric overflow.
Summary medians describe matched sensors, not statewide averages. No health,
wildfire-causation, or event-severity judgment is assigned.

Run the offline test suite in the installed environment:

```bash
python -m unittest discover -s tests -v
```

The separate `Offline tests` workflow checks Python 3.11 for code/test changes.
It does not require API secrets or perform live collection. See `AGENTS.md` for
repository guidance and `SPEC.md` for investigator requirements and progress.

## Repository layout

- `src/nys_aq/` — production code
- `scripts/` — CLI entry points
- `notebooks/` — exploration and development (outputs should be cleared before commits)
- `data/` — NYS boundary GeoJSON, daily CSV summary, sensor metadata cache
- `notes/` — daily notes (Markdown)
- `reports/` — daily SVG artifacts

### Notes on rate limits

OpenAQ enforces rate limits. The workflow throttles requests and uses backoff to remain reliable in CI. Runs are expected to complete within a couple of minutes.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Author

**Christopher Bolduc**  
[LinkedIn](https://www.linkedin.com/in/christopher-david-bolduc/) • [GitHub](https://github.com/christopherbolduc)
