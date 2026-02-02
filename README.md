[![Daily NYS Air Quality](https://github.com/christopherbolduc/nys-air-quality/actions/workflows/daily.yml/badge.svg)](https://github.com/christopherbolduc/nys-air-quality/actions/workflows/daily.yml)


# NYS Air Quality (OpenAQ) — Daily Automation

This repository produces a daily, reproducible air-quality snapshot for New York State using the OpenAQ API. It filters OpenAQ locations to the NYS boundary (GeoJSON), fetches the latest readings for a stable sample of locations, runs basic data quality checks, and writes a daily note and reports.

## What it generates

For each report date (yesterday in `America/New_York`):

- `notes/YYYY-MM-DD.md`  
  A daily write-up with:
  - technical summary (counts, timings, quality checks)
  - plain-language summary
  - links to generated report artifacts

- `reports/YYYY-MM-DD/parameter_coverage.svg`  
  A simple SVG chart showing which parameters appeared in the daily sample.

- `reports/YYYY-MM-DD/map.svg`  
  An SVG map of NYS with sampled points colored by the most common parameter (typically PM2.5).

- `data/daily.csv`  
  A CSV summary with one row per report date (idempotent upsert by date).

The automation opens a daily pull request so changes are reviewed and merged by a human.

## How it works (high level)

1. Fetch OpenAQ locations in a bbox covering NYS (performance hint).
2. Filter locations to those whose coordinates fall inside the NYS boundary GeoJSON.
3. Select a stable (deterministic) sample of locations (so daily comparisons are meaningful).
4. Fetch `/locations/{id}/latest` for each sampled location with backoff for rate limits.
5. Normalize measurements into rows and label them using a cached sensor metadata map.
6. Compute basic data quality checks (e.g., stale data fraction).
7. Write daily artifacts (CSV + note + SVGs), overwriting per-day outputs.

## Local setup

### Requirements
- Python 3.11+
- An OpenAQ API key

### Install
```bash
conda activate nys-aq
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Configure Secrets
Create `.env `(not committed) from `.env.example`:

```bash
cp .env.example .env
```
Set:
- `OPENAQ_API_KEY=...`

## Run Locally
Default behavior generates a report for yesterday in `America/New_York`:
```bash
python scripts/run_daily.py
```
Run for a specifc day:
```bash
python scripts/run_daily.py --date YYYY-MM-DD
```
## GitHub Actions automation

A scheduled workflow runs daily and:

- runs `python scripts/run_daily.py`
- commits generated artifacts
- opens a pull request for review/merge (human-in-the-loop)

### Secrets in GitHub

Add this repository secret:

- `OPENAQ_API_KEY`

Repo Settings → Secrets and variables → Actions → New repository secret.

## Repository layout

- `src/nys_aq/` — production code
- `scripts/` — CLI entry points
- `notebooks/` — exploration and development (kept; outputs should be cleared before commits)
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