# Air-quality investigator: draft specification and readiness review

Status: snapshot storage, concentration map correction, and deterministic
inspection/comparison tools implemented locally, October 31, 2026; not deployed.
No investigator is implemented.
This document separates observed production behavior from proposed requirements.
Update it as decisions are made; the proposals below are not a deployment decision.

## Current validation gate

All 37 offline tests pass on local Python 3.14.2 and on Python 3.11.14 in the
original Conda environment at `miniforge3/envs/nys-aq`. Earlier stage notes saying
local Python 3.11 validation was pending are superseded by this result. GitHub's
clean install and runner validation is the remaining check before merge; a draft
feature PR is the next step. No model is connected or scheduled.

## Purpose

Help the owner understand what changed in the observed New York air-quality
measurements, where it changed, whether the evidence is trustworthy, and whether
authoritative sources support a possible explanation. Ordinary days should receive
brief treatment. Limited evidence should produce an explicit limitation.

Start with one manually invoked investigator. A model chooses which permitted
read-only tools to use, inspects their results, and writes an evidence-based
assessment. Python performs calculations and validation. A summary written from
a fixed input alone is a useful baseline, but does not exercise investigation.

`AGENTS.md` guides coding assistants working on this repository. It does not
implement or configure the future runtime investigator.

## Agreed decisions

### Snapshot meaning — October 31, 2026

Each report represents the latest available readings fetched during collection.
It is not a calendar-day average or a complete account of yesterday's air quality.
Readings can have different observation times; the snapshot is not a simultaneous
measurement of all locations.

Store collection timing separately from each reading's observation timestamp.
Future reports and investigator assessments must make those times explicit and
describe freshness and coverage. Keep the existing report-date labels and paths
for compatibility; identify them as labels rather than observation periods.
Passing `--date` must not be presented as historical measurement retrieval.

Full-day historical collection and averaging are outside the initial investigator
scope. This decision defines intended semantics; production code and report text
have not yet been changed.

## Current system: verified baseline

- GitHub Actions runs daily at 11:00 UTC, collects OpenAQ latest readings, generates
  artifacts, and opens a PR. Merging the PR updates GitHub `main`.
- Locations are filtered to the NY boundary. Deterministic sampling selects up to
  100 locations; changing the location catalog can change sample membership.
- The date label defaults to yesterday in `America/New_York`. Observations are
  fetched at run time, not restricted to that calendar day. Freshness is measured
  against collection-time clock readings, with a default threshold of 12 hours.
- The CSV retains counts, freshness, timing, errors, and pollutant coverage.
  It does not retain individual concentrations or observation timestamps.
- Maps use concentrations, but retain drawn points/colors rather than a structured
  measurement dataset. The note's plain-language explanation is fixed text.
- Retention defaults to 91 days and removes dates strictly older than the cutoff.
  This currently leaves 92 date labels inclusive. CSV history is retained.
- The workflow includes generated output paths explicitly in `add-paths`; any new
  stored dataset needs an explicit collection/retention/CI integration decision.
- Five offline unit tests pass locally on Python 3.14.2. CI uses Python 3.11.
  Local `pip check` reports no broken requirements. Existing tests cover sampling,
  basic geometry, CSV upsert, pruning, and fetch warnings in outputs.

The fetched baseline is commit `d1a3054`, with a report through October 30.
Its CSV has 271 dates, January 30–October 30; July 6–8 are absent. Those gaps
must not be interpreted as zero pollution, and their cause has not been checked.

## Prerequisites before agent implementation

### 1. Define snapshot and measurement semantics

Specify a versioned daily measurement format before implementing persistence.
Proposed fields: snapshot/report label, collection time in UTC, observation time
in UTC, sensor ID, location ID/name/coordinates, parameter, original units, value,
and validation/freshness status. Preserve sampled and requested location IDs and
fetch outcomes so unavailable and unsampled locations can be distinguished.

Define validation for null/nonfinite/negative values, invalid or future timestamps,
duplicates, unknown metadata, and mixed units. Preserve enough evidence to explain
exclusions. An empty usable dataset must be reported as insufficient data.

Choose file format, output location, retention, expected repository growth, and
handling of interrupted/repeated runs. Preserve existing output contracts. Local
experiments and tests must write to isolated directories.

Do not fabricate historical structured data from map colors. Investigate historical
OpenAQ availability separately if backfilling is needed; otherwise accumulate new
snapshots and use clearly labeled synthetic fixtures for development.

#### Storage contract — implemented locally

Write one `reports/YYYY-MM-DD/snapshot.json` alongside the map and coverage chart.
JSON supports both measurement rows and collection metadata in a single file;
no database or extra API requests are needed. Preserve all parameters already
fetched, even if the first investigator analyzes only PM2.5.

| Section | Required contents |
| --- | --- |
| Format | `schema_version`, initially `1`; readers reject unsupported versions |
| Collection | Report-date label, report timezone, UTC collection start/end, source `OpenAQ /latest`, freshness reference time and threshold |
| Locations | NY location IDs/names/coordinates, sampled flag, per-request start/end and outcome for selected locations |
| Measurements | Location ID, sensor ID, observation time in UTC, parameter, original units, numeric value or null, quality flags |
| Quality summary | Counts of returned, usable, stale, invalid, and duplicate rows; fetch failures; metadata gaps |

Outcomes distinguish unsampled locations, failed requests, successful empty
responses, and responses containing measurements. A successful HTTP request alone
does not establish usable data. Collection times refer to fetching the readings;
observation times refer to when the source says they were measured.

Use a single explicit UTC freshness reference at collection end for the new
snapshot. Preserve original timestamp/value representations for invalid records
in separate JSON-safe fields so exclusions remain auditable. Nonfinite numbers
must not be emitted as JSON `NaN` or `Infinity`. Deterministic quality flags
identify missing/nonfinite/negative values, absent or invalid times, future times,
unknown parameter/units, and duplicates. Staleness is separate from validity;
initial comparisons exclude stale or invalid readings. These new validation rules
must not silently change existing report calculations during the storage change.

Match records across snapshots using location ID, sensor ID, parameter, units,
and observation timestamp. Identical repeated observations do not count as new
measurements. Preserve and flag conflicting records rather than silently picking
one. Specify exact validation and comparison rules with tests before implementing
the comparison tools; no automatic unit conversion is proposed for storage.

Retention follows the existing dated report directory policy: deleting a dated
folder also deletes its snapshot. Keep the cumulative summary CSV indefinitely.
Do not copy snapshots into `reports/latest/` initially; use dated paths. Existing
workflow `add-paths: reports/` should include the new files, to be verified during
implementation. Keep the current inclusive retention behavior rather than changing
the number of retained dates as part of this work.

Write a snapshot to a temporary file in its destination directory, validate it,
then replace the final file atomically. A successful rerun for the same date label
replaces that date's snapshot; it is a new collection, not historical retrieval.
A failed replacement must not leave a partial JSON file or lead an investigator
to mistake an older snapshot for a successful new collection. Cross-file atomicity
for CSV/notes/maps is not provided by the current pipeline; do not claim it.

Measure actual file size using representative offline fixtures before merging.
Report both retained working-tree size and projected annual Git-history growth:
pruning files does not remove them from previous commits. Revisit storage if that
growth is excessive; avoid introducing hosted storage before it is needed.

Acceptance checks for the storage increment: valid round-trip JSON; traceable
sensor/time/unit metadata; explicit empty/failed/unsampled cases; invalid and
duplicate evidence retained safely; atomic replacement failure handling; rerun
behavior; snapshot pruning alongside reports; unchanged existing report outputs;
and no live network access or repository artifacts written by tests.

Implementation details and validation:

- `snapshot.py` builds the new evidence independently of report calculations.
  A reading is initially usable only if valid, fresh, and not duplicated. Future
  times have zero tolerance; timezone-less timestamps are invalid. Duplicate
  counts include every copy; conflicting copies are retained and flagged.
- `run_daily(repo_root=...)` supports isolated execution. Snapshot failure raises
  and fails the run after legacy output generation, before pruning. Prior snapshot
  evidence remains atomic and timestamped. Consumers can use
  `read_snapshot(..., expected_collection_start=...)` to reject an older run.
- All 15 offline tests pass locally on Python 3.14.2. A fixed-clock, mocked
  comparison against the baseline produced byte-identical CSV, note, map, and
  coverage chart. No live collection was performed.
- A separate offline CI workflow targets Python 3.11; execution on that version
  is pending GitHub validation before merging. The daily workflow is unchanged;
  its existing `reports/` inclusion covers snapshots.
- A synthetic 345-reading/123-location snapshot is 156,662 bytes: about 13.75 MiB
  for 92 retained snapshots, or 54.53 MiB of uncompressed new files annually.
  Actual Git compression and future catalog sizes will vary.

### 2. Review map correctness and labeling

The baseline map used older PM2.5 AQI breakpoints and converted each selected latest
value directly into an AQI category. Correct breakpoints alone would not establish
the averaging period required for a daily or current AQI product.

Implemented approach: display concentrations directly, with no AQI conversion or
health-category labels. A sequential blue scale uses the 5th–95th percentiles of
fresh, valid concentrations in matching units. Values outside the endpoints use
the end colors; identical concentrations use one uniform color. When no fresh
valid readings exist, show an explicit message and no numeric scale. Colors are
relative within each snapshot and cannot establish changes across dates.

Maps show collection start/end in UTC, the separate report-date label, and counts
of fresh, stale, and unavailable locations. Point tooltips show observed value,
units, and observation time. Unavailable includes unsampled/missing observations,
invalid values/times, incompatible units, and conflicting latest readings.
Stale points are gray; unavailable points are light gray. These map states are
coarser than the detailed request outcomes and flags preserved in snapshots.

Select the newest observation by parsed UTC instant before validating its value;
never fall back from an invalid latest value to an older valid concentration.
Future timestamps, negative/nonfinite/non-numeric values, and unknown or mismatched
units cannot influence the scale. Map freshness uses collection end and the
configured threshold, matching the snapshot reference. Existing CSV metrics remain
unchanged; their normalization-time freshness reference can differ slightly.

Daily notes explain snapshot semantics and include collection timing. Historical
notes/maps remain as originally generated. Coverage charts, CSV calculations,
retention, and the daily workflow are unchanged by the map correction.

Validation: 23 offline tests pass locally, including concentration endpoints,
constant/empty/all-stale scales, timezone ordering, future times, incompatible
units, invalid latest values, conflicting readings, and freshness boundaries.
Synthetic normal and empty-data maps were rendered and visually checked. Python
3.11 CI execution remains pending before merging. The earlier byte-identical
baseline comparison applies to the storage-only stage; map/note presentation
changes in this stage are intentional.

Sources checked during review:

- [EPA AQI breakpoints](https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html)
- [AirNow: using the AQI and NowCast](https://www.airnow.gov/aqi/aqi-basics/using-air-quality-index/)

### 3. Build deterministic comparison tools

Implemented in `src/nys_aq/compare.py`, independently of the daily pipeline:

- `inspect_snapshot`: returns latest eligible sensor readings summarized separately
  by original units, their observation interval, evidence pointers, and exclusions.
- `compare_snapshots`: compares exactly two caller-selected collections, in
  chronological, non-overlapping order, using the same freshness threshold.
  Defaults to PM2.5 and exact `µg/m³`; no unit conversion or implicit baseline
  selection is performed.
- Select the newest timestamp per location/sensor for the requested parameter
  before validating units/value. A newly invalid reading or changed unit cannot
  cause fallback to an older compatible reading. Tied latest rows are excluded.
- Revalidate time, value, request outcome, sampled identity, and source quality
  flags. Exclude repeated/conflicting observation timestamps, time regressions,
  excessive observation gaps, or changed location coordinates across collections.
- Return per-sensor values, absolute/percentage changes, timestamps, geographic
  metadata where available, and JSON pointers identifying both source rows.
  Percent change is null for zero baselines or numeric overflow, with a reason.
- Report before-only/after-only sensors and other-unit coverage separately. Missing
  or stale data never becomes a zero concentration. Summary medians and directional
  counts use only matched eligible sensor pairs; report unique location counts too.

Initial configurable calculation policy: maximum 36 hours between paired
observations, minimum 1 distinct matched location. The time limit is an engineering
guard for roughly daily snapshots, not a health threshold. One location allows
arithmetic; it does not support regional attribution. Raise `min_matched_locations`
for a stricter evidence requirement. Below that minimum, preserve pair evidence
but return `insufficient_data` with no aggregate summary.

Report actual collection/observation gaps. Count intervening report-date labels
without claiming those dates are missing from the repository: this tool reads
only two snapshots and explicitly does not inspect intervening history. No
calendar-day average or population-weighted statewide estimate is produced.

The read-only `scripts/compare_snapshots.py` CLI inspects one file or compares two,
prints strict JSON, and reports input paths. It needs no API keys or network.
Synthetic scenarios in `tests/test_compare.py` cover stable readings, broad
increases, a single outlier, stale/empty/invalid evidence, sample/unit changes,
repeated and conflicting timestamps, gaps, geography changes, and source tracing.
All 37 offline tests pass locally. CI now also triggers on CLI script changes;
Python 3.11 execution remains pending before merge.

Still undecided: event thresholds, regional evidence requirements/grouping, and
automatic baseline windows. The tools deliberately assign no event-severity,
health-category, or wildfire-causation judgments. Longer sensor-history tools and
external-source retrieval can be added when the investigator contract needs them.

### 4. Specify the investigator's operating contract

Proposed tools: inspect a snapshot, compare snapshots, inspect sensor history,
and retrieve approved authoritative contextual sources. External research can be
a second increment after local measurement investigation works.

Each assessment should identify:

- The snapshot and observation period actually examined.
- Meaningful changes, affected observed locations, and numerical comparisons.
- Data quality, coverage, and limits on interpretation.
- Observed facts versus possible explanations, with evidence references.
- Whether further investigation is justified or evidence is insufficient.

The first version reads data and writes a separate local assessment. It does not
edit production data/code, merge PRs, send alerts, or publish commentary. Limit
model/tool calls, execution time, and cost; record configuration, tool results,
source dates, and usage for review. Missing credentials or an investigator failure
must leave report generation usable. Treat retrieved content as untrusted evidence.

Choose runtime/provider/model, credentials, spending limit, source access, and
output destination before the first live model run. No SDK or model is selected
by this specification, and no extra service is needed for preparation.

### 5. Evaluate before scheduling

Prepare fixed offline scenarios and expected findings before tuning prompts:

| Scenario | Expected behavior |
| --- | --- |
| Ordinary stable measurements | Brief assessment; no invented event |
| Broad PM2.5 increase | Quantifies increase and observed extent |
| One unusually high sensor | Distinguishes local outlier from broad evidence |
| Mostly stale or empty data | Explains insufficient evidence; no clean-air claim |
| Changed sample, mixed units, repeated timestamps | Compares only valid compatible evidence |
| Elevated PM2.5 without contextual sources | Does not assert wildfire causation |
| API/tool/model failure or hostile retrieved text | Stops within limits; production reports remain usable |

August 31 is a candidate historical event: 14 plotted points are orange under
the existing map classification. September 14 and 20 have approximately 90%
stale readings in the CSV. These are useful review examples, not validated
wildfire events or complete measurement fixtures.

Check numerical accuracy, traceable evidence, uncertainty, tool selection, and
budget/failure behavior. Compare against a deterministic report baseline; the
investigator should add useful investigation rather than just longer prose.

## Proposed implementation order

1. Review this specification and settle measurement semantics and map labeling.
2. Implement and test measurement storage plus isolated execution support.
3. Resolve map correctness with appropriate tests and documentation.
4. Implement deterministic comparisons and offline evaluation fixtures.
5. Build one manual investigator with bounded read-only tools.
6. Evaluate results, then decide whether external research or scheduling is useful.

Preserve the existing daily automation throughout. Before merging runtime work,
incorporate the latest `origin/main`, verify compatibility on Python 3.11, and
review generated-output and retention changes explicitly. Local `main` need not
be updated daily.
