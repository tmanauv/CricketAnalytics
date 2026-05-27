# `data/`

This directory is the canonical location for the dataset used by
`SDP.ipynb`. The raw dataset is **not committed** to the repository — see
[Provenance](#provenance) below for how to obtain it.

## Expected file

```
data/Cricket Data Set.xlsx
```

Or, equivalently, place it at the repository root (the notebook currently
reads `Cricket Data Set.xlsx` from the working directory). Either path is
matched by `.gitignore`, so you cannot accidentally commit the workbook.

## Expected schema

The workbook is a single `.xlsx` file with multiple sheets. The notebook
expects (at least) the following sheets:

### `control_sheet`

A wide cross-tab where each block of three columns represents a ball type
(by `ball symbol`), and the rows enumerate shot types. Within each block:

| Column                                                | Meaning                                          |
| ----------------------------------------------------- | ------------------------------------------------ |
| `Runs Scored`                                         | Runs scored by that shot against that ball type. |
| `Correct execution of shot according to the ball`     | `0.0` / `1.0` — was the shot the right choice?   |
| `Effectiveness of the shot execution`                 | `0.0`–`1.0` (steps of `0.1`) or `-1.0` for a wicket. |

The first column of each row group identifies the *Shot Type* (the column
is unnamed in the source workbook and is renamed to `Shot Type` by the
notebook).

### `parameters`

The legend mapping symbols to human-readable names:

| Column            | Example values                          |
| ----------------- | --------------------------------------- |
| `Ball Type`       | `Off Cutter`, `Leg Break`, `Yorker`, …  |
| `ball symbol`     | `OC`, `LB`, `Y`, …                      |
| `Shot Type`       | `Drive`, `Cut`, `Pull`, …               |
| `shot symbol`     | `D`, `C`, `P`, …                        |
| `Dismissal Kind`  | `Caught`, `Bowled`, `Stumped`, …        |
| `dismisal symbol` | `Ct`, `B`, `St`, …                      |

### `analysis_parameters`

Auxiliary parameters used by the per-batsman scoring functions. Treated as
a freeform reference table.

### `<team1>_vs_<team2>` (one per match)

Ball-by-ball data for a single match. Examples observed in the original
notebook: `IND_vs_AFG`, `ind_vs_afg`, etc. Expected columns include:

| Column                          | Notes                                       |
| ------------------------------- | ------------------------------------------- |
| `Sno`                           | Sequence number (dropped during cleaning).  |
| `Player Name` / `Player ID`     | Batsman; renamed to `Batsman Name` / `Batsman Player ID`. |
| `Bowler` / `Player ID.1`        | Bowler; renamed to `Bowler Name` / `Bowler Player ID`. |
| `Ball type`                     | One of the symbols from `parameters['ball symbol']`. |
| `Type of Shot`                  | One of the symbols from `parameters['shot symbol']`. |
| `Pitch Type`                    | Optional, dropped if uniform.               |
| `Match ID`                      | Optional, dropped if uniform.               |
| `Wide`, `No ball`, `Wicket`, `Hit the Bat` | Boolean-ish flags; coerced to `0.0` / `1.0`. |
| `Dismissal kind`                | One of the symbols from `parameters['dismisal symbol']`, or empty. |
| `Control`                       | `0.0` / `1.0`.                              |
| `X` (Effectiveness)             | `0.0`–`1.0` step `0.1`, or `-1.0`. Renamed to `Effectiveness`. |

The notebook's cleaning code (`clean_df`, `validate_column_values`) is the
source of truth for the exact accepted values per column.

## Provenance

The dataset was hand-curated from ball-by-ball data scraped from
[ESPN Cricinfo](https://www.espncricinfo.com/), with effectiveness scores
assigned manually per the framework described in the project README.

A reproducible scraper is **not yet** part of this repository — it is
tracked as PR-10e in the improvement plan
(`devin/<ts>-cricinfo-scraper`). Until that lands, contributors who do not
already have the workbook should either:

- request it directly from the project author, or
- generate a small synthetic fixture for testing (this is the approach
  PR-02 will take for CI).

## Layout convention going forward

```
data/
├── README.md           # this file
├── raw/                # untouched source files (gitignored)
├── interim/            # intermediate cleaned outputs (gitignored)
└── processed/          # model-ready feature tables (gitignored)
```

None of `raw/`, `interim/`, or `processed/` exist yet — they will be
created by the package extraction in PR-05. The directories listed here
are reserved so future PRs do not need to revisit `.gitignore`.
