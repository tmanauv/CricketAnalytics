# Roadmap

This file consolidates the plan that was previously spread across PR
descriptions, `README.md` notes, and `data/README.md` references to
"the improvement plan". It is the single source of truth for what this
repository is being turned into and in what order.

## End goal

`SDP.ipynb` is retired in favour of a `cricket_analytics` Python package
that reproduces the whole study end to end from raw data:

```
scrape  →  clean  →  engineer features  →  train / tune  →  evaluate / report
```

Concretely, "done" means all of the following hold:

1. Every step is importable, parameterised code under
   `src/cricket_analytics/` — no notebook globals, no cell-execution
   order dependencies.
2. `cricket-analytics scrape | prepare | train | evaluate` runs the whole
   study from a CLI, configured by `configs/*.yaml`.
3. The pipeline is statistically defensible: no target leakage, no
   hyper-parameter selection on the test set, no pre-split
   `fit_transform`, and a splitting strategy that respects the fact that
   rows are grouped by batsman.
4. Bad input data fails loudly (or warns with counts) rather than being
   silently dropped.
5. CI runs lint + unit tests on a committed synthetic fixture, so the
   uncommitted, hand-curated `Cricket Data Set.xlsx` is never required
   to develop or review.
6. Reported metrics in `README.md` come from a reproducible command, not
   from stale notebook output.

## Known defects driving the plan

Derived from the current notebook and the open PRs:

| Defect | Where | Phase |
| --- | --- | --- |
| Target leakage: per-batsman aggregate features are computed over the full dataset, including each row's own `Effectiveness` | notebook cells 51–77 | 2 |
| Hyper-parameters chosen by `r2_score` on the *test* set (`fit_and_evaluate` grid loops, then "top 10 refit and print R²") | cells 89–105 | 2 |
| `StandardScaler().fit_transform(X)` and `LabelEncoder` fit before `train_test_split` | cells 81–87 | 2 |
| Random `train_test_split` ignores batsman grouping, so the same batsman (and their aggregate features) appears in train and test | cell 87 | 2 |
| Silent data drops / stale-global guard in `drop_incorrect_data`, cryptic `KeyError`s on bad symbols | cells 36–39 | 1 (PR #9) |
| Vulnerable pins (keras 2.15, protobuf 4.25) | `requirements.txt` | 1 (PR #10) |
| Seven copy-pasted scoring functions, eight duplicated `cross_val_score` calls, five-deep tuning loops | cells 51–105 | 1 (PRs #11, #12) |
| Final DL cells cannot run as written (`header=None` object dtypes, `input_dim=6` against 24 features, undefined `X1`/`X2`) | cells 106–109 | 3 |
| No reproducible path from Cricinfo to the workbook | — | 4 |

## Phases

Each phase is one or more PRs that land independently on `main`.

### Phase 0 — reproducible fixture (done, #13)

`tools/make_synthetic_dataset.py` generates a schema-faithful
`Cricket Data Set.xlsx` (control sheet with three-column ball-type
blocks, `parameters`, `analysis_parameters`, and `<team>_vs_<team>` match
sheets), optionally with deliberately invalid rows for error-path tests.
Every later phase is testable and reviewable without the private
workbook, and CI has something to run.

### Phase 1 — correctness and hygiene of the existing notebook (in review)

Already open, no new work needed beyond review and merge:

- **#9** — fail fast on bad data; warn with counts instead of silent drops.
- **#10** — security pins (keras 3, protobuf ≥ 5.29.6) and the Keras 3
  optimizer migration.
- **#11** — extraction of loading / cleaning / features / preprocessing /
  evaluation into `src/cricket_analytics` with unit tests.
- **#12** — de-duplication of the seven scoring functions, the
  cross-validation calls, the tuning loops, and the importance plots.

Recommended merge order: #9 → #11 → #12 → #10 (#11 and #12 both add
`src/cricket_analytics`, so whichever lands second rebases onto the
other; #10 touches only pins and one optimizer call site).

The older #3/#4 pair (`pyproject.toml` scaffold, Cricinfo scraper) is a
parallel lineage of the same plan; its scaffold is superseded by #11/#12
and its scraper is folded into Phase 4.

### Phase 2 — statistically defensible modelling (this PR)

The core of the end goal. A `cricket_analytics.modeling` layer that:

- computes per-batsman aggregate features **inside** each
  cross-validation fold, from training rows only, so a row's own
  `Effectiveness` never reaches its own features;
- puts encoding and scaling inside an sklearn `Pipeline`/`ColumnTransformer`
  so they are fit per fold;
- splits with `GroupKFold`/`GroupShuffleSplit` on `Batsman Name`;
- tunes with an inner CV on the training split only, leaving the test
  split untouched until a single final evaluation;
- reports a baseline (predict-the-mean and a per-ball-type mean) so the
  honest numbers have something to beat.

Expect the reported R² to drop sharply relative to the notebook. That is
the point: the notebook's numbers are inflated by leakage and by tuning
on the test set. `tests/test_leakage_regression.py` pins this down — on a
dataset whose target is pure noise the notebook's recipe still reports
R² > 0.15, while the pipeline here stays at or below zero.

One further column of leakage is removed here rather than fixed: `Wicket`
and `Dismissal kind` are dropped from the feature set, because
`Effectiveness` is `-1.0` exactly when the ball is a wicket, so they hand
a model the answer for every negative target.

### Phase 3 — deep-learning comparison, repaired

Rebuild the Keras MLP path on top of the Phase 2 pipeline: a
`KerasRegressor` inside the same `Pipeline` and the same group-aware CV,
replacing the `matches_df.csv` round-trip, the `header=None` object
dtypes, and the hard-coded `input_dim`.

### Phase 4 — reproducible ingestion

Land the Cricinfo scraper (#4) as `cricket_analytics.scraper`, plus the
mapping from scraped deliveries to the workbook's ball-type / shot-type /
effectiveness encoding. This is what makes the dataset regenerable
rather than hand-curated; it is last because everything downstream can
be developed against the Phase 0 fixture.

### Phase 5 — CLI, reporting, and notebook retirement

`cricket-analytics` Typer CLI over the Phase 2–4 modules, config files
under `configs/`, a metrics/plot report written to `reports/`, refreshed
`README.md` numbers, and `SDP.ipynb` reduced to a thin narrative that
imports the package (or deleted, with the original preserved under
`notebooks/`).

## Non-goals

- Committing the dataset. It stays uncommitted and `.gitignore`d; the
  fixture and the Phase 4 scraper are the substitutes.
- Serving the model (API, web UI, deployment). This is a study, not a
  product.
- Chasing the notebook's headline R² numbers. Phase 2 deliberately
  invalidates them.
