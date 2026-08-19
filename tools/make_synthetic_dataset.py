"""Generate a schema-faithful synthetic ``Cricket Data Set.xlsx``.

The real workbook is hand-curated and not committed (see ``data/README.md``).
This generator produces a workbook with the same sheet layout, column names,
and value domains so the pipeline, its tests, and CI can run without it.

    python tools/make_synthetic_dataset.py --output "Cricket Data Set.xlsx"

Pass ``--invalid`` to inject the malformed rows the cleaning code is meant to
reject (unknown symbols, out-of-range effectiveness, missing flags, blank rows).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BALL_TYPES: dict[str, str] = {
    "OC": "Off Cutter",
    "LC": "Leg Cutter",
    "Y": "Yorker",
    "B": "Bouncer",
    "GY": "Googly",
    "LB": "Leg Break",
}

SHOT_TYPES: dict[str, str] = {
    "D": "Defence",
    "Dr": "Drive",
    "DR": "Off Drive",
    "ODR": "On Drive",
    "C": "Cut",
    "P": "Pull",
    "Sw": "Sweep",
    "RS": "Reverse Sweep",
    "F": "Flick",
    "R": "Ramp",
    "LG": "Leg Glance",
    "SC": "Scoop",
    "L": "Leave",
}

DISMISSAL_KINDS: dict[str, str] = {
    "Caught": "Ct",
    "Bowled": "B",
    "Stumped": "St",
    "Run Out": "RO",
    "LBW": "LBW",
}

MATCHES: dict[str, tuple[list[str], list[str]]] = {
    "IND_vs_AFG": (
        ["Rohit Sharma", "Virat Kohli", "Shikhar Dhawan", "Kedhar Jadhav"],
        ["Rashid Khan", "Mujeeb Ur Rahman", "Aftab Alam"],
    ),
    "AUS_vs_ENG": (
        ["David Warner", "Steve Smith", "Marnus Labuschagne"],
        ["Jofra Archer", "Ben Stokes", "Adil Rashid"],
    ),
}

EFFECTIVENESS_VALUES: list[float] = [round(x / 10.0, 1) for x in range(11)]
WICKET_EFFECTIVENESS = -1.0

MATCH_COLUMNS = [
    "Sno",
    "Player Name",
    "Player ID",
    "Bowler",
    "Player ID.1",
    "Ball type",
    "Type of Shot",
    "Pitch Type",
    "Match ID",
    "Wide",
    "No ball",
    "Wicket",
    "Hit the Bat",
    "Dismissal kind",
    "Control",
    "X",
]

BLOCK_COLUMNS = [
    "Runs Scored",
    "Correct execution of shot according to the ball",
    "Effectiveness of the shot execution",
]


def build_parameters_sheet() -> pd.DataFrame:
    """The legend sheet: three independent symbol lists, ragged and NaN-padded."""
    columns = {
        "Ball Type": list(BALL_TYPES.values()),
        "ball symbol": list(BALL_TYPES),
        "Shot Type": list(SHOT_TYPES.values()),
        "shot symbol": list(SHOT_TYPES),
        "Dismissal Kind": list(DISMISSAL_KINDS),
        "dismisal symbol": list(DISMISSAL_KINDS.values()),
    }
    height = max(len(values) for values in columns.values())
    padded = {
        name: values + [np.nan] * (height - len(values))
        for name, values in columns.items()
    }
    return pd.DataFrame(padded)


def build_analysis_parameters_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Parameter": [
                "Highest percentage of off side shots",
                "Highest percentage of leg side shots",
                "Highest percentage of shots on a bouncy track",
                "Highest percentage of aggressive shots",
                "Highest percentage of defense shots",
                "Highest percentage of shots against spin",
                "Highest percentage of controlled shots",
            ],
            "Definition": [
                "share of off side shots weighted by mean effectiveness",
                "share of leg side shots weighted by mean effectiveness",
                "share of bouncy track shots weighted by mean effectiveness",
                "share of aggressive shots weighted by mean effectiveness",
                "share of defensive shots weighted by mean effectiveness",
                "share of shots against spin weighted by mean effectiveness",
                "share of controlled shots weighted by mean effectiveness",
            ],
        }
    )


def build_control_sheet(rng: np.random.Generator) -> pd.DataFrame:
    """Wide cross-tab: two leading unnamed columns then one 3-column block per ball type.

    ``Shot Type`` is written sparsely (only on the first row of each shot's
    group) because the notebook forward-fills it, and blank spacer rows are
    included because the notebook drops all-NaN rows.
    """
    shot_symbols = list(SHOT_TYPES)
    rows_per_shot = 2

    shot_type_column: list[object] = []
    for symbol in shot_symbols:
        shot_type_column.append(symbol)
        shot_type_column.extend([np.nan] * (rows_per_shot - 1))
        shot_type_column.append(np.nan)  # spacer row, dropped by dropna(how='all')

    height = len(shot_type_column)
    spacer_positions = {
        index
        for index, value in enumerate(shot_type_column)
        if index % (rows_per_shot + 1) == rows_per_shot
    }

    frame = pd.DataFrame(
        {
            "Unnamed: 0": shot_type_column,
            "Unnamed: 1": [np.nan] * height,
        }
    )

    for position, ball_symbol in enumerate(BALL_TYPES):
        runs, correct, effectiveness = [], [], []
        for index in range(height):
            if index in spacer_positions:
                runs.append(np.nan)
                correct.append(np.nan)
                effectiveness.append(np.nan)
                continue
            runs.append(float(rng.integers(0, 7)))
            correct.append(float(rng.integers(0, 2)))
            effectiveness.append(
                WICKET_EFFECTIVENESS
                if rng.random() < 0.08
                else float(rng.choice(EFFECTIVENESS_VALUES))
            )
        # Only the first column of each block carries the ball symbol; the
        # other two are read positionally and renamed by the pipeline.
        suffix = "" if position == 0 else f".{position}"
        frame[ball_symbol] = runs
        frame[f"{BLOCK_COLUMNS[1]}{suffix}"] = correct
        frame[f"{BLOCK_COLUMNS[2]}{suffix}"] = effectiveness

    return frame


def build_match_sheet(
    batsmen: list[str],
    bowlers: list[str],
    balls: int,
    rng: np.random.Generator,
    match_id: int,
) -> pd.DataFrame:
    batsman_ids = {player: 1000 + index for index, player in enumerate(sorted(batsmen))}
    bowler_ids = {player: 2000 + index for index, player in enumerate(sorted(bowlers))}
    ball_symbols = list(BALL_TYPES)
    shot_symbols = list(SHOT_TYPES)
    dismissal_symbols = list(DISMISSAL_KINDS.values())

    rows = []
    for ball in range(balls):
        batsman = str(rng.choice(batsmen))
        bowler = str(rng.choice(bowlers))
        is_wicket = rng.random() < 0.05
        rows.append(
            {
                "Sno": ball + 1,
                "Player Name": batsman,
                "Player ID": batsman_ids[batsman],
                "Bowler": bowler,
                "Player ID.1": bowler_ids[bowler],
                "Ball type": str(rng.choice(ball_symbols)),
                "Type of Shot": str(rng.choice(shot_symbols)),
                "Pitch Type": "Normal",
                "Match ID": match_id,
                # Flag columns are sparse in the source workbook: a mark when
                # true, blank otherwise.
                "Wide": "Wide" if rng.random() < 0.03 else np.nan,
                "No ball": "No ball" if rng.random() < 0.02 else np.nan,
                "Wicket": "Wicket" if is_wicket else np.nan,
                "Hit the Bat": "Hit" if rng.random() < 0.4 else np.nan,
                "Dismissal kind": str(rng.choice(dismissal_symbols))
                if is_wicket
                else np.nan,
                "Control": float(rng.integers(0, 2)),
                "X": WICKET_EFFECTIVENESS
                if is_wicket
                else float(rng.choice(EFFECTIVENESS_VALUES)),
            }
        )

    frame = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    return frame


def inject_invalid_rows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append the malformed rows the cleaning code is expected to reject."""
    template = frame.iloc[0].to_dict()
    next_sno = int(frame["Sno"].max()) + 1

    bad_ball_type = {**template, "Sno": next_sno, "Ball type": "Zz"}
    bad_shot_type = {**template, "Sno": next_sno + 1, "Type of Shot": "Qq"}
    bad_effectiveness = {**template, "Sno": next_sno + 2, "X": 7.5}
    bad_control = {**template, "Sno": next_sno + 3, "Control": 5.0}
    # A dismissal recorded without the Wicket flag: cleaning must backfill it.
    wicket_without_flag = {
        **template,
        "Sno": next_sno + 4,
        "Wicket": np.nan,
        "Dismissal kind": "Ct",
        "X": WICKET_EFFECTIVENESS,
    }
    blank_row = dict.fromkeys(MATCH_COLUMNS, np.nan)

    extra = pd.DataFrame(
        [
            bad_ball_type,
            bad_shot_type,
            bad_effectiveness,
            bad_control,
            wicket_without_flag,
            blank_row,
        ],
        columns=MATCH_COLUMNS,
    )
    return pd.concat([frame, extra], ignore_index=True)


def build_workbook(
    balls_per_match: int = 120,
    seed: int = 12345,
    invalid: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build every sheet of the workbook, keyed by sheet name."""
    rng = np.random.default_rng(seed)
    sheets: dict[str, pd.DataFrame] = {
        "control_sheet": build_control_sheet(rng),
        "parameters": build_parameters_sheet(),
        "analysis_parameters": build_analysis_parameters_sheet(),
    }
    for match_id, (name, (batsmen, bowlers)) in enumerate(MATCHES.items(), start=1):
        frame = build_match_sheet(batsmen, bowlers, balls_per_match, rng, match_id)
        if invalid:
            frame = inject_invalid_rows(frame, rng)
        sheets[name] = frame
    return sheets


def write_workbook(sheets: dict[str, pd.DataFrame], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Cricket Data Set.xlsx"),
        help="destination .xlsx path (default: ./Cricket Data Set.xlsx)",
    )
    parser.add_argument(
        "--balls-per-match",
        type=int,
        default=120,
        help="number of deliveries per match sheet (default: 120)",
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="random seed (default: 12345)"
    )
    parser.add_argument(
        "--invalid",
        action="store_true",
        help="inject malformed rows for exercising the cleaning error paths",
    )
    args = parser.parse_args(argv)

    sheets = build_workbook(
        balls_per_match=args.balls_per_match, seed=args.seed, invalid=args.invalid
    )
    path = write_workbook(sheets, args.output)
    print(f"wrote {path} ({len(sheets)} sheets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
