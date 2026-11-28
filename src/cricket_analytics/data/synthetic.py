"""Generate a small synthetic workbook with the same layout as the real dataset.

The real ``Cricket Data Set.xlsx`` is not committed; this fixture lets the
pipeline, tests and CI run end-to-end. It deliberately includes the dirty
values the cleaning code handles (blank rows, lowercase codes, ``-1`` control
values, free-text dismissals, misspelled batsman names, an invalid shot code).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openpyxl import Workbook

BALL_TYPES = {
    "OC": "Off Cutter",
    "LB": "Leg Break",
    "GY": "Googly",
    "Y": "Yorker",
    "BO": "Bouncer",
    "SL": "Slower Ball",
    "IS": "In Swing",
    "OS": "Out Swing",
}
SHOT_TYPES = {
    "D": "Defence",
    "DR": "Drive",
    "C": "Cut",
    "P": "Pull",
    "SW": "Sweep",
    "RS": "Reverse Sweep",
    "ODR": "On Drive",
    "F": "Flick",
    "R": "Ramp",
    "LG": "Leg Glance",
    "SC": "Scoop",
    "L": "Leave",
    "S": "Square Drive",
}
DISMISSALS = {"Ct": "Caught", "B": "Bowled", "St": "Stumped", "LBW": "Leg Before Wicket"}
BATSMEN = [
    "Virat Kohli",
    "Rohit Sharma",
    "Shikhar Dhawan",
    "Kedhar Jadhav",
    "MS Dhoni",
    "Rashid Khan",
    "Mohammad Nabi",
    "Hashmatullah Shahidi",
]
BOWLERS = ["Mujeeb Ur Rahman", "Gulbadin Naib", "Jasprit Bumrah", "Yuzvendra Chahal"]
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


def _parameters_rows() -> list[list]:
    balls, shots, dismissals = (
        list(BALL_TYPES.items()),
        list(SHOT_TYPES.items()),
        list(DISMISSALS.items()),
    )
    rows = [
        [
            "Ball Type",
            "ball symbol",
            "Shot Type",
            "shot symbol",
            "Dismissal Kind",
            "dismisal symbol",
        ]
    ]
    for i in range(max(len(balls), len(shots), len(dismissals))):
        row: list = []
        for pairs in (balls, shots, dismissals):
            symbol, name = pairs[i] if i < len(pairs) else (None, None)
            row += [name, symbol]
        rows.append(row)
    return rows


def _control_rows(rng: np.random.Generator) -> list[list]:
    header: list = [None, None]
    for symbol in BALL_TYPES:
        header += ["Sl" if symbol == "SL" else symbol, None, None]
    rows = [header]
    for shot in SHOT_TYPES.values():
        for i, label in enumerate(["Runs", "Edge"]):
            row: list = [shot if i == 0 else None, label]
            for _ in BALL_TYPES:
                row += [
                    int(rng.integers(0, 7)),
                    float(rng.integers(0, 2)),
                    round(float(rng.integers(0, 11)) / 10, 1),
                ]
            rows.append(row)
        rows.append([None] * len(header))
    return rows


def _match_rows(rng: np.random.Generator, match_id: int, n_balls: int) -> list[list]:
    balls, shots = list(BALL_TYPES), list(SHOT_TYPES)
    affinity = {(b, s): rng.uniform(0.1, 0.9) for b in balls for s in shots}
    skill = {name: rng.uniform(-0.15, 0.15) for name in BATSMEN}
    rows = [MATCH_COLUMNS]
    for sno in range(1, n_balls + 1):
        batsman = BATSMEN[int(rng.integers(len(BATSMEN)))]
        bowler = BOWLERS[int(rng.integers(len(BOWLERS)))]
        ball, shot = balls[int(rng.integers(len(balls)))], shots[int(rng.integers(len(shots)))]
        effectiveness = float(
            np.clip(affinity[(ball, shot)] + skill[batsman] + rng.normal(0, 0.1), 0, 1)
        )
        wicket = rng.random() < 0.04
        dismissal = (
            rng.choice(["Ct", "C", "B", "Leg Stump", "Stump Out", "Caught"]) if wicket else None
        )
        control = 0.0 if wicket else float(effectiveness > 0.5)
        if rng.random() < 0.03:
            control = -1.0
        rows.append(
            [
                sno,
                batsman,
                1000 + BATSMEN.index(batsman),
                bowler,
                2000 + BOWLERS.index(bowler),
                ball.lower() if rng.random() < 0.05 else ball,
                shot.lower() if rng.random() < 0.05 else shot,
                "Flat" if rng.random() < 0.9 else None,
                match_id,
                "Wide" if rng.random() < 0.03 else None,
                "No ball" if rng.random() < 0.02 else None,
                "Wicket" if wicket and rng.random() < 0.7 else None,
                "Yes" if rng.random() < 0.8 else None,
                dismissal,
                control,
                -1.0 if wicket else round(effectiveness, 1),
            ]
        )
    rows[5][1] = "Kedar Jadhav"
    rows[6][1] = "Shikar Dhawan"
    rows[7][6] = "ZZ"
    rows.insert(10, [None] * len(MATCH_COLUMNS))
    return rows


def make_synthetic_workbook(path: str | Path, n_balls: int = 300, seed: int = 0) -> Path:
    """Write the synthetic workbook to ``path`` and return it."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    sheets = {
        "Control Sheet": _control_rows(rng),
        "Parameters": _parameters_rows(),
        "Analysis Parameters": [
            ["Parameter", "Description"],
            ["Off side score", "Share of off-side shots x mean effectiveness"],
            [None, None],
            ["Control score", "Share of controlled shots x mean effectiveness"],
        ],
        "IND_vs_AFG": _match_rows(rng, 1, n_balls),
        "AUS vs ENG": _match_rows(rng, 2, n_balls),
    }
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path
