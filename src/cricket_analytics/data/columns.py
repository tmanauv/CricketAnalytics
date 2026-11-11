"""Canonical column names of the cleaned ball-by-ball table."""

BATSMAN = "Batsman Name"
BATSMAN_ID = "Batsman Player ID"
BOWLER = "Bowler Name"
BOWLER_ID = "Bowler Player ID"
BALL_TYPE = "Ball type"
SHOT = "Type of Shot"
CONTROL = "Control"
DISMISSAL = "Dismissal kind"
WICKET = "Wicket"
TARGET = "Effectiveness"
MATCH = "Match"

NO_DISMISSAL = "NONE"
CONTROL_VALUES = (0.0, 1.0)
EFFECTIVENESS_VALUES = tuple(round(x / 10, 1) for x in range(11)) + (-1.0,)
