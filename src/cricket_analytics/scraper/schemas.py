"""Pydantic models for ball-by-ball cricket data."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class BallEvent(BaseModel):
    """A single ball delivery in a cricket match."""

    match_id: str
    innings: int = Field(ge=1, le=4)
    over: int = Field(ge=0)
    ball: int = Field(ge=1)
    batsman: str
    bowler: str
    ball_type: str = ""
    shot_type: str = ""
    runs_batter: int = Field(ge=0, default=0)
    runs_extras: int = Field(ge=0, default=0)
    runs_total: int = Field(ge=0, default=0)
    is_wide: bool = False
    is_no_ball: bool = False
    is_bye: bool = False
    is_leg_bye: bool = False
    is_wicket: bool = False
    dismissal_kind: str | None = None
    dismissed_batsman: str | None = None
    is_four: bool = False
    is_six: bool = False


class MatchMeta(BaseModel):
    """Match-level metadata."""

    match_id: str
    series: str = ""
    match_format: str = ""
    venue: str = ""
    date: datetime.date | None = None
    team1: str = ""
    team2: str = ""
    winner: str = ""
    toss_winner: str = ""
    toss_decision: str = ""


class MatchData(BaseModel):
    """Complete data for a single match."""

    meta: MatchMeta
    ball_events: list[BallEvent] = Field(default_factory=list)
