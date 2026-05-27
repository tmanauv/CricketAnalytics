"""Tests for the scraper module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cricket_analytics.scraper.espncricinfo import (
    _extract_match_id,
    _extract_series_id,
    load_raw,
    save_raw,
)
from cricket_analytics.scraper.schemas import BallEvent, MatchData, MatchMeta


class TestExtractMatchId:
    def test_numeric_url(self):
        url = "https://www.espncricinfo.com/series/something/match/1234567"
        assert _extract_match_id(url) == "1234567"

    def test_url_with_query(self):
        url = "https://www.espncricinfo.com/matches/engine/match/1234567.html"
        assert _extract_match_id(url) == "1234567"

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Cannot extract match ID"):
            _extract_match_id("https://www.google.com")

    def test_object_id_param(self):
        url = "https://example.com/page?objectId=1234567&other=true"
        assert _extract_match_id(url) == "1234567"


class TestExtractSeriesId:
    def test_series_url(self):
        url = "https://www.espncricinfo.com/series/icc-world-cup-2023-1367856"
        assert _extract_series_id(url) == "1367856"

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Cannot extract series ID"):
            _extract_series_id("https://www.google.com/no-numbers")


class TestBallEvent:
    def test_create_basic(self):
        event = BallEvent(
            match_id="123",
            innings=1,
            over=5,
            ball=3,
            batsman="Virat Kohli",
            bowler="Pat Cummins",
            runs_batter=4,
            runs_total=4,
            is_four=True,
        )
        assert event.match_id == "123"
        assert event.batsman == "Virat Kohli"
        assert event.runs_batter == 4
        assert event.is_four is True
        assert event.is_wicket is False

    def test_wicket_event(self):
        event = BallEvent(
            match_id="123",
            innings=1,
            over=10,
            ball=1,
            batsman="Rohit Sharma",
            bowler="Mitchell Starc",
            runs_batter=0,
            is_wicket=True,
            dismissal_kind="caught",
            dismissed_batsman="Rohit Sharma",
        )
        assert event.is_wicket is True
        assert event.dismissal_kind == "caught"

    def test_serialization_roundtrip(self):
        event = BallEvent(
            match_id="123",
            innings=1,
            over=1,
            ball=1,
            batsman="Test",
            bowler="Test",
        )
        json_str = event.model_dump_json()
        restored = BallEvent.model_validate_json(json_str)
        assert event == restored


class TestMatchMeta:
    def test_create_minimal(self):
        meta = MatchMeta(match_id="123")
        assert meta.match_id == "123"
        assert meta.series == ""

    def test_full_meta(self):
        import datetime

        meta = MatchMeta(
            match_id="1234567",
            series="ICC World Cup 2023",
            match_format="ODI",
            venue="Wankhede Stadium, Mumbai",
            date=datetime.date(2023, 11, 19),
            team1="India",
            team2="Australia",
            winner="Australia",
        )
        assert meta.team1 == "India"
        assert meta.date.year == 2023


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path: Path):
        meta = MatchMeta(match_id="test123", series="Test Series")
        events = [
            BallEvent(
                match_id="test123",
                innings=1,
                over=0,
                ball=1,
                batsman="Batter A",
                bowler="Bowler B",
                runs_batter=4,
                runs_total=4,
                is_four=True,
            ),
            BallEvent(
                match_id="test123",
                innings=1,
                over=0,
                ball=2,
                batsman="Batter A",
                bowler="Bowler B",
                runs_batter=0,
                runs_total=0,
                is_wicket=True,
                dismissal_kind="bowled",
                dismissed_batsman="Batter A",
            ),
        ]
        match = MatchData(meta=meta, ball_events=events)

        saved = save_raw([match], tmp_path)
        assert len(saved) == 1
        assert saved[0].exists()

        loaded = load_raw(saved[0])
        assert loaded.meta.match_id == "test123"
        assert loaded.meta.series == "Test Series"
        assert len(loaded.ball_events) == 2
        assert loaded.ball_events[0].batsman == "Batter A"
        assert loaded.ball_events[0].is_four is True
        assert loaded.ball_events[1].is_wicket is True
        assert loaded.ball_events[1].dismissal_kind == "bowled"

    def test_empty_file_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.ndjson"
        empty.write_text("")
        with pytest.raises(ValueError, match="Empty file"):
            load_raw(empty)
