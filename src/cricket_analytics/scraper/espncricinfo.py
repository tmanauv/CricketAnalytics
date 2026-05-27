"""Scrape ball-by-ball data from ESPN Cricinfo.

Uses the Cricinfo JSON commentary API to extract structured delivery data.
The primary API endpoint pattern is:
    https://hs-consumer-api.espncricinfo.com/v1/pages/match/comments?seriesId=<id>&matchId=<id>&...

For series listings:
    https://hs-consumer-api.espncricinfo.com/v1/pages/series/schedule?seriesId=<id>
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests

from cricket_analytics.scraper.schemas import BallEvent, MatchData, MatchMeta

logger = logging.getLogger(__name__)

_BASE_URL = "https://hs-consumer-api.espncricinfo.com/v1/pages"
_SCORECARD_URL = "https://www.espncricinfo.com/matches/engine/match/{match_id}.json"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _extract_match_id(url: str) -> str:
    """Extract the numeric match ID from an ESPN Cricinfo URL."""
    match = re.search(r"/(\d{4,8})(?:[/?.]|$)", url)
    if match:
        return match.group(1)
    match = re.search(r"objectId=(\d+)", url)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract match ID from URL: {url}")


def _extract_series_id(url: str) -> str:
    """Extract the numeric series ID from an ESPN Cricinfo URL."""
    match = re.search(r"/(\d{4,8})(?:[/.]|$)", url)
    if match:
        return match.group(1)
    match = re.search(r"-(\d{4,8})$", url)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract series ID from URL: {url}")


def _fetch_json(url: str, params: dict | None = None) -> dict:
    """Fetch JSON from an ESPN Cricinfo API endpoint."""
    resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_dismissal(wicket_info: dict) -> tuple[str | None, str | None]:
    """Parse dismissal kind and dismissed batsman from wicket data."""
    if not wicket_info:
        return None, None
    kind = wicket_info.get("kind", wicket_info.get("dismissalType", ""))
    batsman = ""
    player = wicket_info.get("playerOut", wicket_info.get("player", {}))
    if isinstance(player, dict):
        batsman = player.get("longName", player.get("name", ""))
    elif isinstance(player, str):
        batsman = player
    return kind or None, batsman or None


def scrape_match_scorecard(match_id: str) -> MatchData:
    """Scrape ball-by-ball data from the legacy JSON scorecard endpoint.

    This endpoint returns the full scorecard with innings-level detail.
    """
    url = _SCORECARD_URL.format(match_id=match_id)
    logger.info("Fetching scorecard for match %s", match_id)
    data = _fetch_json(url)

    meta = MatchMeta(match_id=match_id)

    # Parse match info
    match_info = data.get("match", {})
    meta.series = match_info.get("series_name", "")
    meta.match_format = match_info.get("international_class_card", "")
    meta.venue = match_info.get("ground_name", "")
    meta.team1 = match_info.get("team1_name", "")
    meta.team2 = match_info.get("team2_name", "")

    date_str = match_info.get("start_date_raw")
    if date_str:
        try:
            import datetime

            meta.date = datetime.date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            pass

    ball_events: list[BallEvent] = []

    # Parse innings
    for innings_data in data.get("innings", []):
        innings_num = innings_data.get("innings_number", 1)

        for over_data in innings_data.get("overs", []):
            over_num = over_data.get("over", 0)

            for ball_data in over_data.get("balls", []):
                ball_num = ball_data.get("ball", 1)

                batsman_name = ""
                batter = ball_data.get("batsman", ball_data.get("batter", {}))
                if isinstance(batter, dict):
                    batsman_name = batter.get("longName", batter.get("name", ""))
                elif isinstance(batter, str):
                    batsman_name = batter

                bowler_name = ""
                bowler = ball_data.get("bowler", {})
                if isinstance(bowler, dict):
                    bowler_name = bowler.get("longName", bowler.get("name", ""))
                elif isinstance(bowler, str):
                    bowler_name = bowler

                runs_batter = int(ball_data.get("batsmanRuns", ball_data.get("runs", 0)))
                runs_extras = int(ball_data.get("extras", 0))
                runs_total = int(ball_data.get("totalRuns", runs_batter + runs_extras))

                is_wide = bool(ball_data.get("isWide", ball_data.get("wide", False)))
                is_no_ball = bool(ball_data.get("isNoBall", ball_data.get("noBall", False)))
                is_bye = bool(ball_data.get("isBye", ball_data.get("bye", False)))
                is_leg_bye = bool(ball_data.get("isLegBye", ball_data.get("legBye", False)))
                is_wicket = bool(ball_data.get("isWicket", ball_data.get("wicket", False)))

                dismissal_kind, dismissed_batsman = _parse_dismissal(
                    ball_data.get("wicket", ball_data.get("dismissal", {}))
                    if is_wicket
                    else {}
                )

                event = BallEvent(
                    match_id=match_id,
                    innings=innings_num,
                    over=over_num,
                    ball=ball_num,
                    batsman=batsman_name,
                    bowler=bowler_name,
                    runs_batter=runs_batter,
                    runs_extras=runs_extras,
                    runs_total=runs_total,
                    is_wide=is_wide,
                    is_no_ball=is_no_ball,
                    is_bye=is_bye,
                    is_leg_bye=is_leg_bye,
                    is_wicket=is_wicket,
                    dismissal_kind=dismissal_kind,
                    dismissed_batsman=dismissed_batsman,
                    is_four=runs_batter == 4,
                    is_six=runs_batter == 6,
                )
                ball_events.append(event)

    return MatchData(meta=meta, ball_events=ball_events)


def scrape_match_commentary(match_id: str, delay: float = 1.0) -> MatchData:
    """Scrape ball-by-ball data from the commentary API.

    This uses the HS consumer API with paginated commentary data.
    """
    logger.info("Fetching commentary for match %s", match_id)

    # First get match detail for metadata
    detail_url = f"{_BASE_URL}/match/details"
    params = {"matchId": match_id, "latest": "true"}

    try:
        detail = _fetch_json(detail_url, params)
    except requests.HTTPError:
        logger.warning("Commentary API failed for %s, falling back to scorecard", match_id)
        return scrape_match_scorecard(match_id)

    match_info = detail.get("match", {})
    series_info = detail.get("series", {})

    meta = MatchMeta(
        match_id=match_id,
        series=series_info.get("longName", series_info.get("name", "")),
        match_format=match_info.get("format", ""),
        venue=match_info.get("ground", {}).get("longName", ""),
        team1=match_info.get("teams", [{}])[0].get("team", {}).get("longName", "")
        if match_info.get("teams")
        else "",
        team2=match_info.get("teams", [{}])[1].get("team", {}).get("longName", "")
        if len(match_info.get("teams", [])) > 1
        else "",
    )

    date_str = match_info.get("startDate", "")
    if date_str:
        try:
            import datetime

            meta.date = datetime.date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            pass

    # Fetch commentary pages
    ball_events: list[BallEvent] = []
    innings_list = detail.get("scorecard", {}).get("innings", [])

    for innings_idx, innings_info in enumerate(innings_list, start=1):
        innings_id = innings_info.get("inningsId", innings_idx)

        page = 1
        while True:
            comment_url = f"{_BASE_URL}/match/comments"
            comment_params = {
                "matchId": match_id,
                "inningsId": innings_id,
                "page": page,
                "filter": "ball",
                "sortDirection": "asc",
            }

            try:
                comment_data = _fetch_json(comment_url, comment_params)
            except requests.HTTPError:
                break

            comments = comment_data.get("comments", [])
            if not comments:
                break

            for comment in comments:
                over_num = int(comment.get("oversActual", comment.get("overNumber", 0)))
                ball_num = int(comment.get("ballNumber", 1))

                batsman_name = comment.get("batsmanName", "")
                if not batsman_name:
                    batter = comment.get("batsman", {})
                    if isinstance(batter, dict):
                        batsman_name = batter.get("longName", batter.get("name", ""))

                bowler_name = comment.get("bowlerName", "")
                if not bowler_name:
                    bowler = comment.get("bowler", {})
                    if isinstance(bowler, dict):
                        bowler_name = bowler.get("longName", bowler.get("name", ""))

                runs_batter = int(comment.get("batsmanRuns", comment.get("runs", 0)))
                runs_extras = int(comment.get("extras", 0))
                runs_total = int(
                    comment.get("totalRuns", runs_batter + runs_extras)
                )

                is_wide = bool(comment.get("wpiIsWide", comment.get("isWide", False)))
                is_no_ball = bool(comment.get("wpiIsNoBall", comment.get("isNoBall", False)))
                is_wicket = bool(comment.get("isWicket", False))

                dismissal_kind = None
                dismissed_batsman = None
                if is_wicket:
                    dismissal_kind = comment.get(
                        "dismissalType", comment.get("wicketType", None)
                    )
                    dismissed_batsman = comment.get(
                        "dismissedBatsmanName", comment.get("outBatsmanName", None)
                    )

                event = BallEvent(
                    match_id=match_id,
                    innings=innings_idx,
                    over=over_num,
                    ball=ball_num,
                    batsman=batsman_name,
                    bowler=bowler_name,
                    runs_batter=runs_batter,
                    runs_extras=runs_extras,
                    runs_total=runs_total,
                    is_wide=is_wide,
                    is_no_ball=is_no_ball,
                    is_wicket=is_wicket,
                    dismissal_kind=dismissal_kind,
                    dismissed_batsman=dismissed_batsman,
                    is_four=runs_batter == 4,
                    is_six=runs_batter == 6,
                )
                ball_events.append(event)

            pagination = comment_data.get("pagination", {})
            if page >= pagination.get("pageCount", page):
                break
            page += 1
            time.sleep(delay)

    return MatchData(meta=meta, ball_events=ball_events)


def scrape_match(match_url_or_id: str, delay: float = 1.0) -> MatchData:
    """Scrape a single match. Tries commentary API first, falls back to scorecard."""
    match_id = match_url_or_id if match_url_or_id.isdigit() else _extract_match_id(match_url_or_id)

    try:
        result = scrape_match_commentary(match_id, delay=delay)
        if result.ball_events:
            return result
    except Exception:
        logger.warning("Commentary scrape failed for %s, trying scorecard", match_id)

    return scrape_match_scorecard(match_id)


def get_series_match_ids(series_url_or_id: str) -> list[str]:
    """Get all match IDs from a series page."""
    series_id = (
        series_url_or_id if series_url_or_id.isdigit() else _extract_series_id(series_url_or_id)
    )

    url = f"{_BASE_URL}/series/schedule"
    params = {"seriesId": series_id}

    data = _fetch_json(url, params)
    match_ids: list[str] = []

    for match_group in data.get("content", {}).get("matchGroups", []):
        for match in match_group.get("matches", []):
            mid = str(match.get("objectId", match.get("matchId", "")))
            if mid:
                match_ids.append(mid)

    return match_ids


def scrape_series(
    series_url_or_id: str, delay: float = 2.0
) -> list[MatchData]:
    """Scrape all matches in a series."""
    match_ids = get_series_match_ids(series_url_or_id)
    logger.info("Found %d matches in series", len(match_ids))

    results: list[MatchData] = []
    for i, mid in enumerate(match_ids, 1):
        logger.info("Scraping match %d/%d (ID: %s)", i, len(match_ids), mid)
        try:
            result = scrape_match(mid, delay=delay)
            results.append(result)
        except Exception:
            logger.exception("Failed to scrape match %s", mid)
        time.sleep(delay)

    return results


def save_raw(matches: list[MatchData], output_dir: str | Path) -> list[Path]:
    """Save scraped match data as NDJSON files in the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for match in matches:
        filename = f"match_{match.meta.match_id}.ndjson"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            # First line: match metadata
            f.write(match.meta.model_dump_json() + "\n")
            # Subsequent lines: ball events
            for event in match.ball_events:
                f.write(event.model_dump_json() + "\n")

        saved.append(filepath)
        logger.info("Saved %d events to %s", len(match.ball_events), filepath)

    return saved


def load_raw(filepath: str | Path) -> MatchData:
    """Load a match from an NDJSON file previously saved by save_raw."""
    path = Path(filepath)
    with open(path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {path}")

    meta = MatchMeta.model_validate_json(lines[0])
    events = [BallEvent.model_validate_json(line) for line in lines[1:] if line.strip()]

    return MatchData(meta=meta, ball_events=events)
