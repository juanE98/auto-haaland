"""
FPL API client with rate limiting and error handling.

The official FPL API is free and requires no authentication.
Rate limits are generous, but we implement exponential backoff to be respectful.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class FPLApiError(Exception):
    """Base exception for FPL API errors."""

    pass


class FPLApiClient:
    """
    Client for the Fantasy Premier League API.

    Official API base: https://fantasy.premierleague.com/api/
    No authentication required, free to use.
    """

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(
        self, timeout: int = 30, max_retries: int = 3, base_delay: float = 1.0
    ):
        """
        Initialise FPL API client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = httpx.Client(timeout=timeout)

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry logic.

        Args:
            endpoint: API endpoint (will be appended to BASE_URL)
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            FPLApiError: If request fails after retries
        """
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching {url} (attempt {attempt + 1}/{self.max_retries})"
                )

                response = self.client.get(url, params=params)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", self.base_delay)
                    )
                    logger.warning(
                        f"Rate limited. Waiting {retry_after}s before retry..."
                    )
                    time.sleep(retry_after)
                    continue

                # Raise for other HTTP errors
                response.raise_for_status()

                logger.info(f"Successfully fetched {url}")
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
                if attempt == self.max_retries - 1:
                    raise FPLApiError(f"Failed to fetch {url}: {e}")

                # Exponential backoff
                delay = self.base_delay * (2**attempt)
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                if attempt == self.max_retries - 1:
                    raise FPLApiError(f"Failed to fetch {url}: {e}")

                delay = self.base_delay * (2**attempt)
                time.sleep(delay)

        raise FPLApiError(f"Failed to fetch {url} after {self.max_retries} attempts")

    def get_bootstrap_static(self) -> Dict[str, Any]:
        """
        Get bootstrap-static data (current season overview).

        This is the main endpoint containing:
        - All events (gameweeks)
        - All teams
        - All players (elements)
        - Game settings

        Returns:
            Dictionary with keys: events, teams, elements, etc.
        """
        return self._make_request("bootstrap-static/")

    def get_fixtures(self, gameweek: Optional[int] = None) -> list:
        """
        Get fixtures data.

        Args:
            gameweek: Specific gameweek number (None for all)

        Returns:
            List of fixture dictionaries
        """
        params = {"event": gameweek} if gameweek else None
        return self._make_request("fixtures/", params=params)

    def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get detailed summary for a specific player.

        Includes:
        - Player history for current season
        - Fixtures
        - Past seasons summary

        Args:
            player_id: Player ID from bootstrap-static

        Returns:
            Dictionary with history and fixtures
        """
        return self._make_request(f"element-summary/{player_id}/")

    def get_manager_history(self, team_id: int) -> Dict[str, Any]:
        """
        Get history for a specific FPL manager/team.

        Args:
            team_id: Manager's team ID

        Returns:
            Dictionary with manager's history
        """
        return self._make_request(f"entry/{team_id}/history/")

    def get_current_gameweek(self) -> Optional[int]:
        """
        Get the current active gameweek number.

        Returns:
            Current gameweek number, or None if season not started
        """
        bootstrap = self.get_bootstrap_static()
        events = bootstrap.get("events", [])

        for event in events:
            if event.get("is_current"):
                return event["id"]

        return None

    def is_gameweek_finished(self, gameweek: int) -> bool:
        """
        Check if a specific gameweek has finished.

        Args:
            gameweek: Gameweek number to check

        Returns:
            True if gameweek is finished, False otherwise
        """
        bootstrap = self.get_bootstrap_static()
        events = bootstrap.get("events", [])

        for event in events:
            if event["id"] == gameweek:
                return event.get("finished", False)

        return False

    def get_season_string(self) -> str:
        """
        Get current season string in format YYYY_YY.

        Example: "2024_25" for 2024/25 season

        Returns:
            Season string
        """
        now = datetime.now()
        # Season starts in August
        if now.month >= 8:
            return f"{now.year}_{str(now.year + 1)[-2:]}"
        else:
            return f"{now.year - 1}_{str(now.year)[-2:]}"

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
