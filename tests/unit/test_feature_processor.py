"""
Unit tests for feature_processor Lambda handler.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from lambdas.feature_processor.handler import (
    handler,
    calculate_rolling_average,
    calculate_minutes_pct,
    get_team_strength,
    get_opponent_info,
    engineer_features,
)


class TestCalculateRollingAverage:
    """Tests for calculate_rolling_average function."""

    def test_rolling_average_3_games(self):
        """Test rolling average with exactly 3 games."""
        points = [5, 8, 3]
        result = calculate_rolling_average(points, window=3)
        assert result == pytest.approx(5.33, rel=0.01)

    def test_rolling_average_5_games(self):
        """Test rolling average with exactly 5 games."""
        points = [5, 8, 3, 10, 4]
        result = calculate_rolling_average(points, window=5)
        assert result == pytest.approx(6.0, rel=0.01)

    def test_rolling_average_more_than_window(self):
        """Test rolling average takes only last N games."""
        points = [2, 2, 2, 10, 10, 10]  # Last 3: [10, 10, 10]
        result = calculate_rolling_average(points, window=3)
        assert result == pytest.approx(10.0, rel=0.01)

    def test_rolling_average_insufficient_data(self):
        """Test rolling average with fewer games than window."""
        points = [5, 8]  # Only 2 games, window is 3
        result = calculate_rolling_average(points, window=3)
        # Should average available data: (5 + 8) / 2 = 6.5
        assert result == pytest.approx(6.5, rel=0.01)

    def test_rolling_average_empty_list(self):
        """Test rolling average with empty list."""
        result = calculate_rolling_average([], window=3)
        assert result == 0.0

    def test_rolling_average_single_game(self):
        """Test rolling average with single game."""
        result = calculate_rolling_average([12], window=5)
        assert result == 12.0


class TestCalculateMinutesPct:
    """Tests for calculate_minutes_pct function."""

    def test_full_minutes(self):
        """Test player who played all 90 minutes."""
        history = [
            {"minutes": 90},
            {"minutes": 90},
            {"minutes": 90}
        ]
        result = calculate_minutes_pct(history, window=3)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_partial_minutes(self):
        """Test player with partial minutes."""
        history = [
            {"minutes": 45},
            {"minutes": 90},
            {"minutes": 45}
        ]
        # Total: 180, Max: 270 -> 180/270 = 0.667
        result = calculate_minutes_pct(history, window=3)
        assert result == pytest.approx(0.667, rel=0.01)

    def test_no_minutes(self):
        """Test player who didn't play."""
        history = [
            {"minutes": 0},
            {"minutes": 0}
        ]
        result = calculate_minutes_pct(history, window=5)
        assert result == 0.0

    def test_empty_history(self):
        """Test with empty history."""
        result = calculate_minutes_pct([], window=5)
        assert result == 0.0

    def test_window_larger_than_history(self):
        """Test when window is larger than available history."""
        history = [{"minutes": 90}, {"minutes": 45}]
        # Total: 135, Max: 180 -> 135/180 = 0.75
        result = calculate_minutes_pct(history, window=5)
        assert result == pytest.approx(0.75, rel=0.01)


class TestGetTeamStrength:
    """Tests for get_team_strength function."""

    def test_get_existing_team_strength(self):
        """Test getting strength for existing team."""
        teams = [
            {"id": 1, "name": "Arsenal", "strength": 4},
            {"id": 10, "name": "Liverpool", "strength": 5}
        ]
        result = get_team_strength(teams, team_id=10)
        assert result == 5

    def test_get_team_strength_not_found(self):
        """Test default strength when team not found."""
        teams = [{"id": 1, "name": "Arsenal", "strength": 4}]
        result = get_team_strength(teams, team_id=99)
        assert result == 3  # Default

    def test_get_team_strength_missing_field(self):
        """Test default when strength field missing."""
        teams = [{"id": 1, "name": "Arsenal"}]  # No strength field
        result = get_team_strength(teams, team_id=1)
        assert result == 3  # Default


class TestGetOpponentInfo:
    """Tests for get_opponent_info function."""

    def test_home_game(self):
        """Test player's team playing at home."""
        fixtures = [
            {"team_h": 10, "team_a": 1}  # Liverpool (10) vs Arsenal (1) at home
        ]
        teams = [
            {"id": 1, "name": "Arsenal", "strength": 4},
            {"id": 10, "name": "Liverpool", "strength": 5}
        ]

        opponent_strength, is_home = get_opponent_info(
            player_team_id=10, fixtures=fixtures, teams=teams
        )

        assert opponent_strength == 4  # Arsenal's strength
        assert is_home == 1

    def test_away_game(self):
        """Test player's team playing away."""
        fixtures = [
            {"team_h": 1, "team_a": 10}  # Arsenal (1) vs Liverpool (10) at Arsenal
        ]
        teams = [
            {"id": 1, "name": "Arsenal", "strength": 4},
            {"id": 10, "name": "Liverpool", "strength": 5}
        ]

        opponent_strength, is_home = get_opponent_info(
            player_team_id=10, fixtures=fixtures, teams=teams
        )

        assert opponent_strength == 4  # Arsenal's strength
        assert is_home == 0

    def test_no_fixture_found(self):
        """Test when team has no fixture (blank gameweek)."""
        fixtures = [
            {"team_h": 1, "team_a": 2}  # Fixture doesn't include team 10
        ]
        teams = [
            {"id": 1, "strength": 4},
            {"id": 2, "strength": 3}
        ]

        opponent_strength, is_home = get_opponent_info(
            player_team_id=10, fixtures=fixtures, teams=teams
        )

        assert opponent_strength == 3  # Default
        assert is_home == 0


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    @pytest.fixture
    def sample_bootstrap(self):
        """Sample bootstrap data."""
        return {
            "elements": [
                {
                    "id": 350,
                    "web_name": "Salah",
                    "team": 10,
                    "element_type": 3,
                    "form": "8.5",
                    "chance_of_playing_next_round": 100,
                    "minutes": 900
                },
                {
                    "id": 328,
                    "web_name": "Haaland",
                    "team": 11,
                    "element_type": 4,
                    "form": "7.2",
                    "chance_of_playing_next_round": 75,
                    "minutes": 810
                }
            ],
            "teams": [
                {"id": 1, "name": "Arsenal", "strength": 4},
                {"id": 10, "name": "Liverpool", "strength": 5},
                {"id": 11, "name": "Man City", "strength": 5}
            ]
        }

    @pytest.fixture
    def sample_fixtures(self):
        """Sample fixtures data."""
        return [
            {"team_h": 10, "team_a": 1},   # Liverpool vs Arsenal (home)
            {"team_h": 2, "team_a": 11}    # Someone vs Man City (away)
        ]

    @pytest.fixture
    def sample_histories(self):
        """Sample player histories."""
        return {
            350: [
                {"total_points": 8, "minutes": 90, "round": 18},
                {"total_points": 12, "minutes": 90, "round": 19},
                {"total_points": 6, "minutes": 90, "round": 20}
            ]
        }

    def test_engineer_features_historical_mode(
        self, sample_bootstrap, sample_fixtures, sample_histories
    ):
        """Test feature engineering in historical mode."""
        df = engineer_features(
            bootstrap=sample_bootstrap,
            fixtures=sample_fixtures,
            player_histories=sample_histories,
            mode="historical",
            gameweek=20
        )

        assert len(df) == 2  # 2 players

        # Check Salah's features (has history)
        salah = df[df["player_id"] == 350].iloc[0]
        assert salah["player_name"] == "Salah"
        assert salah["gameweek"] == 20
        assert salah["points_last_3"] == pytest.approx(8.67, rel=0.01)  # (8+12+6)/3
        assert salah["home_away"] == 1  # Home game
        assert salah["opponent_strength"] == 4  # Arsenal
        assert "actual_points" in df.columns  # Historical mode includes target

    def test_engineer_features_prediction_mode(
        self, sample_bootstrap, sample_fixtures, sample_histories
    ):
        """Test feature engineering in prediction mode."""
        df = engineer_features(
            bootstrap=sample_bootstrap,
            fixtures=sample_fixtures,
            player_histories=sample_histories,
            mode="prediction",
            gameweek=20
        )

        assert len(df) == 2
        assert "actual_points" not in df.columns  # Prediction mode excludes target

    def test_engineer_features_fallback_no_history(
        self, sample_bootstrap, sample_fixtures
    ):
        """Test fallback to bootstrap when no history available."""
        df = engineer_features(
            bootstrap=sample_bootstrap,
            fixtures=sample_fixtures,
            player_histories={},  # Empty histories
            mode="prediction",
            gameweek=10
        )

        # Should use form field as fallback
        salah = df[df["player_id"] == 350].iloc[0]
        assert salah["points_last_3"] == 8.5  # Falls back to form
        assert salah["points_last_5"] == 8.5  # Falls back to form
        assert salah["form_score"] == 8.5

    def test_engineer_features_form_x_difficulty(
        self, sample_bootstrap, sample_fixtures
    ):
        """Test interaction feature calculation."""
        df = engineer_features(
            bootstrap=sample_bootstrap,
            fixtures=sample_fixtures,
            player_histories={},
            mode="prediction",
            gameweek=10
        )

        salah = df[df["player_id"] == 350].iloc[0]
        # form_x_difficulty = form_score * opponent_strength
        # 8.5 * 4 = 34.0
        assert salah["form_x_difficulty"] == pytest.approx(34.0, rel=0.01)


class TestFeatureProcessorHandler:
    """Tests for feature processor Lambda handler."""

    @patch('lambdas.feature_processor.handler.get_s3_client')
    @patch('lambdas.feature_processor.handler.load_bootstrap_from_s3')
    @patch('lambdas.feature_processor.handler.load_fixtures_from_s3')
    @patch('lambdas.feature_processor.handler.load_player_histories_from_s3')
    @patch('lambdas.feature_processor.handler.save_features_to_s3')
    def test_handler_returns_correct_structure(
        self,
        mock_save,
        mock_load_histories,
        mock_load_fixtures,
        mock_load_bootstrap,
        mock_get_s3
    ):
        """Test handler returns correct response structure."""
        # Setup mocks
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        mock_load_bootstrap.return_value = {
            "elements": [
                {"id": 1, "web_name": "Player", "team": 1, "element_type": 3,
                 "form": "5.0", "chance_of_playing_next_round": 100, "minutes": 900}
            ],
            "teams": [{"id": 1, "strength": 3}]
        }
        mock_load_fixtures.return_value = [{"team_h": 1, "team_a": 2}]
        mock_load_histories.return_value = {}
        mock_save.return_value = "processed/season_2024_25/gw20_features_training.parquet"

        event = {
            "gameweek": 20,
            "season": "2024_25",
            "mode": "historical"
        }

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["season"] == "2024_25"
        assert result["mode"] == "historical"
        assert "features_file" in result
        assert "rows_processed" in result
        assert "timestamp" in result

    def test_handler_missing_gameweek(self):
        """Test handler returns error when gameweek missing."""
        event = {"season": "2024_25"}
        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "Missing required field: gameweek" in result["error"]

    def test_handler_missing_season(self):
        """Test handler returns error when season missing."""
        event = {"gameweek": 20}
        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "Missing required field: season" in result["error"]

    def test_handler_invalid_mode(self):
        """Test handler returns error for invalid mode."""
        event = {
            "gameweek": 20,
            "season": "2024_25",
            "mode": "invalid_mode"
        }
        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "Invalid mode" in result["error"]

    @patch('lambdas.feature_processor.handler.get_s3_client')
    @patch('lambdas.feature_processor.handler.load_bootstrap_from_s3')
    def test_handler_s3_not_found_error(self, mock_load_bootstrap, mock_get_s3):
        """Test handler handles missing S3 data gracefully."""
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Simulate S3 NoSuchKey error
        from botocore.exceptions import ClientError
        mock_load_bootstrap.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "GetObject"
        )

        event = {
            "gameweek": 20,
            "season": "2024_25"
        }

        result = handler(event, None)

        assert result["statusCode"] == 404
        assert "error" in result
        assert "not found" in result["error"].lower()
