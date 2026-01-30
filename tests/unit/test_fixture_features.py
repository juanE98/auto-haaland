"""
Unit tests for fixture context features module.
"""

import pytest

from lambdas.common.feature_categories.fixture_features import (
    FIXTURE_FEATURES,
    compute_fixture_features,
)


@pytest.mark.unit
class TestFixtureFeaturesList:
    def test_has_16_features(self):
        """FIXTURE_FEATURES should contain 16 features."""
        assert len(FIXTURE_FEATURES) == 16

    def test_no_duplicates(self):
        """All fixture feature names should be unique."""
        assert len(FIXTURE_FEATURES) == len(set(FIXTURE_FEATURES))

    def test_key_features_present(self):
        """Verify key fixture features are in the list."""
        expected = [
            "fdr_current",
            "fdr_next_3_avg",
            "is_double_gameweek",
            "is_blank_gameweek",
            "days_since_last_game",
            "kickoff_hour",
            "is_weekend_game",
        ]
        for feat in expected:
            assert feat in FIXTURE_FEATURES, f"Missing: {feat}"


@pytest.mark.unit
class TestComputeFixtureFeatures:
    @pytest.fixture
    def sample_player(self):
        return {"id": 350, "team": 10, "element_type": 3}

    @pytest.fixture
    def sample_current_fixture(self):
        return {
            "team_h": 10,
            "team_a": 5,
            "team_h_difficulty": 2,
            "team_a_difficulty": 4,
            "kickoff_time": "2024-02-03T15:00:00Z",  # Saturday 3pm
        }

    @pytest.fixture
    def sample_upcoming_fixtures(self):
        return [
            {"team_h": 3, "team_a": 10, "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"team_h": 10, "team_a": 8, "team_h_difficulty": 3, "team_a_difficulty": 3},
            {
                "team_h": 12,
                "team_a": 10,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            },
            {"team_h": 10, "team_a": 1, "team_h_difficulty": 5, "team_a_difficulty": 2},
            {"team_h": 6, "team_a": 10, "team_h_difficulty": 3, "team_a_difficulty": 4},
        ]

    @pytest.fixture
    def sample_past_fixtures(self):
        return [
            {
                "team_h": 10,
                "team_a": 15,
                "team_h_difficulty": 2,
                "team_a_difficulty": 3,
                "kickoff_time": "2024-01-20T15:00:00Z",
            },
            {
                "team_h": 7,
                "team_a": 10,
                "team_h_difficulty": 3,
                "team_a_difficulty": 4,
                "kickoff_time": "2024-01-27T17:30:00Z",
            },
        ]

    def test_returns_16_features(self):
        result = compute_fixture_features()
        assert len(result) == 16

    def test_all_feature_names_present(self):
        result = compute_fixture_features()
        for name in FIXTURE_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_fdr_current_home_game(self, sample_player, sample_current_fixture):
        """Home team should get team_h_difficulty."""
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=sample_current_fixture,
        )
        # Player team 10 is home, so FDR = team_h_difficulty = 2
        assert result["fdr_current"] == 2.0

    def test_fdr_current_away_game(self, sample_player):
        """Away team should get team_a_difficulty."""
        away_fixture = {
            "team_h": 5,
            "team_a": 10,
            "team_h_difficulty": 2,
            "team_a_difficulty": 4,
        }
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=away_fixture,
        )
        # Player team 10 is away, so FDR = team_a_difficulty = 4
        assert result["fdr_current"] == 4.0

    def test_fdr_next_3_avg(self, sample_player, sample_upcoming_fixtures):
        """Should average FDR of next 3 fixtures."""
        result = compute_fixture_features(
            player=sample_player,
            upcoming_fixtures=sample_upcoming_fixtures,
        )
        # Team 10 is away in first (4), home in second (3), away in third (4)
        # Average = (4 + 3 + 4) / 3 = 3.67
        assert result["fdr_next_3_avg"] == pytest.approx(3.67, abs=0.01)

    def test_is_tough_fixture(self, sample_player):
        """Should flag FDR >= 4 as tough."""
        tough_fixture = {"team_h": 10, "team_a": 5, "team_h_difficulty": 4}
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=tough_fixture,
        )
        assert result["is_tough_fixture"] == 1.0

    def test_is_easy_fixture(self, sample_player, sample_current_fixture):
        """Should flag FDR <= 2 as easy."""
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=sample_current_fixture,
        )
        # FDR = 2, which is <= 2
        assert result["is_easy_fixture"] == 1.0

    def test_is_double_gameweek(self, sample_player):
        """Should detect when player has 2+ fixtures."""
        result = compute_fixture_features(
            player=sample_player,
            current_gw_fixtures=[{}, {}],  # 2 fixtures
        )
        assert result["is_double_gameweek"] == 1.0
        assert result["dgw_fixture_count"] == 2.0

    def test_is_blank_gameweek(self, sample_player):
        """Should detect when player has 0 fixtures."""
        result = compute_fixture_features(
            player=sample_player,
            current_gw_fixtures=[],  # 0 fixtures
        )
        assert result["is_blank_gameweek"] == 1.0
        assert result["dgw_fixture_count"] == 0.0

    def test_days_since_last_game(self, sample_player, sample_past_fixtures):
        """Should calculate days since last fixture."""
        result = compute_fixture_features(
            player=sample_player,
            past_fixtures=sample_past_fixtures,
            current_date="2024-02-01T12:00:00Z",
        )
        # Last fixture was 2024-01-27T17:30:00Z, current is 2024-02-01T12:00:00Z
        # That's 4 days and 18.5 hours = 4 full days
        assert result["days_since_last_game"] == 4.0

    def test_kickoff_hour(self, sample_player, sample_current_fixture):
        """Should extract kickoff hour."""
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=sample_current_fixture,
        )
        # Kickoff at 15:00
        assert result["kickoff_hour"] == 15.0

    def test_is_weekend_game(self, sample_player, sample_current_fixture):
        """Should detect weekend games."""
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=sample_current_fixture,
        )
        # 2024-02-03 is a Saturday
        assert result["is_weekend_game"] == 1.0

    def test_is_evening_kickoff(self, sample_player):
        """Should detect evening kickoffs (17:00+)."""
        evening_fixture = {
            "team_h": 10,
            "team_a": 5,
            "kickoff_time": "2024-02-03T20:00:00Z",
        }
        result = compute_fixture_features(
            player=sample_player,
            current_fixture=evening_fixture,
        )
        assert result["is_evening_kickoff"] == 1.0

    def test_fixture_congestion(self, sample_player, sample_past_fixtures):
        """Should count games in recent days."""
        result = compute_fixture_features(
            player=sample_player,
            past_fixtures=sample_past_fixtures,
            current_date="2024-02-01T12:00:00Z",
        )
        # Fixtures on 01-20 and 01-27
        # 7 days: only 01-27 is within 7 days of 02-01
        # 14 days: both are within 14 days
        assert result["fixture_congestion_7d"] == 1.0
        assert result["fixture_congestion_14d"] == 2.0

    def test_empty_inputs_return_defaults(self):
        """Should return sensible defaults with no data."""
        result = compute_fixture_features()

        assert result["fdr_current"] == 3.0  # Default medium difficulty
        assert result["fdr_next_3_avg"] == 3.0
        assert result["is_double_gameweek"] == 0.0
        assert result["days_since_last_game"] == 7.0  # Default
        assert result["kickoff_hour"] == 15.0  # Default 3pm
        assert result["is_weekend_game"] == 1.0  # Default Saturday
