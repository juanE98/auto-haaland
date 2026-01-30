"""
Unit tests for opponent analysis features module.
"""

import pytest

from lambdas.common.feature_categories.opponent_features import (
    OPPONENT_FEATURES,
    compute_opponent_features,
)


@pytest.mark.unit
class TestOpponentFeaturesList:
    def test_has_24_features(self):
        """OPPONENT_FEATURES should contain 24 features."""
        assert len(OPPONENT_FEATURES) == 24

    def test_no_duplicates(self):
        """All opponent feature names should be unique."""
        assert len(OPPONENT_FEATURES) == len(set(OPPONENT_FEATURES))

    def test_key_features_present(self):
        """Verify key opponent features are in the list."""
        expected = [
            "opp_goals_conceded_last_3",
            "opp_clean_sheets_rate",
            "opp_defensive_rating",
            "opp_goals_scored_last_5",
            "opp_attacking_rating",
            "opp_league_position",
            "opp_form_score",
        ]
        for feat in expected:
            assert feat in OPPONENT_FEATURES, f"Missing: {feat}"


@pytest.mark.unit
class TestComputeOpponentFeatures:
    @pytest.fixture
    def sample_opponent_data(self):
        return {
            "id": 5,
            "position": 12,
            "clean_sheets": 6,
            "played": 20,
            "goals_against": 25,
            "strength_defence_home": 1200,
            "strength_defence_away": 1150,
            "strength_attack_home": 1250,
            "strength_attack_away": 1180,
            "strength_overall_home": 1220,
            "strength_overall_away": 1165,
        }

    @pytest.fixture
    def sample_opponent_fixtures(self):
        return [
            {
                "team_h": 5,
                "team_a": 10,
                "team_h_score": 2,
                "team_a_score": 1,
                "kickoff_time": "2024-01-01T15:00:00Z",
            },
            {
                "team_h": 8,
                "team_a": 5,
                "team_h_score": 0,
                "team_a_score": 0,
                "kickoff_time": "2024-01-08T15:00:00Z",
            },
            {
                "team_h": 5,
                "team_a": 3,
                "team_h_score": 1,
                "team_a_score": 2,
                "kickoff_time": "2024-01-15T15:00:00Z",
            },
            {
                "team_h": 12,
                "team_a": 5,
                "team_h_score": 3,
                "team_a_score": 1,
                "kickoff_time": "2024-01-22T15:00:00Z",
            },
            {
                "team_h": 5,
                "team_a": 15,
                "team_h_score": 2,
                "team_a_score": 0,
                "kickoff_time": "2024-01-29T15:00:00Z",
            },
        ]

    @pytest.fixture
    def sample_opponent_players(self):
        return [
            {
                "id": 100,
                "element_type": 1,
                "saves": 45,
                "minutes": 1800,
                "own_goals": 0,
            },
            {"id": 101, "element_type": 2, "threat": "80.0", "own_goals": 1},
            {"id": 102, "element_type": 3, "threat": "120.0", "own_goals": 0},
            {"id": 103, "element_type": 4, "threat": "180.0", "own_goals": 0},
        ]

    def test_returns_24_features(self):
        result = compute_opponent_features()
        assert len(result) == 24

    def test_all_feature_names_present(self):
        result = compute_opponent_features()
        for name in OPPONENT_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_opp_goals_conceded_last_3(
        self, sample_opponent_data, sample_opponent_fixtures
    ):
        """Should average goals conceded in last 3 fixtures."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
        )
        # Last 3 fixtures: conceded 0 (vs 15), 3 (vs 12), 2 (vs 3) = avg 1.67
        assert result["opp_goals_conceded_last_3"] == pytest.approx(1.67, abs=0.01)

    def test_opp_clean_sheets_rate(self, sample_opponent_data):
        """Should calculate season clean sheet rate."""
        result = compute_opponent_features(opponent_data=sample_opponent_data)
        # 6 clean sheets / 20 games = 0.3
        assert result["opp_clean_sheets_rate"] == pytest.approx(0.3)

    def test_opp_form_score(self, sample_opponent_data, sample_opponent_fixtures):
        """Should calculate points from last 5 matches."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
        )
        # Results: W(3), D(1), L(0), L(0), W(3) = 7
        assert result["opp_form_score"] == 7.0

    def test_opp_league_position(self, sample_opponent_data):
        """Should return opponent league position."""
        result = compute_opponent_features(opponent_data=sample_opponent_data)
        assert result["opp_league_position"] == 12.0

    def test_opp_defensive_errors(
        self, sample_opponent_data, sample_opponent_fixtures, sample_opponent_players
    ):
        """Should count defensive errors (own goals, etc)."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
            opponent_players=sample_opponent_players,
        )
        # 1 own goal in sample players
        assert result["opp_defensive_errors_last_5"] == 1.0

    def test_opp_saves_rate(
        self, sample_opponent_data, sample_opponent_fixtures, sample_opponent_players
    ):
        """Should calculate GK saves rate."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
            opponent_players=sample_opponent_players,
        )
        # GK has 45 saves, team conceded 25 = 70 shots faced
        # saves_rate = 45/70 = 0.643
        assert result["opp_saves_rate"] == pytest.approx(0.643, abs=0.01)

    def test_days_rest_calculation(
        self, sample_opponent_data, sample_opponent_fixtures
    ):
        """Should calculate days since last fixture."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
            current_date="2024-02-01T12:00:00Z",
        )
        # Last fixture was 2024-01-29T15:00:00Z, current is 2024-02-01T12:00:00Z
        # That's 2 days and 21 hours = 2 full days
        assert result["opp_days_rest"] == 2.0

    def test_fixture_congestion(self, sample_opponent_data, sample_opponent_fixtures):
        """Should count games in last 14 days."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
            current_date="2024-02-01T12:00:00Z",
        )
        # Fixtures on 01-22 and 01-29 are within 14 days of 02-01
        assert result["opp_fixture_congestion"] == 2.0

    def test_empty_inputs_return_defaults(self):
        """Should return sensible defaults with no data."""
        result = compute_opponent_features()

        assert result["opp_goals_conceded_last_3"] == 0.0
        assert result["opp_clean_sheets_rate"] == 0.0
        assert result["opp_league_position"] == 10.0
        assert result["opp_days_rest"] == 7.0  # Default
        assert result["opp_fixture_congestion"] == 2.0  # Default

    def test_home_away_goals_split(
        self, sample_opponent_data, sample_opponent_fixtures
    ):
        """Should calculate home/away goals correctly."""
        result = compute_opponent_features(
            opponent_data=sample_opponent_data,
            opponent_fixtures=sample_opponent_fixtures,
        )
        # Home fixtures: conceded 1, 2, 0 (avg 1.0)
        # Away fixtures: conceded 0, 3 (avg 1.5)
        assert result["opp_goals_conceded_home"] == pytest.approx(1.0)
        assert result["opp_goals_conceded_away"] == pytest.approx(1.5)
