"""
Unit tests for team context features module.
"""

import pytest

from lambdas.common.feature_categories.team_features import (
    TEAM_FEATURES,
    compute_team_features,
)


@pytest.mark.unit
class TestTeamFeaturesList:
    def test_has_28_features(self):
        """TEAM_FEATURES should contain 28 features."""
        assert len(TEAM_FEATURES) == 28

    def test_no_duplicates(self):
        """All team feature names should be unique."""
        assert len(TEAM_FEATURES) == len(set(TEAM_FEATURES))

    def test_key_features_present(self):
        """Verify key team features are in the list."""
        expected = [
            "team_goals_scored_last_3",
            "team_clean_sheets_last_5",
            "team_form_score",
            "team_strength_overall",
            "team_league_position",
            "player_share_of_team_points",
            "squad_depth_at_position",
        ]
        for feat in expected:
            assert feat in TEAM_FEATURES, f"Missing: {feat}"


@pytest.mark.unit
class TestComputeTeamFeatures:
    @pytest.fixture
    def sample_player(self):
        return {
            "id": 350,
            "team": 10,
            "element_type": 3,
            "total_points": 100,
            "goals_scored": 10,
            "ict_index": "150.5",
            "status": "a",
        }

    @pytest.fixture
    def sample_team_data(self):
        return {
            "id": 10,
            "strength_overall_home": 1300,
            "strength_overall_away": 1250,
            "strength_attack_home": 1320,
            "strength_attack_away": 1280,
            "strength_defence_home": 1310,
            "strength_defence_away": 1260,
            "position": 5,
            "points": 45,
            "goal_difference": 15,
        }

    @pytest.fixture
    def sample_team_players(self, sample_player):
        return [
            sample_player,
            {
                "id": 351,
                "team": 10,
                "element_type": 3,
                "total_points": 80,
                "goals_scored": 8,
                "ict_index": "120.0",
                "status": "a",
            },
            {
                "id": 352,
                "team": 10,
                "element_type": 4,
                "total_points": 120,
                "goals_scored": 15,
                "ict_index": "180.0",
                "status": "a",
            },
            {
                "id": 353,
                "team": 10,
                "element_type": 2,
                "total_points": 60,
                "goals_scored": 3,
                "ict_index": "80.0",
                "status": "i",
            },
        ]

    @pytest.fixture
    def sample_fixtures(self):
        return [
            {"team_h": 10, "team_a": 5, "team_h_score": 2, "team_a_score": 1},
            {"team_h": 3, "team_a": 10, "team_h_score": 1, "team_a_score": 1},
            {"team_h": 10, "team_a": 8, "team_h_score": 3, "team_a_score": 0},
            {"team_h": 12, "team_a": 10, "team_h_score": 0, "team_a_score": 2},
            {"team_h": 10, "team_a": 15, "team_h_score": 1, "team_a_score": 2},
        ]

    def test_returns_28_features(self, sample_player):
        result = compute_team_features(sample_player)
        assert len(result) == 28

    def test_all_feature_names_present(self, sample_player):
        result = compute_team_features(sample_player)
        for name in TEAM_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_team_form_score_calculation(
        self, sample_player, sample_team_data, sample_fixtures
    ):
        """Form score should be points from last 5 matches."""
        result = compute_team_features(
            sample_player, team_data=sample_team_data, team_fixtures=sample_fixtures
        )
        # Results: W(3) + D(1) + W(3) + W(3) + L(0) = 10
        assert result["team_form_score"] == 10.0

    def test_team_goals_scored_last_3(
        self, sample_player, sample_team_data, sample_fixtures
    ):
        """Should average goals from last 3 fixtures."""
        result = compute_team_features(
            sample_player, team_data=sample_team_data, team_fixtures=sample_fixtures
        )
        # Last 3: 3, 2, 1 = avg 2.0
        assert result["team_goals_scored_last_3"] == pytest.approx(2.0)

    def test_team_clean_sheets_last_3(
        self, sample_player, sample_team_data, sample_fixtures
    ):
        """Should count clean sheets in last 3 fixtures."""
        result = compute_team_features(
            sample_player, team_data=sample_team_data, team_fixtures=sample_fixtures
        )
        # Last 3: conceded 2, 0, 0 = 2 clean sheets / 3 = 0.67
        assert result["team_clean_sheets_last_3"] == pytest.approx(0.67, abs=0.01)

    def test_team_strength_overall(self, sample_player, sample_team_data):
        """Should average home and away overall strength."""
        result = compute_team_features(sample_player, team_data=sample_team_data)
        # (1300 + 1250) / 2 = 1275
        assert result["team_strength_overall"] == 1275.0

    def test_player_share_of_team_points(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should calculate player's share of team total points."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player: 100, Team total: 100+80+120+60=360
        # Share: 100/360 = 0.2778
        assert result["player_share_of_team_points"] == pytest.approx(0.2778, abs=0.001)

    def test_squad_depth_at_position(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should count players at same position."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player is element_type 3 (MID), 2 MIDs in team
        assert result["squad_depth_at_position"] == 2.0

    def test_team_players_available(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should count players with status 'a'."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # 3 out of 4 have status 'a'
        assert result["team_players_available"] == 3.0

    def test_empty_inputs_return_defaults(self, sample_player):
        """Should return sensible defaults with no team data."""
        result = compute_team_features(sample_player)

        assert result["team_form_score"] == 0.0
        assert result["team_strength_overall"] == 0.0
        assert result["team_league_position"] == 10.0  # Default middle position
