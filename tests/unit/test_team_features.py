"""
Unit tests for team context features module.
"""

import pytest

from lambdas.common.feature_categories.team_features import (
    TEAM_FEATURES,
    _count_games_at_current_team,
    compute_team_features,
)


@pytest.mark.unit
class TestTeamFeaturesList:
    def test_has_35_features(self):
        """TEAM_FEATURES should contain 35 features."""
        assert len(TEAM_FEATURES) == 35

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
            "player_rank_in_team_points",
            "player_rank_in_team_ict",
            "player_share_of_team_assists",
            "player_share_of_team_xgi",
            "player_minutes_share",
            "player_points_vs_position_avg",
            "games_at_current_team",
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
            "assists": 5,
            "minutes": 1800,
            "ict_index": "150.5",
            "expected_goal_involvements": "8.5",
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
                "assists": 3,
                "minutes": 1600,
                "ict_index": "120.0",
                "expected_goal_involvements": "6.0",
                "status": "a",
            },
            {
                "id": 352,
                "team": 10,
                "element_type": 4,
                "total_points": 120,
                "goals_scored": 15,
                "assists": 4,
                "minutes": 2000,
                "ict_index": "180.0",
                "expected_goal_involvements": "12.0",
                "status": "a",
            },
            {
                "id": 353,
                "team": 10,
                "element_type": 2,
                "total_points": 60,
                "goals_scored": 3,
                "assists": 2,
                "minutes": 1400,
                "ict_index": "80.0",
                "expected_goal_involvements": "3.5",
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

    def test_returns_35_features(self, sample_player):
        result = compute_team_features(sample_player)
        assert len(result) == 35

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

    def test_player_rank_in_team_points(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should rank player by total_points within team."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Points: 352=120, 350=100, 351=80, 353=60 → player 350 is rank 2
        assert result["player_rank_in_team_points"] == 2.0

    def test_player_rank_in_team_ict(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should rank player by ICT index within team."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # ICT: 352=180, 350=150.5, 351=120, 353=80 → player 350 is rank 2
        assert result["player_rank_in_team_ict"] == 2.0

    def test_player_share_of_team_assists(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should calculate player's share of team assists."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player: 5 assists, Team total: 5+3+4+2=14
        # Share: 5/14 = 0.3571
        assert result["player_share_of_team_assists"] == pytest.approx(
            0.3571, abs=0.001
        )

    def test_player_share_of_team_xgi(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should calculate player's share of team xGI."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player: 8.5 xGI, Team total: 8.5+6.0+12.0+3.5=30.0
        # Share: 8.5/30.0 = 0.2833
        assert result["player_share_of_team_xgi"] == pytest.approx(0.2833, abs=0.001)

    def test_player_minutes_share(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should calculate player minutes relative to team average."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player: 1800, Team avg: (1800+1600+2000+1400)/4 = 1700
        # Share: 1800/1700 = 1.059
        assert result["player_minutes_share"] == pytest.approx(1.059, abs=0.001)

    def test_player_points_vs_position_avg(
        self, sample_player, sample_team_data, sample_team_players
    ):
        """Should calculate player points minus avg of same-position teammates."""
        result = compute_team_features(
            sample_player,
            team_data=sample_team_data,
            team_players=sample_team_players,
        )
        # Player 350 is MID (element_type=3), teammate 351 is also MID with 80 pts
        # player_points - avg(others) = 100 - 80 = 20
        assert result["player_points_vs_position_avg"] == pytest.approx(20.0)

    def test_games_at_current_team_with_history(self, sample_player):
        """Should count consecutive recent games at current team."""
        history = [
            {"fixture": 1},
            {"fixture": 2},
            {"fixture": 3},
        ]
        all_fixtures = [
            {"id": 1, "team_h": 10, "team_a": 5},
            {"id": 2, "team_h": 3, "team_a": 10},
            {"id": 3, "team_h": 10, "team_a": 8},
        ]
        result = compute_team_features(
            sample_player, player_history=history, all_fixtures=all_fixtures
        )
        assert result["games_at_current_team"] == 3.0

    def test_games_at_current_team_without_history(self, sample_player):
        """Should default to 0 when no history provided."""
        result = compute_team_features(sample_player)
        assert result["games_at_current_team"] == 0.0

    def test_empty_inputs_return_defaults(self, sample_player):
        """Should return sensible defaults with no team data."""
        result = compute_team_features(sample_player)

        assert result["team_form_score"] == 0.0
        assert result["team_strength_overall"] == 0.0
        assert result["team_league_position"] == 10.0  # Default middle position
        assert result["player_rank_in_team_points"] == 0.0
        assert result["player_rank_in_team_ict"] == 0.0
        assert result["player_share_of_team_assists"] == 0.0
        assert result["player_share_of_team_xgi"] == 0.0
        assert result["player_minutes_share"] == 0.0
        assert result["player_points_vs_position_avg"] == 0.0
        assert result["games_at_current_team"] == 0.0


@pytest.mark.unit
class TestCountGamesAtCurrentTeam:
    def test_all_games_at_current_team(self):
        """All history entries match current team fixtures."""
        history = [{"fixture": 1}, {"fixture": 2}, {"fixture": 3}]
        fixtures_by_id = {
            1: {"team_h": 10, "team_a": 5},
            2: {"team_h": 3, "team_a": 10},
            3: {"team_h": 10, "team_a": 8},
        }
        assert _count_games_at_current_team(history, 10, fixtures_by_id) == 3

    def test_transfer_midseason(self):
        """Player transferred from team 5 to team 10 after fixture 2."""
        history = [{"fixture": 1}, {"fixture": 2}, {"fixture": 3}, {"fixture": 4}]
        fixtures_by_id = {
            1: {"team_h": 5, "team_a": 7},  # Old team
            2: {"team_h": 8, "team_a": 5},  # Old team
            3: {"team_h": 10, "team_a": 3},  # New team
            4: {"team_h": 6, "team_a": 10},  # New team
        }
        assert _count_games_at_current_team(history, 10, fixtures_by_id) == 2

    def test_empty_history(self):
        """Empty history returns 0."""
        assert _count_games_at_current_team([], 10, {}) == 0

    def test_missing_fixture_in_lookup(self):
        """Missing fixture ID stops the count."""
        history = [{"fixture": 1}, {"fixture": 999}]
        fixtures_by_id = {
            1: {"team_h": 10, "team_a": 5},
        }
        # fixture 999 not found, treated as empty dict → breaks the streak
        # Since we iterate in reverse: 999 first → breaks → count=0
        assert _count_games_at_current_team(history, 10, fixtures_by_id) == 0

    def test_single_game(self):
        """Single game at current team."""
        history = [{"fixture": 1}]
        fixtures_by_id = {1: {"team_h": 10, "team_a": 5}}
        assert _count_games_at_current_team(history, 10, fixtures_by_id) == 1
