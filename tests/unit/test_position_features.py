"""
Unit tests for position-specific features module.
"""

import pytest

from lambdas.common.feature_categories.position_features import (
    POSITION_FEATURES,
    compute_position_features,
)


@pytest.mark.unit
class TestPositionFeaturesList:
    def test_has_8_features(self):
        """POSITION_FEATURES should contain 8 features."""
        assert len(POSITION_FEATURES) == 8

    def test_no_duplicates(self):
        """All position feature names should be unique."""
        assert len(POSITION_FEATURES) == len(set(POSITION_FEATURES))

    def test_key_features_present(self):
        """Verify all position features are in the list."""
        expected = [
            "gk_saves_per_90",
            "gk_penalty_save_rate",
            "def_clean_sheet_rate",
            "def_goal_involvement",
            "mid_goal_involvement_rate",
            "mid_creativity_threat_ratio",
            "fwd_shots_per_90",
            "fwd_conversion_rate",
        ]
        for feat in expected:
            assert feat in POSITION_FEATURES, f"Missing: {feat}"


@pytest.mark.unit
class TestComputePositionFeatures:
    @pytest.fixture
    def sample_goalkeeper(self):
        return {
            "id": 100,
            "element_type": 1,  # GK
            "minutes": 1800,
            "saves": 60,
            "penalties_saved": 2,
            "goals_scored": 0,
            "assists": 1,
            "clean_sheets": 8,
        }

    @pytest.fixture
    def sample_defender(self):
        return {
            "id": 200,
            "element_type": 2,  # DEF
            "minutes": 1620,
            "goals_scored": 3,
            "assists": 2,
            "clean_sheets": 10,
        }

    @pytest.fixture
    def sample_midfielder(self):
        return {
            "id": 300,
            "element_type": 3,  # MID
            "minutes": 1800,
            "goals_scored": 8,
            "assists": 10,
            "creativity": "150.0",
            "threat": "200.0",
        }

    @pytest.fixture
    def sample_forward(self):
        return {
            "id": 400,
            "element_type": 4,  # FWD
            "minutes": 1620,
            "goals_scored": 15,
            "threat": "300.0",
        }

    @pytest.fixture
    def sample_history(self):
        return [
            {"minutes": 90, "goals_conceded": 1},
            {"minutes": 90, "goals_conceded": 0},
            {"minutes": 90, "goals_conceded": 2},
            {"minutes": 75, "goals_conceded": 1},
            {"minutes": 90, "goals_conceded": 0},
        ]

    def test_returns_8_features(self):
        result = compute_position_features()
        assert len(result) == 8

    def test_all_feature_names_present(self):
        result = compute_position_features()
        for name in POSITION_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_gk_saves_per_90(self, sample_goalkeeper, sample_history):
        """Should calculate saves per 90 minutes for goalkeepers."""
        result = compute_position_features(
            player=sample_goalkeeper,
            history=sample_history,
        )
        # 60 saves / 1800 minutes * 90 = 3.0
        assert result["gk_saves_per_90"] == 3.0

    def test_gk_penalty_save_rate(self, sample_goalkeeper, sample_history):
        """Should calculate penalty save rate for goalkeepers."""
        result = compute_position_features(
            player=sample_goalkeeper,
            history=sample_history,
        )
        # 2 saved, estimated 1 penalty goal from history (4 conceded * 0.05 = 1)
        # 2 / 3 = 0.667
        assert result["gk_penalty_save_rate"] == pytest.approx(0.667, abs=0.01)

    def test_gk_features_zero_for_non_gk(self, sample_defender):
        """GK features should be 0 for non-goalkeepers."""
        result = compute_position_features(player=sample_defender)
        assert result["gk_saves_per_90"] == 0.0
        assert result["gk_penalty_save_rate"] == 0.0

    def test_def_clean_sheet_rate(self, sample_defender, sample_history):
        """Should calculate clean sheet rate for defenders."""
        result = compute_position_features(
            player=sample_defender,
            history=sample_history,
        )
        # 10 clean sheets, 5 games started (all have minutes >= 60)
        # 10 / 5 = 2.0 (season total CS / games started in history sample)
        # Note: using player's season total CS, not history CS
        assert result["def_clean_sheet_rate"] == pytest.approx(2.0, abs=0.01)

    def test_def_goal_involvement(self, sample_defender, sample_history):
        """Should calculate goal involvement for defenders."""
        result = compute_position_features(
            player=sample_defender,
            history=sample_history,
        )
        # 3 goals + 2 assists = 5, 5 games started (all have minutes >= 60)
        # 5 / 5 = 1.0
        assert result["def_goal_involvement"] == pytest.approx(1.0, abs=0.01)

    def test_def_features_zero_for_non_def(self, sample_midfielder):
        """DEF features should be 0 for non-defenders."""
        result = compute_position_features(player=sample_midfielder)
        assert result["def_clean_sheet_rate"] == 0.0
        assert result["def_goal_involvement"] == 0.0

    def test_mid_goal_involvement_rate(self, sample_midfielder):
        """Should calculate goal involvement rate for midfielders."""
        result = compute_position_features(player=sample_midfielder)
        # 8 goals + 10 assists = 18, 1800 minutes
        # 18 / 1800 * 90 = 0.9
        assert result["mid_goal_involvement_rate"] == 0.9

    def test_mid_creativity_threat_ratio(self, sample_midfielder):
        """Should calculate creativity/threat ratio for midfielders."""
        result = compute_position_features(player=sample_midfielder)
        # 150 creativity / (200 threat + 1) = 0.746
        assert result["mid_creativity_threat_ratio"] == pytest.approx(0.746, abs=0.01)

    def test_mid_features_zero_for_non_mid(self, sample_forward):
        """MID features should be 0 for non-midfielders."""
        result = compute_position_features(player=sample_forward)
        assert result["mid_goal_involvement_rate"] == 0.0
        assert result["mid_creativity_threat_ratio"] == 0.0

    def test_fwd_shots_per_90(self, sample_forward):
        """Should estimate shots per 90 for forwards."""
        result = compute_position_features(player=sample_forward)
        # Estimated shots = 300 threat / 15 = 20
        # 20 / 1620 * 90 = 1.11
        assert result["fwd_shots_per_90"] == pytest.approx(1.11, abs=0.01)

    def test_fwd_conversion_rate(self, sample_forward):
        """Should calculate conversion rate for forwards."""
        result = compute_position_features(player=sample_forward)
        # 15 goals / 20 estimated shots = 0.75
        assert result["fwd_conversion_rate"] == 0.75

    def test_fwd_features_zero_for_non_fwd(self, sample_midfielder):
        """FWD features should be 0 for non-forwards."""
        result = compute_position_features(player=sample_midfielder)
        assert result["fwd_shots_per_90"] == 0.0
        assert result["fwd_conversion_rate"] == 0.0

    def test_empty_inputs_return_zeros(self):
        """Should return zeros with no data."""
        result = compute_position_features()

        for feat in POSITION_FEATURES:
            assert result[feat] == 0.0, f"{feat} should be 0.0"

    def test_zero_minutes_returns_defaults(self, sample_goalkeeper):
        """Should handle zero minutes gracefully."""
        player = sample_goalkeeper.copy()
        player["minutes"] = 0
        result = compute_position_features(player=player)

        assert result["gk_saves_per_90"] == 0.0
