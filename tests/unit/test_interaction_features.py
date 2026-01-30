"""
Unit tests for interaction features module.
"""

import pytest

from lambdas.common.feature_categories.interaction_features import (
    INTERACTION_FEATURES,
    compute_interaction_features,
)


@pytest.mark.unit
class TestInteractionFeaturesList:
    def test_has_5_features(self):
        """INTERACTION_FEATURES should contain 5 features."""
        assert len(INTERACTION_FEATURES) == 5

    def test_no_duplicates(self):
        """All interaction feature names should be unique."""
        assert len(INTERACTION_FEATURES) == len(set(INTERACTION_FEATURES))

    def test_key_features_present(self):
        """Verify all interaction features are in the list."""
        expected = [
            "form_x_fixture_difficulty",
            "ict_x_minutes",
            "ownership_x_form",
            "value_x_form",
            "momentum_score",
        ]
        for feat in expected:
            assert feat in INTERACTION_FEATURES, f"Missing: {feat}"


@pytest.mark.unit
class TestComputeInteractionFeatures:
    @pytest.fixture
    def sample_player(self):
        return {
            "id": 350,
            "form": "8.5",
            "ict_index": "150.0",
            "selected_by_percent": "45.3",
            "now_cost": 130,  # 13.0M
            "minutes": 1800,
            "starts": 20,
        }

    @pytest.fixture
    def sample_rolling_features(self):
        return {
            "ict_index_last_5": 85.0,
            "points_last_3": 7.5,
            "minutes_last_5": 90.0,
        }

    @pytest.fixture
    def sample_fixture_features(self):
        return {
            "fdr_current": 2.0,  # Easy fixture
        }

    @pytest.fixture
    def sample_bootstrap_features(self):
        return {
            "net_transfers_event": 50000,
            "transfer_momentum": 0.8,
        }

    def test_returns_5_features(self):
        result = compute_interaction_features()
        assert len(result) == 5

    def test_all_feature_names_present(self):
        result = compute_interaction_features()
        for name in INTERACTION_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_form_x_fixture_difficulty(self, sample_player, sample_fixture_features):
        """Form x inverse FDR should reward easy fixtures."""
        result = compute_interaction_features(
            player=sample_player,
            fixture_features=sample_fixture_features,
        )
        # Form 8.5 * inverse_fdr (6 - 2 = 4) = 34.0
        assert result["form_x_fixture_difficulty"] == 34.0

    def test_form_x_fixture_difficulty_tough(self, sample_player):
        """Tough fixture should reduce form x difficulty."""
        fixture_features = {"fdr_current": 5.0}
        result = compute_interaction_features(
            player=sample_player,
            fixture_features=fixture_features,
        )
        # Form 8.5 * inverse_fdr (6 - 5 = 1) = 8.5
        assert result["form_x_fixture_difficulty"] == 8.5

    def test_ict_x_minutes(self, sample_player, sample_rolling_features):
        """ICT x minutes should reward high ICT players who play."""
        result = compute_interaction_features(
            player=sample_player,
            rolling_features=sample_rolling_features,
        )
        # ICT 85.0 * (90 mins / 90) = 85.0
        assert result["ict_x_minutes"] == 85.0

    def test_ownership_x_form(self, sample_player):
        """Ownership x form should identify popular in-form players."""
        result = compute_interaction_features(player=sample_player)
        # Ownership 45.3 * form 8.5 = 385.05
        assert result["ownership_x_form"] == pytest.approx(385.05, abs=0.1)

    def test_value_x_form(self, sample_player):
        """Value x form should identify premium in-form players."""
        result = compute_interaction_features(player=sample_player)
        # Cost 130 / 10 = 13.0M * form 8.5 = 110.5
        assert result["value_x_form"] == pytest.approx(110.5, abs=0.1)

    def test_momentum_score(
        self, sample_player, sample_rolling_features, sample_bootstrap_features
    ):
        """Momentum score should combine transfers, form, and points."""
        result = compute_interaction_features(
            player=sample_player,
            rolling_features=sample_rolling_features,
            bootstrap_features=sample_bootstrap_features,
        )
        # Transfer signal: min(1, 50000/100000) = 0.5
        # Form signal: min(1, 8.5/10) = 0.85
        # Points signal: min(1, 7.5/20) = 0.375
        # Momentum = (0.5 * 0.3) + (0.85 * 0.4) + (0.375 * 0.3) = 0.6025
        # Scaled * 10 = 6.025
        assert result["momentum_score"] == pytest.approx(6.03, abs=0.1)

    def test_momentum_score_negative_transfers(self, sample_player):
        """Negative transfers should reduce momentum."""
        bootstrap_features = {"net_transfers_event": -80000}
        rolling_features = {"points_last_3": 2.0}
        result = compute_interaction_features(
            player=sample_player,
            rolling_features=rolling_features,
            bootstrap_features=bootstrap_features,
        )
        # Transfer signal: max(-1, -80000/100000) = -0.8
        # Momentum will be lower due to negative transfers
        assert result["momentum_score"] < 5.0

    def test_empty_inputs_return_defaults(self):
        """Should return sensible defaults with no data."""
        result = compute_interaction_features()

        assert result["form_x_fixture_difficulty"] == 0.0
        assert result["ict_x_minutes"] == 0.0
        assert result["ownership_x_form"] == 0.0
        assert result["value_x_form"] == 0.0
        assert result["momentum_score"] == 0.0

    def test_with_string_form(self, sample_rolling_features):
        """Should handle form as string (FPL API format)."""
        player = {"form": "7.5", "selected_by_percent": "30.0", "now_cost": 100}
        result = compute_interaction_features(
            player=player,
            rolling_features=sample_rolling_features,
        )
        # Should parse form correctly
        assert result["ownership_x_form"] == pytest.approx(225.0, abs=0.1)
