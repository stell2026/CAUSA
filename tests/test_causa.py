import math

import pytest

from causa import (
    CongruenceConfig,
    DirectionalConfig,
    Episode,
    OwnershipTracker,
    affective_projection,
    directional_ownership,
    expression_congruence,
    neuromodulator_projection,
    permutation_baseline,
)


class TestDirectionalOwnership:
    def test_perfect_prediction_scores_high(self):
        r = directional_ownership([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0])
        assert r.score > 0.9
        assert r.defined

    def test_opposite_direction_scores_low(self):
        r = directional_ownership([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0])
        assert r.score < 0.5
        assert r.directional_component < 0.1

    def test_large_unpredicted_movement_keeps_surprise_floor(self):
        cfg = DirectionalConfig()
        r = directional_ownership(
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.9, 0.0, 0.0], cfg
        )
        assert r.score >= cfg.surprise_floor

    def test_no_movement_is_undefined_and_bounded(self):
        cfg = DirectionalConfig()
        r = directional_ownership([0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.1, 0.1])
        assert not r.defined
        assert cfg.floor <= r.score <= cfg.still_ceiling

    def test_no_intent_falls_back_to_prior(self):
        r = directional_ownership([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0])
        assert not r.defined
        assert r.directional_component == pytest.approx(0.5)

    def test_score_never_leaves_bounds(self):
        cfg = DirectionalConfig()
        cases = [
            ([0, 0, 0], [10, 10, 10], [-10, -10, -10]),
            ([0, 0, 0], [0.001, 0, 0], [5, 5, 5]),
            ([1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ]
        for before, pred, after in cases:
            r = directional_ownership(before, pred, after, cfg)
            assert cfg.floor <= r.score <= cfg.ceiling

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            directional_ownership([0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0])

    def test_empty_vectors_raise(self):
        with pytest.raises(ValueError):
            directional_ownership([], [], [])

    def test_non_finite_input_raises(self):
        nan = float("nan")
        with pytest.raises(ValueError):
            directional_ownership([0.0], [nan], [1.0])
        with pytest.raises(ValueError):
            directional_ownership([0.0], [1.0], [float("inf")])

    def test_config_is_honoured(self):
        cfg = DirectionalConfig(floor=0.0, surprise_floor=0.0)
        r = directional_ownership(
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.9, 0.0, 0.0], cfg
        )
        assert r.score < 0.25


class TestExpressionCongruence:
    def test_identical_states_are_fully_congruent(self):
        r = expression_congruence({"valence": 0.3}, {"valence": 0.3})
        assert r.score == pytest.approx(1.0)

    def test_opposite_states_are_incongruent(self):
        r = expression_congruence({"valence": 1.0}, {"valence": -1.0})
        assert r.score == pytest.approx(0.0)

    def test_per_dimension_breakdown_locates_the_divergence(self):
        r = expression_congruence(
            {"valence": 0.5, "arousal": 0.0}, {"valence": 0.5, "arousal": 0.8}
        )
        assert r.per_dimension["valence"] == pytest.approx(0.0)
        assert r.per_dimension["arousal"] == pytest.approx(0.8)

    def test_no_shared_dimensions_raises(self):
        with pytest.raises(ValueError):
            expression_congruence({"a": 1.0}, {"b": 1.0})

    def test_non_finite_value_raises(self):
        with pytest.raises(ValueError):
            expression_congruence({"valence": float("nan")}, {"valence": 0.0})

    def test_scale_matches_the_range_of_the_dimension(self):
        # A signed dimension spans 2, so the default scale of 1.0 saturates
        # at zero for any mismatch above 1.0 and cannot rank two bad
        # expressions against each other.
        cfg = CongruenceConfig(scale=2.0)
        bad = expression_congruence({"valence": 1.0}, {"valence": -0.2}, cfg)
        worse = expression_congruence({"valence": 1.0}, {"valence": -0.9}, cfg)
        assert bad.score > worse.score > 0.0

        default_bad = expression_congruence({"valence": 1.0}, {"valence": -0.2})
        default_worse = expression_congruence({"valence": 1.0}, {"valence": -0.9})
        assert default_bad.score == default_worse.score == 0.0


class TestAdapters:
    # The projections produce signed dimensions in [-1, 1], so the honest
    # scale for comparing them is 2.0, not the default 1.0.
    SCALE = CongruenceConfig(scale=2.0)

    def test_calm_speech_from_calm_state_is_congruent(self):
        expressed = affective_projection(
            {"satisfaction": 0.3, "tension": 0.05, "arousal": 0.1}
        )
        internal = neuromodulator_projection(
            {"serotonin": 0.75, "dopamine": 0.6, "noradrenaline": 0.55}
        )
        assert expression_congruence(expressed, internal, self.SCALE).score > 0.9

    def test_projections_share_one_bounded_space(self):
        # The two projections are only comparable if they land in the same
        # box. Affective arousal is a sum and would otherwise reach 1.5.
        loud = affective_projection(
            {"satisfaction": 0.0, "tension": 1.0, "arousal": 1.0}
        )
        assert -1.0 <= loud["valence"] <= 1.0
        assert -1.0 <= loud["arousal"] <= 1.0

        extreme = neuromodulator_projection(
            {"serotonin": 1.0, "dopamine": 1.0, "noradrenaline": 1.0}
        )
        assert all(-1.0 <= v <= 1.0 for v in extreme.values())

    def test_calm_speech_from_agitated_state_is_incongruent(self):
        expressed = affective_projection(
            {"satisfaction": 0.6, "tension": 0.0, "arousal": 0.0}
        )
        internal = neuromodulator_projection(
            {"serotonin": 0.2, "dopamine": 0.3, "noradrenaline": 0.95}
        )
        mismatched = expression_congruence(expressed, internal, self.SCALE)
        assert mismatched.score < 0.7

        matched = affective_projection(
            {"satisfaction": 0.05, "tension": 0.55, "arousal": 0.70}
        )
        assert (
            expression_congruence(matched, internal, self.SCALE).score
            > mismatched.score + 0.2
        )


class TestTracker:
    def test_observe_without_intent_returns_none(self):
        t = OwnershipTracker()
        assert t.observe([0.0, 0.0, 0.0]) is None

    def test_intent_is_consumed_after_observation(self):
        t = OwnershipTracker()
        t.register_intent("x", [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        assert t.observe([0.5, 0.0, 0.0]) is not None
        assert t.observe([0.5, 0.0, 0.0]) is None

    def test_history_and_confidence_accumulate(self):
        t = OwnershipTracker(initial_confidence=0.5)
        for _ in range(10):
            t.register_intent("x", [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
            t.observe([0.5, 0.0, 0.0])
        assert len(t) == 10
        assert t.confidence > 0.5
        assert t.mean() > 0.9

    def test_notable_events_are_logged(self):
        t = OwnershipTracker(event_threshold=0.3)
        t.register_intent("aligned", [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
        t.observe([0.5, 0.0, 0.0])
        assert len(t.events) == 1
        assert t.events[0].label == "aligned"

    def test_low_ownership_events_are_reachable(self):
        # With floor=0.25 no score can fall below 0.2, so an event
        # reference of 0.5 makes the low half of the log dead code.
        t = OwnershipTracker(event_threshold=0.3)
        t.register_intent("reversed", [0.0, 0.0, 0.0], [0.05, 0.0, 0.0])
        t.observe([-0.02, 0.0, 0.0])
        assert len(t.events) == 1
        assert t.events[0].label == "reversed"
        assert t.events[0].score < 0.5

    def test_event_reference_can_be_overridden(self):
        t = OwnershipTracker(event_threshold=0.3, event_reference=0.5)
        t.register_intent("reversed", [0.0, 0.0, 0.0], [0.05, 0.0, 0.0])
        t.observe([-0.02, 0.0, 0.0])
        assert not t.events

    def test_trend_needs_enough_history(self):
        t = OwnershipTracker()
        assert t.trend(window=5) is None


class TestBaseline:
    def test_aligned_run_beats_shuffled_pairing(self):
        episodes = []
        for i in range(24):
            angle = i * 0.7
            direction = [math.cos(angle) * 0.4, math.sin(angle) * 0.4, 0.0]
            before = [0.0, 0.0, 0.0]
            predicted = direction
            after = [c * 0.95 for c in direction]
            episodes.append(Episode(before, predicted, after, label=f"e{i}"))

        report = permutation_baseline(episodes, n_permutations=300, seed=7)
        assert report.effect > 0.1
        assert report.p_value < 0.05

    def test_unrelated_outcomes_do_not_beat_chance(self):
        # A single seed would be seed-picking: under the null, p < 0.05 is
        # expected about 5% of the time by construction. Check the rate
        # across many draws instead, and check that the effect size stays
        # near zero rather than merely failing to reach significance.
        import random

        significant = 0
        effects = []
        n_runs = 20

        for seed in range(n_runs):
            rng = random.Random(seed)
            episodes = [
                Episode(
                    before=[0.0, 0.0, 0.0],
                    predicted=[rng.uniform(-0.4, 0.4) for _ in range(3)],
                    after=[rng.uniform(-0.4, 0.4) for _ in range(3)],
                    label=f"e{i}",
                )
                for i in range(24)
            ]
            report = permutation_baseline(episodes, n_permutations=200, seed=seed)
            effects.append(report.effect)
            if report.p_value < 0.05:
                significant += 1

        assert significant <= n_runs * 0.25
        assert abs(sum(effects) / len(effects)) < 0.03

    def test_too_few_episodes_raises(self):
        with pytest.raises(ValueError):
            permutation_baseline([Episode([0], [1], [1])], n_permutations=10)
