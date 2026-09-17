"""End-to-end walkthrough: two metrics, a tracked run, and a baseline.

Run with ``python examples/quickstart.py`` from the repository root.
"""

from __future__ import annotations

import random

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


def one_shot_ownership() -> None:
    print("--- directional ownership -------------------------------------")

    tracked = directional_ownership(
        state_before=[0.10, 0.20, 0.50],
        state_predicted=[0.40, 0.20, 0.60],
        state_after=[0.35, 0.25, 0.60],
    )
    print(f"outcome tracked the intention : {tracked.score:.3f}")

    missed = directional_ownership(
        state_before=[0.10, 0.20, 0.50],
        state_predicted=[0.40, 0.20, 0.60],
        state_after=[-0.30, 0.15, 0.45],
    )
    print(f"outcome went the other way    : {missed.score:.3f}")
    print(f"  direction component         : {missed.directional_component:.3f}")
    print(f"  distance travelled          : {missed.distance_travelled:.3f}")

    still = directional_ownership(
        state_before=[0.10, 0.20, 0.50],
        state_predicted=[0.40, 0.20, 0.60],
        state_after=[0.10, 0.20, 0.50],
    )
    print(f"nothing happened (defined={still.defined}) : {still.score:.3f}")
    print()


def one_shot_congruence() -> None:
    print("--- expression congruence -------------------------------------")

    agitated_state = neuromodulator_projection(
        {"serotonin": 0.20, "dopamine": 0.30, "noradrenaline": 0.95}
    )

    calm_words = affective_projection(
        {"satisfaction": 0.60, "tension": 0.00, "arousal": 0.05}
    )
    matching_words = affective_projection(
        {"satisfaction": 0.05, "tension": 0.55, "arousal": 0.70}
    )

    # The projections in causa.adapters produce signed dimensions in
    # [-1, 1], a range of 2. The default scale of 1.0 assumes [0, 1] and
    # would flatten every large mismatch to the same zero.
    cfg = CongruenceConfig(scale=2.0)

    a = expression_congruence(calm_words, agitated_state, cfg)
    b = expression_congruence(matching_words, agitated_state, cfg)

    print(f"agitated state, calm words    : {a.score:.3f}  {a.per_dimension}")
    print(f"agitated state, matching words: {b.score:.3f}  {b.per_dimension}")
    print()


def tracked_run() -> None:
    print("--- tracked run -----------------------------------------------")

    rng = random.Random(3)

    # floor and surprise_floor are relaxed so that a step which went the
    # wrong way can register as an event. Under the defaults the score of
    # such a step is held at 0.38 and only the good steps reach the log.
    tracker = OwnershipTracker(
        history_size=40,
        event_threshold=0.3,
        config=DirectionalConfig(floor=0.0, surprise_floor=0.0),
    )

    for step in range(40):
        before = [rng.uniform(-0.2, 0.2) for _ in range(3)]
        intended_shift = [rng.uniform(-0.4, 0.4) for _ in range(3)]
        predicted = [b + s for b, s in zip(before, intended_shift, strict=True)]

        # The agent mostly gets where it meant to go, with noise — and is
        # knocked off course entirely every seventh step.
        if step % 7 == 6:
            after = [
                b - s + rng.gauss(0, 0.05)
                for b, s in zip(before, intended_shift, strict=True)
            ]
        else:
            after = [p + rng.gauss(0, 0.06) for p in predicted]

        tracker.register_intent(f"step-{step}", before, predicted)
        tracker.observe(after)

    print(f"mean ownership : {tracker.mean():.3f}")
    print(f"confidence     : {tracker.confidence:.3f}")
    print(f"trend (last 10): {tracker.trend(window=10):+.3f}")
    print(
        f"notable events : {len(tracker.events)}"
        f"  (reference {tracker.event_reference:.2f})"
    )
    for event in list(tracker.events)[:6]:
        print(f"  step {event.step:>3}  {event.label:<10} {event.score:.3f}")
    print()


def baseline() -> None:
    print("--- permutation baseline --------------------------------------")

    rng = random.Random(3)

    real, null = [], []
    for _ in range(60):
        before = [rng.uniform(-0.2, 0.2) for _ in range(3)]
        shift = [rng.uniform(-0.4, 0.4) for _ in range(3)]
        predicted = [b + s for b, s in zip(before, shift, strict=True)]

        real.append(
            Episode(before, predicted, [p + rng.gauss(0, 0.06) for p in predicted])
        )
        null.append(
            Episode(before, predicted, [rng.uniform(-0.5, 0.5) for _ in range(3)])
        )

    print("agent whose outcomes follow its intentions:")
    print("  " + permutation_baseline(real, n_permutations=1000, seed=0).summary())
    print("agent whose outcomes are unrelated:")
    print("  " + permutation_baseline(null, n_permutations=1000, seed=0).summary())
    print()
    print("The second case is what a decorative state vector looks like.")


if __name__ == "__main__":
    one_shot_ownership()
    one_shot_congruence()
    tracked_run()
    baseline()
