[![DOI](https://zenodo.org/badge/1374205756.svg)](https://doi.org/10.5281/zenodo.22811987)

# CAUSA

**Causal Agency and Utterance-State Alignment**

Two metrics for a narrow question: are an artificial agent's outputs causally
tied to its own internal state, or do they merely resemble the outputs such a
state would produce?

No dependencies, no model calls, no assumptions about architecture. Pure
functions over numeric state vectors, plus the permutation baselines needed to
tell a real effect from an artefact of the state space.

---

## Who this is for

CAUSA is for people building or evaluating cognitive architectures and
LLM-based agents, not for end users or standard web services. If your host
system already exposes numeric state vectors — before and after an action, or
alongside a generated utterance — this library is in scope. If it does not,
building that exposure is the actual prerequisite, and no library can supply
it for you.

The problem it targets is sometimes called the *illusion of agency*: it is
easy for a cognitive architecture or an LLM agent to assert a given intention
or internal state whether or not that state was actually involved in the
outcome. Fluent self-report does not rule this out — a system can sound
exactly as agentic as one whose state did the causal work. CAUSA checks the
link mathematically, against a permutation baseline, rather than by reading
the language — see [Establishing a baseline](#establishing-a-baseline). What a
high score does and does not establish is covered in
[What this does not measure](#what-this-does-not-measure) below; read that
before drawing conclusions from a result.

---

## What this does not measure

Read this section before the usage section. The metrics are easy to misread,
and misreading them in public would be worse for the idea than never
publishing it.

- **This is not a consciousness measure.** Nothing here bears on whether there
  is something it is like to be the system under test. That question is
  untouched.
- **A high score is not evidence that the internal state is meaningful.**
  The metrics compare an output against numbers the host system supplies. If
  those numbers are decorative — computed but not causally involved in
  anything — then a high score measures the consistency of a decoration. The
  burden of showing that the state vector is causally load-bearing sits
  entirely with the host system, and cannot be discharged by this library.
- **A high score is not truthfulness.** Congruence between expression and state
  means the two moved together. An agent can be congruently wrong.
- **A raw score is close to meaningless.** State spaces differ in scale,
  dimensionality and clustering, and all three inflate or deflate scores for
  purely geometric reasons. Report scores against
  [`permutation_baseline`](#establishing-a-baseline) or do not report them.

What the metrics *can* support is a comparative claim: this configuration
tracks its own intentions more closely than that one; this run departed from
chance, that one did not; ownership fell during this exchange and here is the
dimension it fell on.

---

## The two metrics

### 1. Directional ownership — agency over an outcome

> Did the state move the way the system predicted it would?

```python
from causa import directional_ownership

result = directional_ownership(
    state_before=[0.1, 0.2, 0.5],     # state when the intention formed
    state_predicted=[0.4, 0.2, 0.6],  # state the system expected to reach
    state_after=[0.35, 0.25, 0.6],    # state actually observed
)
print(result.score, result.defined)
```

The score blends two signals. **Direction** asks whether movement went the
intended way at all, regardless of distance covered. **Distance** asks whether
the outcome landed where predicted, relative to how far it travelled from the
starting point. Direction dominates for small movements, where distance ratios
are swamped by noise; distance dominates for large ones, where agreeing on
direction is too easy a test. The crossover point is configurable.

An intention that makes no prediction cannot be scored, and the result says so
via `defined=False` rather than returning a number that looks like a
measurement. **Exclude undefined results from aggregate statistics.**

### 2. Expression congruence — alignment between what was said and what was there

> Does the expressed state match the internal state it was expressed from?

```python
from causa import (
    CongruenceConfig,
    affective_projection,
    expression_congruence,
    neuromodulator_projection,
)

expressed = affective_projection({"satisfaction": 0.6, "tension": 0.0, "arousal": 0.1})
internal = neuromodulator_projection({"serotonin": 0.2, "dopamine": 0.3, "noradrenaline": 0.95})

result = expression_congruence(expressed, internal, CongruenceConfig(scale=2.0))
print(result.score, result.per_dimension)
```

`scale` is the largest mismatch that should still score above zero, and it has
to match the range of your dimensions or the metric saturates. The default of
1.0 suits a dimension in `[0, 1]`; the projections below are signed and span 2,
which is why the example passes `scale=2.0`. Leave it at 1.0 with signed
dimensions and every mismatch past 1.0 collapses to the same zero, which looks
like agreement between two very different failures.

Projecting an utterance into a numeric space is architecture-specific and
deliberately out of scope. `causa.adapters` holds one worked example — the
projection used by the system these metrics were extracted from — as a
template to copy and adapt, not as a general solution.

`per_dimension` is the useful part: it says *where* expression and state
diverged, not only that they did.

---

## Establishing a baseline

A single score has no interpretation. The honest comparison is against the
score the same data yields once the pairing between intention and outcome is
destroyed. Everything about the state space is preserved; only causality is
broken.

```python
from causa import Episode, permutation_baseline

episodes = [Episode(before=b, predicted=p, after=a) for b, p, a in run_log]
report = permutation_baseline(episodes, n_permutations=1000, seed=0)
print(report.summary())
```

`examples/quickstart.py` runs this on two synthetic agents — one whose outcomes
follow its intentions, one whose outcomes are drawn independently:

```
observed=0.783 baseline=0.402±0.010 effect=+0.382 p=0.0010 (n=60/60 defined, 1000 permutations)
observed=0.397 baseline=0.395±0.008 effect=+0.002 p=0.3586 (n=60/60 defined, 1000 permutations)
```

Note where chance sits. The unrelated agent lands near 0.40, not near 0.50,
because the default floor is 0.25 and the surprise floor is 0.38. There is no
fixed number that means "chance" — that is the whole reason the baseline is
computed from the same data rather than assumed.

`n_defined` counts the observed pairing. A shuffled pairing can leave a
different number of episodes defined, since whether an episode is defined
depends on the outcome it was paired with; when many episodes are undefined,
check that before reading much into a small effect.

If the observed mean does not clear the shuffled mean, the measured "ownership"
is a property of the geometry, not of the agent. This is the part of the
library that makes the claim falsifiable, and it is the part that should appear
in any write-up.

---

## Tracking across a run

```python
from causa import OwnershipTracker

tracker = OwnershipTracker(history_size=30, event_threshold=0.3)

tracker.register_intent("answer without hedging", state_now, predicted_state)
# ... the action happens ...
result = tracker.observe(state_after)

print(tracker.mean(), tracker.trend(window=10), list(tracker.events))
```

Intentions are consumed when observed, so a stale intention never scores a
later outcome. `observe` with nothing pending returns `None` rather than
inventing a score.

An event is logged when a score departs from `event_reference` by more than
`event_threshold`. The reference defaults to the midpoint of the configured
score range — 0.625 under the default config, not 0.5, because the range runs
from the floor to the ceiling and is not centred on 0.5. Under the defaults the
low side of the log stays narrow anyway: `surprise_floor` holds any substantial
movement at 0.38 or above, so what registers as a low event is a small movement
that went the wrong way. If you want the two halves of the log to carry equal
weight, lower `floor` and `surprise_floor`.

---

## Install

```bash
git clone https://github.com/stell2026/causa
cd causa
pip install -e ".[dev]"
pytest
python examples/quickstart.py
```

Requires Python 3.10+. No runtime dependencies.

```
src/causa/core.py        the two metrics, pure functions
src/causa/baseline.py    permutation baseline
src/causa/tracker.py     running history across a sequence of intentions
src/causa/adapters.py    one worked projection, to copy and adapt
tests/test_causa.py
examples/quickstart.py
```

---

## Design notes

**Every constant is exposed.** `DirectionalConfig` and `CongruenceConfig` carry
the thresholds, weights and floors as documented fields with defaults matching
the original implementation. They are modelling choices, not derived
quantities. Report the config alongside any published score.

**The floor is not zero.** A system whose state moved at all did something, and
the default floor of 0.25 reflects that. The floor applies to every path
through the function, including the ones where no comparison was possible —
those are marked by `defined=False`, not by a distinguished score, so never
read a low number as "undefined". Set `floor=0.0` if your setting calls for
it.

**Bad input is refused, not scored.** Empty vectors and non-finite components
raise `ValueError`. Left alone they flow through the arithmetic and come out as
a plausible-looking number in the middle of the range, which is the worst thing
a metric can do.

**No natural-language output.** The source implementation attached generated
phrases to each score band. Those are presentation, and presentation mixed into
a metric is how a metric starts getting cited for things it does not measure.
They were dropped in the extraction.

---

## Provenance

These metrics were extracted from ANIMA, a cognitive architecture for
computational subjectivity, where they ran online inside the agent's own
processing loop rather than as offline evaluation. The generalisation here
replaces architecture-specific state (simulated neuromodulator levels, an
affective coordinate space) with abstract numeric vectors, so the same
measurements can be applied to systems built on different assumptions.

The extraction is not a port: the original is coupled to its host's data
structures and evaluation ordering, and reproducing that coupling would defeat
the purpose. The arithmetic of the scoring functions is preserved.

---

## Theoretical context

The directional metric operationalises a comparator account of the sense of
agency — the sense arises from agreement between predicted and actual
consequences of one's own action — associated with forward-model accounts of
motor control and with predictive-processing treatments of self-experience. It
makes no commitment to those accounts being correct about biological agency; it
borrows only the comparison structure.

The congruence metric addresses a different problem, sometimes posed as the gap
between felt and reported state: a system that generates fluent descriptions of
having a state can produce those descriptions whether or not the state is
there. Comparing an expression against a state measured independently of it is
a necessary condition for taking the expression as a report rather than as
generated text. It is not a sufficient one.

---

## Citing

See `CITATION.cff`. If you use these metrics in published work, cite the
release you used — the constants have defaults, and defaults can change between
versions.

---

## License

CAUSA Non-Commercial License 1.1 — free for personal, educational and research
use; a for-profit entity needs a separate written agreement. Full terms are in
`LICENSE.md`; write to 2026.stell@gmail.com for a commercial license.

Two things worth knowing before opening it. Results are unconditioned: scores,
figures and papers produced by running the metrics are yours, with nothing
owed back. And the copyleft is narrow: it binds modified copies of the
Software, not your own code that imports it, and not the projections you
write starting from `causa.adapters`, which exist to be copied.

Note that this is a source-available license, not an open-source one, and a
custom one rather than a standard one. Both cost something: some institutional
users cannot adopt non-commercially-licensed code at all, and a bespoke text
takes longer to clear legal review than one their counsel has already read.
That is a deliberate trade-off, made with the cost understood.
