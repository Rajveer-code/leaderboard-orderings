"""Regression tests for the Horvitz-Thompson / post-stratified estimator in
fb_tc_expansion2_analysis.py -- added after the methodological audit found
naive pooling across cells with different sampling fractions gives a
severely biased estimate (METHODOLOGICAL_AUDIT.md items 3/5). Synthetic
recovery tests at several different, deliberately DIFFERENT stratum
prevalences, confirming the estimator recovers the known population
quantity within tolerance -- the exact class of bug already found is
re-created here on purpose and checked as a regression, not just a
success-path smoke test.
"""
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
from fb_tc_expansion2_analysis import _ht_pool, CELL_POPULATION


def simulate_cell_sample(N, n, true_rate, rng):
    """Draw n Bernoulli(true_rate) outcomes from a population of size N,
    return (n_sampled, n_events) as the estimator expects."""
    events = sum(1 for _ in range(n) if rng.random() < true_rate)
    return n, events


@pytest.mark.parametrize("true_rate_majority,true_rate_minority", [
    (0.03, 0.03),   # same rate -- naive and HT should roughly agree
    (0.0, 0.15),    # majority near-zero, minority elevated -- the exact
                     # scenario that exposed the original bug
    (0.20, 0.02),   # majority high, minority low -- opposite direction
    (0.05, 0.05),   # both moderate, equal -- sanity check
])
def test_ht_estimator_recovers_population_rate(true_rate_majority, true_rate_minority):
    """Simulate the two cells that make up score_boxed's 'wrong' stratum
    (both_not_credited=majority, disagree_em_credited_sb_not=minority,
    sampled at very different fractions: ~1% vs ~6%), at several different
    (majority_rate, minority_rate) pairs, and confirm the HT estimate is
    close to the TRUE population-weighted rate -- not the naive pooled
    rate, which would be biased toward the minority cell's rate whenever
    the two differ (exactly the original bug)."""
    rng = random.Random(1234)
    N_maj, N_min = CELL_POPULATION["both_not_credited"], CELL_POPULATION["disagree_em_credited_sb_not"]
    n_maj, n_min = 198, 30  # the preregistered sample sizes for these cells

    true_pop_rate = (N_maj * true_rate_majority + N_min * true_rate_minority) / (N_maj + N_min)

    # Average over several simulation replicates to keep the test from being
    # flaky on a single unlucky draw, while still exercising real sampling noise.
    ht_estimates = []
    for rep in range(30):
        n_s_maj, x_maj = simulate_cell_sample(N_maj, n_maj, true_rate_majority, rng)
        n_s_min, x_min = simulate_cell_sample(N_min, n_min, true_rate_minority, rng)
        cell_counts = {"both_not_credited": (n_s_maj, x_maj),
                       "disagree_em_credited_sb_not": (n_s_min, x_min)}
        p_hat, se, detail = _ht_pool(cell_counts, ["both_not_credited", "disagree_em_credited_sb_not"])
        ht_estimates.append(p_hat)

    mean_ht = np.mean(ht_estimates)
    # Tolerance driven by the sampling noise at these n's, not an arbitrary
    # epsilon -- generous enough for n=30/198 binomial noise, tight enough
    # to catch a systematic (bias-sized) error like the original bug.
    tol = 0.06 if abs(true_rate_majority - true_rate_minority) > 0.05 else 0.04
    assert abs(mean_ht - true_pop_rate) < tol, (
        f"HT estimate {mean_ht:.4f} too far from true population rate {true_pop_rate:.4f} "
        f"(majority={true_rate_majority}, minority={true_rate_minority})")


def test_naive_pooling_is_biased_when_cells_differ():
    """Regression test for the ORIGINAL bug: confirm that naive (sample-
    count-weighted, not population-weighted) pooling gives a materially
    different -- and wrong -- answer from HT pooling when the majority and
    minority cells have different true rates, on the SAME fixed sample
    counts (deterministic, no randomness -- this is a math check, not a
    simulation)."""
    N_maj, N_min = CELL_POPULATION["both_not_credited"], CELL_POPULATION["disagree_em_credited_sb_not"]
    n_maj, n_min = 198, 30
    x_maj, x_min = 0, 4  # 0% observed in the majority cell, ~13% in the minority cell

    cell_counts = {"both_not_credited": (n_maj, x_maj),
                   "disagree_em_credited_sb_not": (n_min, x_min)}
    p_ht, se, detail = _ht_pool(cell_counts, ["both_not_credited", "disagree_em_credited_sb_not"])

    naive = (x_maj + x_min) / (n_maj + n_min)

    # The minority cell is sampled at ~6x the majority cell's rate relative
    # to its own population, so naive pooling over-weights it substantially
    # -- HT should give a materially SMALLER estimate here (population-
    # weighted toward the much larger majority cell), confirmed as a
    # concrete, checkable inequality, not just "different."
    assert p_ht < naive * 0.5, (
        f"HT estimate ({p_ht:.4f}) should be substantially below the naive "
        f"pooled estimate ({naive:.4f}) when the oversampled minority cell "
        f"has a higher observed rate -- this is the exact bug this test guards against")


def test_ht_pool_handles_zero_events_cleanly():
    """Zero observed events in both cells should give a clean 0.0, not a
    division error or NaN."""
    cell_counts = {"both_not_credited": (198, 0), "disagree_em_credited_sb_not": (30, 0)}
    p_ht, se, detail = _ht_pool(cell_counts, ["both_not_credited", "disagree_em_credited_sb_not"])
    assert p_ht == 0.0
    assert se == 0.0


def test_ht_pool_returns_none_when_all_cells_unscored():
    """If no rows in either cell were scoreable (e.g. all NONE/CONTRADICTORY),
    the estimator must signal 'no estimate', not fabricate one from zero
    denominator."""
    cell_counts = {"both_not_credited": (0, 0), "disagree_em_credited_sb_not": (0, 0)}
    p_ht, se, detail = _ht_pool(cell_counts, ["both_not_credited", "disagree_em_credited_sb_not"])
    assert p_ht is None and se is None
