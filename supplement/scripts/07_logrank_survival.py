"""
Log-rank test for RQ2 survival comparison.
Tests H0: self-repaid SATD and non-self-repaid SATD share the same survival distribution.
Since every SATD in the dataset has a remove_date, there is no censoring:
all observations are events, so the log-rank reduces to the standard two-sample version.
"""
import json
import math
from collections import Counter

import numpy as np
from scipy import stats

DATA = 'Self Repaid SATD/paper/current/work/data/merged_data_40501_temporal_sorted_by_add_date.json'

def load():
    with open(DATA, encoding='utf-8') as f:
        rows = json.load(f)
    times = np.array([r['satd_survival_days'] for r in rows], dtype=float)
    group = np.array([int(r['is_self_fixed']) for r in rows], dtype=int)
    # drop any row without survival_days just in case
    mask = ~np.isnan(times)
    return times[mask], group[mask]

def logrank_two_sample(times, group):
    """Two-sample log-rank test, no censoring (every observation is an event).
    group: 1 = self-repaid, 0 = non-self-repaid.
    Returns (chi2, p_value, summary).
    """
    unique_times = np.sort(np.unique(times))
    n_total = len(times)
    n1_total = int((group == 1).sum())  # self-repaid
    n0_total = n_total - n1_total

    obs_minus_exp = 0.0
    variance = 0.0
    obs1 = 0
    exp1 = 0.0

    at_risk_1 = n1_total
    at_risk_0 = n0_total

    for t in unique_times:
        at_risk_total = at_risk_1 + at_risk_0
        if at_risk_total < 2:
            break
        # events at time t
        mask = times == t
        d1 = int(((group == 1) & mask).sum())
        d0 = int(((group == 0) & mask).sum())
        d = d1 + d0
        if d == 0:
            continue
        e1 = d * at_risk_1 / at_risk_total
        obs_minus_exp += (d1 - e1)
        obs1 += d1
        exp1 += e1
        if at_risk_total > 1:
            v = (at_risk_1 * at_risk_0 * d * (at_risk_total - d)) / (
                at_risk_total ** 2 * (at_risk_total - 1)
            )
            variance += v
        # decrement at-risk counts AFTER accounting for this time
        at_risk_1 -= d1
        at_risk_0 -= d0

    chi2 = (obs_minus_exp) ** 2 / variance if variance > 0 else float('nan')
    p = 1 - stats.chi2.cdf(chi2, df=1) if not math.isnan(chi2) else float('nan')
    return chi2, p, {
        'n_total': n_total,
        'n_self_repaid': n1_total,
        'n_non_self_repaid': n0_total,
        'observed_self_repaid_events': obs1,
        'expected_self_repaid_events': exp1,
        'variance': variance,
        'median_self_repaid': float(np.median(times[group == 1])),
        'median_non_self_repaid': float(np.median(times[group == 0])),
    }


def main():
    times, group = load()
    print(f'N={len(times)}, self-repaid={int((group==1).sum())}, non-self-repaid={int((group==0).sum())}')
    print(f'Median survival (self-repaid, days): {np.median(times[group==1]):.1f}')
    print(f'Median survival (non-self-repaid, days): {np.median(times[group==0]):.1f}')
    print()
    chi2, p, info = logrank_two_sample(times, group)
    print('=== Log-rank test ===')
    print(f'chi2 = {chi2:.2f}')
    if p == 0:
        print(f'p < 1e-300 (numerical underflow; effectively p ~ 0)')
    else:
        print(f'p = {p:.3e}')
    print(f'Observed events in self-repaid group: {info["observed_self_repaid_events"]}')
    print(f'Expected under H0: {info["expected_self_repaid_events"]:.1f}')
    print(f'Observed - Expected: {info["observed_self_repaid_events"] - info["expected_self_repaid_events"]:+.1f}')
    print(f'Variance of (O-E): {info["variance"]:.1f}')

    # also print for completeness the Mann-Whitney U value for cross-reference
    mw = stats.mannwhitneyu(times[group == 1], times[group == 0], alternative='two-sided')
    print()
    print('=== Cross-reference: Mann-Whitney U (already in paper) ===')
    print(f'U = {mw.statistic:.1f}')
    print(f'p = {mw.pvalue:.3e}' if mw.pvalue > 0 else 'p ~ 0 (numerical underflow)')


if __name__ == '__main__':
    main()
