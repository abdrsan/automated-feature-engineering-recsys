import numpy as np
import polars as pl

from audit.run_checks import independent_user_bootstrap


def test_independent_bootstrap_uses_stable_user_order():
    users = np.array([9, 2, 9, 5, 2, 8, 1, 5, 5, 8, 3, 1])
    labels = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    baseline = np.array([.7, .2, .3, .8, .6, .4, .9, .3, .7, .8, .1, .4])
    candidate = np.array([.8, .3, .1, .7, .9, .2, .7, .4, .8, .9, .2, .2])
    frame = pl.DataFrame({"user_id": users, "label": labels})
    loss = lambda p: -labels*np.log(p)-(1-labels)*np.log1p(-p)
    delta = loss(candidate)-loss(baseline)
    ordered_users = sorted(set(users))
    sums = np.array([delta[users == user].sum() for user in ordered_users])
    counts = np.array([(users == user).sum() for user in ordered_users])
    random = np.random.default_rng(2026)
    estimates = []
    for _ in range(300):
        draw = random.integers(0, len(sums), len(sums))
        estimates.append(sums[draw].sum()/counts[draw].sum())
    expected = np.quantile(estimates, [.025, .975])
    for order in [np.arange(len(users)), np.arange(len(users))[::-1]]:
        actual = independent_user_bootstrap(frame[order], baseline[order], candidate[order])
        np.testing.assert_allclose(actual["ci_95"], expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual["delta_logloss"], delta.mean(), rtol=0, atol=1e-15)
