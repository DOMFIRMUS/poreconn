import numpy as np

from poreconn.budget import fraction_below, mass_balance, threshold_for


def _hist(values, lo=0.0, hi=100.0, n=1000):
    h, e = np.histogram(values, bins=n, range=(lo, hi))
    return 0.5 * (e[:-1] + e[1:]), h


def test_mass_balance_two_phases_with_partial_volume():
    # pore = 10, solid = 90; 30 % pure pore, 10 % half-pore voxels at 50: porosity 0.35 under linear mixing
    rng = np.random.default_rng(0)
    v = np.r_[np.full(30000, 10.0), np.full(10000, 50.0), np.full(60000, 90.0)] + rng.normal(0, 1.0, 100000)
    c, h = _hist(v)
    assert abs(mass_balance(c, h, 10.0, 90.0) - 0.35) < 2e-3
    # ISO-50 (threshold 50) splits the half-pore voxels evenly: 0.30 + 0.05
    assert abs(fraction_below(c, h, 50.0) - 0.35) < 5e-3


def test_threshold_inverts_fraction_below():
    rng = np.random.default_rng(1)
    c, h = _hist(rng.uniform(0, 100, 200000))
    for phi in (0.05, 0.3, 0.77):
        t = threshold_for(c, h, phi)
        assert abs(t - 100 * phi) < 0.5
        assert abs(fraction_below(c, h, t) - phi) < 1e-3


def test_threshold_with_empty_bins():
    v = np.r_[np.full(500, 20.0), np.full(1500, 80.0)]
    c, h = _hist(v)
    t = threshold_for(c, h, 0.25)
    assert 20.0 <= t <= 80.0
    assert abs(fraction_below(c, h, t) - 0.25) < 1e-6
