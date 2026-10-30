"""Tests against cases where we know the answer. Run with python -m pytest tests, or python tests/test_perturb.py."""
import numpy as np

import perturb as pt


def test_kernel_width_recovers_known_correlation():
    # white noise smoothed by a Gaussian of std s has a Gaussian correlation of std s * sqrt(2)
    rng = np.random.default_rng(0)
    x_axis = np.linspace(0, 10, 200)
    s = 0.2
    kernel = np.exp(-0.5 * ((x_axis[:, None] - x_axis[None, :]) / s) ** 2)
    X = rng.standard_normal((4000, 200)) @ kernel
    fit = pt.fit_kernel_width(X, x_axis, normalise=False)
    expected = 2 * np.sqrt(2 * np.log(2)) * s * np.sqrt(2)
    assert abs(fit["fwhm"] - expected) / expected < 0.05, (fit["fwhm"], expected)


def test_linear_black_box_importance_is_analytic():
    # for f(x) = L x the response is linear in xi, so importance_m = sigma_m L w_m exactly
    rng = np.random.default_rng(1)
    x_axis = np.linspace(0, 1, 30)
    sigma = rng.uniform(0.1, 0.3, 30)
    fwhm = 0.1
    L = rng.standard_normal((3, 30))
    perturber = pt.Perturber(lambda X: X @ L.T, x_axis, sigma, fwhm)
    result = perturber(rng.standard_normal(30), pt.magnitude_grid(2, 9))
    W = np.stack([pt.gaussian_window(x_axis, c, fwhm) for c in x_axis])
    assert np.allclose(result.importance, sigma[:, None] * W @ L.T)
    assert np.allclose(result.response[:, result.xi == 0], 0)


def test_linear_black_box_crossing_and_nonlinearity():
    # the response is exactly importance * xi, so there is no bend, and |response| reaches t at |xi| = t / |importance|
    x_axis, sigma, L, x = _linear_case()
    perturber = pt.Perturber(lambda X: X @ L.T, x_axis, sigma, 0.2)
    result = perturber(x, pt.magnitude_grid(2, 401))  # xi step 0.01
    assert np.allclose(result.nonlinearity, 0, atol=1e-12)
    t = np.array([0.3, 0.1])  # one threshold per output
    cross = result.crossing(t)
    exact = t / np.abs(result.importance)
    assert cross.shape == (12, 2, 2)
    assert np.array_equal(np.isnan(cross), np.repeat(exact[:, None] > 2, 2, axis=1))
    reached = exact <= 2
    for s, sign in ((0, -1), (1, 1)):
        c = cross[:, s][reached]
        overshoot = np.abs(c) - exact[reached]
        assert (np.sign(c) == sign).all()
        assert (overshoot >= -1e-9).all() and (overshoot <= 0.01 + 1e-9).all()  # at most one grid step late
    assert np.isnan(result.crossing(1e9)).all()


def test_fraction_fitted_of_exact_linear_retrieval():
    # for least squares through a linear forward model G, the fraction fitted is the same at every magnitude, so
    # every magnitude is in distribution, and a perturbation lying in G's span is reproduced exactly (a fraction of 1)
    rng = np.random.default_rng(2)
    x_axis = np.linspace(0, 1, 40)
    sigma = np.full(40, 0.1)
    fwhm = 0.1
    G = np.stack([pt.gaussian_window(x_axis, c, 0.3) for c in (0.2, 0.5, 0.8)])  # (3, 40)

    def forward(theta):
        return theta @ G

    def retrieve(X):
        return X @ np.linalg.pinv(G)

    x = forward(rng.uniform(size=3))
    result = pt.Perturber(retrieve, x_axis, sigma, fwhm)(x, pt.magnitude_grid(3, 7))
    F = pt.fraction_fitted(result, forward)
    assert np.allclose(F[:, result.xi != 0], F[:, result.xi > 0][:, :1])
    curve = pt.in_distribution_curve(result, forward)
    assert curve["inside"].all()
    assert pt.max_magnitude(curve) == 3

    # passing the same spectra in as your own best fits, in place of forward, should give the same fraction
    fits = forward(result.baseline + result.response)  # (centres, K, M)
    assert np.allclose(pt.fraction_fitted(result, best_fit=fits), F, equal_nan=True)
    for kwargs in ({}, {"best_fit": fits[:, 1:]}):
        try:
            pt.fraction_fitted(result, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} was accepted")

    # with a constant in the span (forward and retrieve both see the changed G) and a window so wide that it is
    # flat, the perturbation lies in the span
    G[0] = 1.0
    wide = pt.Perturber(retrieve, x_axis, sigma, fwhm=1e5)(rng.standard_normal(40), [-1.0, 0, 1])
    assert np.allclose(pt.fraction_fitted(wide, forward)[:, [0, 2]], 1, atol=1e-6)


def test_baseline_is_shared_at_zero_unless_rerun():
    # with a noisy f, by default xi = 0 is never run again (the response is exactly 0, with one baseline for every
    # centre), whereas rerun_zero=True runs it for every centre, where the noise then shows up
    x_axis, sigma, L, x = _linear_case()
    rng = np.random.default_rng(4)
    rows = []

    def noisy(X):
        rows.append(len(X))
        return X @ L.T + rng.normal(size=(len(X), 2))

    xi = pt.magnitude_grid(2, 5)
    shared = pt.Perturber(noisy, x_axis, sigma, 0.2, batch_size=7)(x, xi)
    assert (shared.response[:, xi == 0] == 0).all()
    assert sum(rows) == 1 + 12 * 4
    rows.clear()
    rerun = pt.Perturber(noisy, x_axis, sigma, 0.2, rerun_zero=True)(x, xi)
    assert sum(rows) == 1 + 12 * 5
    assert (rerun.response[:, xi == 0] != 0).all()


def test_scale_rejects_unknown_value():
    try:
        pt.perturbed_samples(np.zeros(5), 1.0, np.arange(5.0), 2, [1.0], 1.0, scale="center")
    except ValueError:
        return
    raise AssertionError("scale='center' was accepted")


def test_max_magnitude():
    xi = np.array([-3, -2, -1, 0, 1, 2, 3.0])
    assert pt.max_magnitude(dict(xi=xi, fraction_in=np.array([0, 1, 1, 1, 1, 0.5, 1.0]))) == 1
    assert pt.max_magnitude(dict(xi=xi, fraction_in=np.ones(7))) == 3
    assert pt.max_magnitude(dict(xi=xi, fraction_in=np.zeros(7))) == 0


def _linear_case(seed=3, M=12):
    rng = np.random.default_rng(seed)
    x_axis = np.linspace(0, 1, M)
    sigma = rng.uniform(0.1, 0.3, M)
    L = rng.standard_normal((2, M))
    x = rng.standard_normal(M)
    return x_axis, sigma, L, x


def test_degenerate_magnitudes_are_rejected():
    # max_magnitude() can return 0, and a grid of zeros has no slope to take, so this must fail loudly, not give NaN
    x_axis, sigma, L, x = _linear_case()
    perturber = pt.Perturber(lambda X: X @ L.T, x_axis, sigma, 0.2)
    for xi in (pt.magnitude_grid(0.0, 21), [0.0]):
        try:
            perturber(x, xi)
        except ValueError:
            continue
        raise AssertionError(f"xi = {xi} was accepted")
    assert np.isfinite(perturber(x, [-1, 0, 0, 1]).importance).all()  # repeats are dropped
    assert 0 in pt.magnitude_grid(3, 1)


def test_batches_match_one_call():
    # batch_size only splits up the work, so the response and the fraction fitted must not change
    x_axis, sigma, L, x = _linear_case()
    xi = pt.magnitude_grid(2, 7)
    calls = []

    def forward(theta):
        return theta @ np.linalg.pinv(L).T

    def f(X):
        calls.append(len(X))
        return X @ L.T

    whole = pt.Perturber(f, x_axis, sigma, 0.2, scale="own")(x, xi)
    calls.clear()
    batched = pt.Perturber(f, x_axis, sigma, 0.2, scale="own", batch_size=5)(x, xi)
    assert max(calls) <= 5
    assert np.allclose(batched.response, whole.response)

    F_batched = pt.fraction_fitted(whole, forward, batch_size=5)
    F_whole = pt.fraction_fitted(whole, forward)
    assert np.allclose(F_batched, F_whole, equal_nan=True)


def test_centres_follow_the_axis_and_save_round_trips(tmp_path=None):
    import tempfile
    from pathlib import Path

    import h5py

    x_axis, sigma, L, x = _linear_case()
    perturber = pt.Perturber(lambda X: X @ L.T, x_axis, sigma, 0.2, scale="own")
    result = perturber(x, pt.magnitude_grid(1, 5), centres=[5, 1, 9])
    assert list(result.centres) == [1, 5, 9]
    path = Path(tmp_path or tempfile.mkdtemp()) / "run.h5"
    result.save(path)
    with h5py.File(path) as h:  # this should be readable without perturb
        assert np.allclose(h["importance"][()], result.importance)
        assert h.attrs["scale"] == "own"

    back = pt.PerturbResult.load(path)
    assert back.scale == "own"
    assert back.fwhm == result.fwhm
    assert np.allclose(back.samples(), result.samples())
    assert np.allclose(back.response, result.response)


def test_normalise_ignores_offset_and_scale_and_drops_flat_samples():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((300, 20))
    shifted = X * rng.uniform(1, 50, (300, 1)) + rng.uniform(-100, 100, (300, 1))
    with_flat = np.vstack([shifted, np.ones((1, 20))])
    assert np.allclose(pt.correlation_matrix(with_flat), pt.correlation_matrix(X))


def test_every_plot_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from perturb import plotting

    rng = np.random.default_rng(5)
    x_axis = np.linspace(0, 1, 30)
    smoothing = np.exp(-0.5 * ((x_axis[:, None] - x_axis) / 0.05) ** 2)
    X_prior = rng.standard_normal((500, 30)) @ smoothing
    fit = pt.fit_kernel_width(X_prior, x_axis, normalise=False)
    L = rng.standard_normal((2, 30))

    def forward(theta):
        return theta @ np.linalg.pinv(L).T

    perturber = pt.Perturber(lambda X: X @ L.T, x_axis, 0.1, fit["fwhm"])
    result = perturber(X_prior[0], pt.magnitude_grid(2, 9))
    curve = pt.in_distribution_curve(result, forward)

    figures = [
        plotting.kernel_fit(fit),
        plotting.correlation_matrix(fit),
        plotting.in_distribution(curve),
        plotting.ood_map(curve, result),
        plotting.gradient_map(result),
        plotting.callouts(result, threshold=0.1),
        plotting.response_curves(result),
        plotting.importance(result),
    ]
    for fig in figures:
        fig.canvas.draw()
        plt.close(fig)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
