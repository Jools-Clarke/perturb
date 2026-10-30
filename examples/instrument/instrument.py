"""The instrument design demo.

A least-squares retrieval of overlapping Gaussian bands, observed by a choice of filters and spectrographs.

    depth(lam) = max(1 - exp(-sum_k a_k b_k(lam)), CLOUD_DECK),   b_k(lam) = sum_j h_kj exp(-(lam - mu_kj)^2 / 2 s_kj^2),
    a_k ~ U(A_LO, A_HI),   lam in [0, 10]

This is calculated on a fine native grid LAM. Each amplitude a_k scales a fixed band shape b_k, made of two or three
Gaussians in different places, each with its own width (COMPONENTS). The bands overlap, so neighbouring amplitudes are
partly degenerate, and the 1 - exp saturates where the bands are deep, so the retrieval is non-linear. A cloud deck
also sets a floor on the depth, below which the amplitudes have no effect.

The available instrumentation sits side by side along lam, and each element turns the native spectrum into data
points by averaging it, weighted by the element's transmission:

    4 filters         one point each, each with its own noise and its own skewed transmission profile; they are
                      unevenly spaced, overlap their neighbours, and can each be kept or dropped
    2 spectrographs   S1 (54 bins of 4 native pixels) and S2 (26 bins of 5), whose noise ramps up logarithmically
                      along each; each can be cropped from either end, and an interior bin merged with a neighbour

A configuration is a dict, {"filters": [filter indices], "S1": bin edges or None, "S2": bin edges or None}, where the
bin edges are native-pixel indices. The noise is drawn once on the native pixels and once per filter, so every
configuration sees the same noise realisation.

The black-box is retrieval(inst), which runs scipy's least_squares on one observation at a time.
"""
import numpy as np
from perturb import gaussian_window
from scipy.optimize import least_squares

LAM = np.round(np.arange(0, 10.0001, 0.02), 6)                 # the native grid
COMPONENTS = [  # the Gaussians of each amplitude, as (centre mu, width s, relative height h)
    [(4.2, 0.35, 1.0), (8.0, 0.15, 0.4)],
    [(3.2, 0.6, 0.6), (6.0, 0.3, 0.7), (8.0, 0.25, 1)],
    [(5.0, 0.3, 1.0), (7.2, 1, 0.2)],
    [(6.4, 0.5, 1.0), (9.1, 0.6, 0.6)],
    [(8.3, 0.9, 1.0), (1.5, 0.4, 0.5)],
]
A_LO, A_HI = 0.1, 2.0                                          # the prior on every amplitude

NAMES = []
for _k, _components in enumerate(COMPONENTS, 1):
    _centres = ", ".join(f"{mu:g}" for mu, _, _ in _components)
    NAMES.append(f"a{_k} (λ={_centres})")


def _band(components):
    """Build one amplitude's band shape b_k on LAM, as the sum of its Gaussians."""
    return sum(h * np.exp(-(LAM - mu) ** 2 / (2 * s ** 2)) for mu, s, h in components)


BANDS = np.array([_band(c) for c in COMPONENTS])               # (P, n_native)

FILTERS = [  # skewed Gaussian transmissions, as (peak, width on the blue side, width on the red side)
    (0.45, 0.16, 0.40),
    (1.15, 0.28, 0.63),
    (1.85, 0.18, 0.45),
    (6.8, 0.20, 0.40),
]
CLOUD_DECK = 0.2  # the depth of a cloud deck


def _skewed_gaussian(peak, blue_width, red_width):
    """Build one filter's transmission on LAM, as a Gaussian with a different width on either side of its peak."""
    offset = LAM - peak
    width = np.where(offset < 0, blue_width, red_width)
    return np.exp(-offset ** 2 / (2 * width ** 2))


FILTER_PROFILES = np.array([_skewed_gaussian(*f) for f in FILTERS])
FILTER_PROFILES[FILTER_PROFILES < 1e-3] = 0                    # finite support, so no filter reaches the spectrographs
FILTER_CENTRES = FILTER_PROFILES @ LAM / FILTER_PROFILES.sum(axis=1)  # effective wavelengths
FILTER_SIGMA = np.array([0.06, 0.05, 0.07, 0.04])
# for each spectrograph: the wavelength of its first native pixel, the number of bins, the native pixels per bin, and
# the noise of its first and last bin on the full instrument (ramping logarithmically between them)
SPECTROGRAPHS = {"S1": (2.2, 54, 4, 0.10, 0.20), "S2": (7.1, 26, 5, 0.08, 0.16)}

SIGMA_PIX = np.zeros(len(LAM))                                 # the noise of one native pixel
for _start, _n, _pix, _s0, _s1 in SPECTROGRAPHS.values():
    _p0 = int(round(_start / (LAM[1] - LAM[0])))
    SIGMA_PIX[_p0:_p0 + _n * _pix] = np.geomspace(_s0, _s1, _n * _pix) * np.sqrt(_pix)

_rng = np.random.default_rng(7)
NOISE_PIX = SIGMA_PIX * _rng.standard_normal(len(LAM))         # one realisation, shared by every configuration
NOISE_FILTER = FILTER_SIGMA * _rng.standard_normal(len(FILTER_CENTRES))


def sample_amplitudes(n, rng):
    """Draw amplitudes from the prior.

    Parameters
    ----------
    n : int
        Number of draws.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray, shape (n, P)
    """
    return rng.uniform(A_LO, A_HI, (n, len(COMPONENTS)))


def contribution(amplitudes):
    """Calculate the optical depth of each amplitude on its own, a_k b_k(lam).

    Parameters
    ----------
    amplitudes : array_like, shape (..., P)

    Returns
    -------
    ndarray, shape (..., P, n_native)
    """
    return np.asarray(amplitudes)[..., :, None] * BANDS


def cloud(amplitudes):
    """Return the depth of the cloud deck on its own.

    Parameters
    ----------
    amplitudes : array_like, shape (..., P)
        Ignored, as the deck is the same for any amplitudes. This is only taken so that it matches `contribution`.

    Returns
    -------
    ndarray, shape (n_native,)
        The depth, which broadcasts against the spectrum of any amplitudes.
    """
    return CLOUD_DECK * np.ones_like(LAM)


def depth(amplitudes):
    """Calculate the native spectrum, without the cloud deck.

    Parameters
    ----------
    amplitudes : array_like, shape (..., P)

    Returns
    -------
    ndarray, shape (..., n_native)
    """
    return 1 - np.exp(-contribution(amplitudes).sum(axis=-2))


def total_depth(amplitudes):
    """Calculate the native spectrum, with the cloud deck as its lower bound.

    Parameters
    ----------
    amplitudes : array_like, shape (..., P)

    Returns
    -------
    ndarray, shape (..., n_native)
    """
    return np.maximum(depth(amplitudes), cloud(amplitudes))


def native_edges(name):
    """Return the bin edges of a whole spectrograph.

    Parameters
    ----------
    name : {"S1", "S2"}

    Returns
    -------
    ndarray of int
        The native-pixel indices of the edges.
    """
    start, n, pix, _, _ = SPECTROGRAPHS[name]
    first_pixel = int(round(start / (LAM[1] - LAM[0])))
    return first_pixel + pix * np.arange(n + 1)


def full_config():
    """Return the full instrument, with every filter and both whole spectrographs.

    Returns
    -------
    dict
        A configuration (see the module docstring).
    """
    config = {"filters": list(range(len(FILTER_CENTRES)))}
    for name in SPECTROGRAPHS:
        config[name] = native_edges(name)
    return config


def instrument(config):
    """Work out the observation that a configuration makes.

    Parameters
    ----------
    config : dict
        A configuration (see the module docstring).

    Returns
    -------
    dict
        One entry per data point, sorted by wavelength:

        - ``B``: (M, n_native) the bandpass average that makes each point
        - ``x_axis``: (M,) the effective wavelength
        - ``sigma``: (M,) the noise
        - ``noise``: (M,) this realisation of the noise
        - ``element``: (M,) which element made the point, e.g. 'f3', 'S1' or 'S2'
    """
    rows, sigma, noise, element = [], [], [], []

    for i in config["filters"]:
        w = FILTER_PROFILES[i]
        rows.append(w / w.sum())
        sigma.append(FILTER_SIGMA[i])
        noise.append(NOISE_FILTER[i])
        element.append(f"f{i}")

    for name in SPECTROGRAPHS:
        e = config[name]
        if e is None:
            continue
        for lo, hi in zip(e[:-1], e[1:]):
            n_pix = hi - lo
            w = np.zeros(len(LAM))
            w[lo:hi] = 1 / n_pix
            rows.append(w)
            sigma.append(np.sqrt(np.sum(SIGMA_PIX[lo:hi] ** 2)) / n_pix)
            noise.append(w @ NOISE_PIX)
            element.append(name)

    B = np.array(rows).reshape(-1, len(LAM))
    x_axis = B @ LAM
    order = np.argsort(x_axis, kind="stable")
    return dict(
        B=B[order],
        x_axis=x_axis[order],
        sigma=np.array(sigma)[order],
        noise=np.array(noise)[order],
        element=np.array(element)[order],
    )


def observe(config, amplitudes):
    """Observe a spectrum with a configuration.

    Parameters
    ----------
    config : dict
        A configuration (see the module docstring).
    amplitudes : array_like, shape (P,)
        The true amplitudes.

    Returns
    -------
    dict
        The `instrument` dict, plus the observation ``y`` = B total_depth(a) + noise.
    """
    inst = instrument(config)
    inst["y"] = inst["B"] @ total_depth(amplitudes) + inst["noise"]
    return inst


def forward(inst):
    """The forward model g for one instrument.

    Parameters
    ----------
    inst : dict
        From `instrument` or `observe`.

    Returns
    -------
    callable
        g, which maps amplitudes (N, P) to noiseless observations (N, M).
    """
    def g(a):
        return total_depth(a) @ inst["B"].T

    return g


def _residual(a, B, sigma, y):
    """The normalised residual of the fit, for least_squares."""
    return (B @ total_depth(a) - y) / sigma


def _jacobian(a, B, sigma, y):
    """The Jacobian of `_residual` with respect to a, with shape (M, P)."""
    clear = depth(a) > cloud(a)  # where the cloud deck sets the depth, the amplitudes have no effect
    d_depth = (np.exp(-a @ BANDS) * clear)[:, None] * BANDS.T  # (n_native, P)
    return B @ d_depth / sigma[:, None]


def _fit(B, sigma, y, a0):
    """Run one least-squares fit of observation y, starting from a0."""
    # trf (the default with bounds) can also handle fewer points than parameters, which happens near the end of the
    # design loop. The tolerances are tight because the perturbations are a fraction of sigma, so a fit that stops
    # early would show up as noise in the response.
    return least_squares(_residual, a0, _jacobian, bounds=(A_LO, A_HI), x_scale=A_HI - A_LO, args=(B, sigma, y),
                         xtol=1e-12, ftol=1e-12, gtol=1e-12)


def retrieval(inst, n_starts=8, seed=0):
    """Build the black-box for one instrument.

    The instrument's own observation ``inst["y"]`` is fitted from n_starts random starting points, and the best fit
    is kept. Every other observation is warm-started from that solution, so that the perturbations are followed
    along the same branch of solutions rather than jumping between local minima.

    Parameters
    ----------
    inst : dict
        From `observe`.
    n_starts : int, optional
        The number of random starts for the first fit.
    seed : int, optional
        The seed for the random starts.

    Returns
    -------
    callable
        f, which maps observations (N, M) to amplitudes (N, P), fitting one row at a time. The best fit to
        ``inst["y"]`` is stored as ``f.best``.
    """
    B, sigma = inst["B"], inst["sigma"]
    starts = sample_amplitudes(n_starts, np.random.default_rng(seed))
    fits = [_fit(B, sigma, inst["y"], a0) for a0 in starts]
    best = min(fits, key=lambda fit: fit.cost).x

    def f(X):
        return np.array([_fit(B, sigma, x, best).x for x in np.atleast_2d(X)])

    f.best = best
    return f


def covariance(inst, a):
    """Calculate the retrieved covariance of the amplitudes.

    This is (J^T Sigma^-1 J + prior precision)^-1, taking the variance of the prior to be that of U(A_LO, A_HI).
    Including the prior keeps the covariance finite (at the width of the prior) for an amplitude that no data point
    can see.

    Parameters
    ----------
    inst : dict
        From `instrument` or `observe`.
    a : ndarray, shape (P,)
        The amplitudes to evaluate the Jacobian at.

    Returns
    -------
    ndarray, shape (P, P)
    """
    J = _jacobian(a, inst["B"], inst["sigma"], None)
    prior_precision = 12 / (A_HI - A_LO) ** 2  # 1 / variance of a uniform prior
    return np.linalg.inv(J.T @ J + np.eye(len(a)) * prior_precision)


def uncertainty(inst, a):
    """Calculate the retrieved 1 sigma uncertainty of each amplitude.

    Parameters
    ----------
    inst : dict
        From `instrument` or `observe`.
    a : ndarray, shape (P,)

    Returns
    -------
    ndarray, shape (P,)
    """
    return np.sqrt(np.diag(covariance(inst, a)))


def scores(result, sigma_retrieved):
    """Score every data point for every output.

    The score is |importance| in units of the output's retrieved sigma, per unit of data that the perturbation moved.
    The importance is the response to moving a whole kernel window, so it grows with the number of points inside
    that window. Dividing by the total weight of the window over the points makes a point in a dense spectrograph
    comparable with a lone filter.

    Parameters
    ----------
    result : perturb.PerturbResult
    sigma_retrieved : array_like, shape (P,)
        The retrieved uncertainty of each output, e.g. from `uncertainty`.

    Returns
    -------
    ndarray, shape (M, P)
    """
    window = gaussian_window(result.x_axis[None, :], result.centre_axis[:, None], result.fwhm)
    mass = window.sum(axis=1)
    return np.abs(result.importance) / sigma_retrieved / mass[:, None]


def next_point(inst, score, k=3):
    """Choose which point a design step removes, and how.

    Each output protects its k highest-scoring points, and the unprotected point with the lowest total score is
    removed. An interior spectrograph bin can only be merged with an unprotected neighbour, and picks whichever of
    these has the lower score. Once there are no unprotected points left that can be removed, protection is
    ignored, so the loop can run all the way down to no points at all.

    Parameters
    ----------
    inst : dict
        From `instrument`.
    score : ndarray, shape (M, P)
        From `scores`.
    k : int, optional
        The number of points each output protects.

    Returns
    -------
    m : int
        The index of the point.
    kind : {"drop", "crop", "merge"}
        "drop" removes a filter, "crop" removes an end bin of a spectrograph, and "merge" joins an interior bin with
        its neighbour.
    partner : int or None
        The neighbour to merge with, for "merge".
    """
    protected = set(np.argsort(-score, axis=0)[:k].ravel())
    total = score.sum(axis=1)

    def action(m, strict):
        el = inst["element"][m]
        if el.startswith("f"):
            return "drop", None
        same = list(np.flatnonzero(inst["element"] == el))  # this spectrograph's bins, in wavelength (= edge) order
        j = same.index(m)
        if j in (0, len(same) - 1):
            return "crop", None
        neighbours = [n for n in (same[j - 1], same[j + 1]) if not (strict and n in protected)]
        if not neighbours:
            return None
        return "merge", min(neighbours, key=lambda n: total[n])

    for strict in (True, False):
        for m in np.argsort(total, kind="stable"):
            if strict and m in protected:
                continue
            act = action(m, strict)
            if act:
                return (int(m), *act)


def design_step(config, inst, score, k=3):
    """Run one step of the design loop, removing the point that `next_point` picks.

    Parameters
    ----------
    config : dict
        The current configuration, which is not modified.
    inst : dict
        ``instrument(config)``.
    score : ndarray, shape (M, P)
        From `scores`.
    k : int, optional
        See `next_point`.

    Returns
    -------
    dict
        A new configuration with one point fewer.
    """
    m, kind, partner = next_point(inst, score, k)

    new = {}
    for key, value in config.items():
        if value is None:
            new[key] = None
        elif isinstance(value, np.ndarray):
            new[key] = value.copy()
        else:
            new[key] = list(value)

    el = inst["element"][m]
    if kind == "drop":
        new["filters"].remove(int(el[1:]))
        return new

    e = new[el]
    same = list(np.flatnonzero(inst["element"] == el))
    j = same.index(m)
    if kind == "crop":
        if len(e) <= 2:
            new[el] = None  # cropping the last bin removes the spectrograph
        elif j == 0:
            new[el] = e[1:]
        else:
            new[el] = e[:-1]
    else:
        new[el] = np.delete(e, max(j, same.index(partner)))  # remove the edge the two bins share
    return new
