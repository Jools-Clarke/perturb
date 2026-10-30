"""Step 2: how far can a sample be perturbed before it leaves the distribution? We ask the retrieval itself.

For each perturbed sample, we feed f's answer back through the forward model g (outputs -> inputs), and compare this
best fit with the perturbation that was applied. The fraction fitted,

    F = dfit . dx / dx . dx    (in units of the noise),

is the fraction of the perturbation dx that the best fit reproduces. By default the best fit is g(f(X')), but you can
pass in your own best-fit spectra as best_fit instead (e.g. the maximum a posteriori spectra from a sampler).

At small magnitudes, F is the retrieval's linear response to the perturbation. We count a perturbed sample as 'in
distribution' while F stays within tol of that value, i.e. while the retrieval is still treating the perturbation the
same way it treats a small one (`in_distribution_curve`). Everything here works from the PerturbResult, so f never
needs to be called again.
"""
import numpy as np


def fraction_fitted(result, forward=None, batch_size=None, best_fit=None):
    """Calculate the fraction of each perturbation that the retrieval's best fit reproduces.

    With dx = X' - x* and dfit = g(f(X')) - g(f(x*)), both in units of sigma, F = dfit . dx / dx . dx. F is 1 when
    the best fit follows the perturbation exactly, and 0 when the retrieval treats it as noise.

    By default the best fit is g(f(X')), the forward model at f's answer. However, if f returns a posterior mean,
    this is not necessarily the best-fitting spectrum (the maximum a posteriori spectrum from a sampler would be, for
    example). In that case, pass those spectra in as best_fit instead.

    Parameters
    ----------
    result : PerturbResult
        The perturbation run.
    forward : callable, optional
        The forward model g, which maps outputs (N, P) to inputs (N, M). This is needed unless best_fit is given.
    batch_size : int, optional
        The maximum number of rows per call to g, which also caps how many perturbed samples are held in memory at
        once. By default everything is done in one call.
    best_fit : array_like, shape (n_centres, K, M), optional
        Your own best-fit spectrum for every perturbed sample, used in place of g(f(X')). The xi = 0 column must
        hold the best fit to x* itself, which is the same for every centre, so only the first centre's is used.

    Returns
    -------
    ndarray, shape (n_centres, K)
        The fraction fitted, which is NaN at xi = 0.

    Raises
    ------
    ValueError
        If neither forward nor best_fit is given, or if best_fit has the wrong shape or no xi = 0 column.
    """
    n_outputs = result.response.shape[-1]
    params = (result.baseline + result.response).reshape(-1, n_outputs)
    if best_fit is not None:
        best_fit = np.asarray(best_fit, float)
        if best_fit.shape != result.response.shape[:2] + result.x.shape:
            raise ValueError(f"best_fit must have shape (n_centres, K, M) = "
                             f"{result.response.shape[:2] + result.x.shape}, not {best_fit.shape}")
        if not (result.xi == 0).any():
            raise ValueError("best_fit needs xi = 0 on the grid, as that column is the best fit to x* itself")
        fit0 = best_fit[0, result.xi == 0][0]
        best_fit = best_fit.reshape(len(params), -1)
    elif forward is None:
        raise ValueError("either forward or best_fit must be given")
    else:
        fit0 = np.asarray(forward(result.baseline[None]))[0]

    n = len(params)
    batch = batch_size or n
    F = np.empty(n)
    for start in range(0, n, batch):
        rows = np.arange(start, min(start + batch, n))
        dx = (result.samples(rows) - result.x) / result.sigma
        fit = best_fit[rows] if best_fit is not None else np.asarray(forward(params[rows]))
        dfit = (fit - fit0) / result.sigma
        with np.errstate(invalid="ignore", divide="ignore"):
            F[rows] = np.sum(dfit * dx, axis=-1) / np.sum(dx * dx, axis=-1)
    return F.reshape(result.response.shape[:2])


def in_distribution_curve(result, forward=None, tol=0.1, batch_size=None, best_fit=None):
    """Find which of the perturbed samples are still in distribution.

    A perturbed sample is in distribution if its fraction fitted is within tol of the value at the smallest non-zero
    magnitudes (+-xi_min) for the same centre.

    Parameters
    ----------
    result : PerturbResult
        The perturbation run.
    forward : callable, optional
        The forward model g (see `fraction_fitted`). This is needed unless best_fit is given.
    tol : float, optional
        How far the fraction fitted is allowed to drift before a sample counts as out of distribution.
    batch_size : int, optional
        See `fraction_fitted`.
    best_fit : array_like, shape (n_centres, K, M), optional
        Your own best-fit spectra, used in place of forward (see `fraction_fitted`).

    Returns
    -------
    dict
        - ``xi``: (K,) the magnitudes
        - ``values``: (n_centres, K) the fraction fitted
        - ``inside``: (n_centres, K) bool, whether each sample is in distribution
        - ``fraction_in``: (K,) the fraction of centres inside at each magnitude
        - ``tol``: the tolerance used

        `max_magnitude` reads the limit on |xi| from this.
    """
    xi = result.xi
    values = fraction_fitted(result, forward, batch_size, best_fit)

    nonzero = np.flatnonzero(xi != 0)
    abs_xi = np.abs(xi[nonzero])
    smallest = nonzero[abs_xi == abs_xi.min()]
    ref = np.nanmean(values[:, smallest], axis=1, keepdims=True)

    inside = (np.abs(values - ref) <= tol) | (xi == 0)
    return dict(xi=xi, values=values, inside=inside, fraction_in=inside.mean(axis=0), tol=tol)


def max_magnitude(curve, level=0.95):
    """Find the largest |xi| that keeps enough of the perturbed samples in distribution.

    At every magnitude up to the returned value, on both sides of zero, at least `level` of the perturbed samples
    must be in distribution.

    Parameters
    ----------
    curve : dict
        From `in_distribution_curve`.
    level : float, optional
        The fraction of samples required to be in distribution.

    Returns
    -------
    float
        The limit on |xi|, or 0 if even the smallest magnitude fails.
    """
    abs_xi = np.abs(curve["xi"])
    ok = curve["fraction_in"] >= level
    if ok.all():
        return float(abs_xi.max())

    first_failure = abs_xi[~ok].min()
    below = abs_xi[abs_xi < first_failure]
    if below.size == 0:
        return 0.0
    return float(below.max())
