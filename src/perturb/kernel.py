"""Step 1: choosing the width of the Gaussian kernel.

Neighbouring input features (e.g. adjacent wavelength bins) tend to move together, and the Pearson correlation between
two features, taken across a set of samples drawn from the prior, falls off with their separation. Fitting a Gaussian
to this fall-off gives us the FWHM of the perturbation window, i.e. the scale over which neighbouring features move
together in real data.
"""
import numpy as np
from scipy.optimize import curve_fit


def gaussian_window(x_axis, centre, fwhm):
    """The Gaussian window w(eps, m).

    The window is defined on the feature axis itself rather than on the bin index, so unevenly spaced features are
    handled properly.

    Parameters
    ----------
    x_axis : array_like
        The feature positions to evaluate the window at.
    centre : float or array_like
        The position of the peak, which broadcasts against x_axis.
    fwhm : float
        The full width at half maximum, in the units of x_axis.

    Returns
    -------
    ndarray
        The window, equal to 1 at the centre.
    """
    offset = np.asarray(x_axis, float) - centre
    return np.exp(-4 * np.log(2) * offset ** 2 / fwhm ** 2)


def minmax(X):
    """Min-max normalise each sample over its features.

    Parameters
    ----------
    X : ndarray, shape (..., M)
        The samples, one per row.

    Returns
    -------
    ndarray, shape (..., M)
        Each sample rescaled to run from 0 to 1. A flat sample comes out as NaN, as it cannot be rescaled.
    """
    lo = X.min(axis=-1, keepdims=True)
    hi = X.max(axis=-1, keepdims=True)
    return (X - lo) / (hi - lo)


def correlation_matrix(X, normalise=True):
    """Calculate the Pearson correlation of every pair of features across a set of samples.

    Parameters
    ----------
    X : array_like, shape (N, M)
        N samples of M features.
    normalise : bool, optional
        Min-max normalise each sample first. This is needed when the overall level of a sample (e.g. the depth of
        a spectrum) varies far more than its features do, as otherwise every pair of features would look
        correlated. Flat samples, which cannot be normalised, are dropped.

    Returns
    -------
    ndarray, shape (M, M)
        The correlation matrix.
    """
    X = np.asarray(X, float)
    if normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            X = minmax(X)
        X = X[np.all(np.isfinite(X), axis=1)]
    return np.corrcoef(X, rowvar=False)


def correlation_vs_separation(corr, x_axis, bin_width, min_count=5):
    """Bin the correlation of every pair of features by their separation.

    Parameters
    ----------
    corr : array_like, shape (M, M)
        The correlation matrix, from `correlation_matrix`.
    x_axis : array_like, shape (M,)
        The feature positions.
    bin_width : float
        The width of the separation bins, which are centred on zero.
    min_count : int, optional
        A bin needs more than this many pairs to be given an error bar.

    Returns
    -------
    dict
        Arrays over the non-empty bins, holding

        - ``separation``: the bin centres
        - ``rho``: the mean correlation
        - ``rho_err``: the standard error of the mean, which is NaN for bins with min_count pairs or fewer
        - ``n``: the number of pairs
        - ``fit``: bool, the bins that can be used in the fit (those with a finite, non-zero error)
    """
    x_axis = np.asarray(x_axis, float)
    sep = (x_axis[None, :] - x_axis[:, None]).ravel()
    rho = np.asarray(corr, float).ravel()

    keep = np.isfinite(rho)  # a constant feature has no correlation
    sep = sep[keep]
    rho = rho[keep]

    n_side = np.ceil(np.abs(sep).max() / bin_width)
    centres = np.arange(-n_side, n_side + 1) * bin_width
    n_bins = len(centres)
    idx = np.round(sep / bin_width).astype(int) + int(n_side)
    idx = np.clip(idx, 0, n_bins - 1)

    n = np.bincount(idx, minlength=n_bins)
    mean = np.bincount(idx, rho, n_bins) / np.maximum(n, 1)
    sum_sq = np.bincount(idx, (rho - mean[idx]) ** 2, n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        # we use the standard error rather than the standard deviation, as the number of pairs falls with separation
        std = np.sqrt(sum_sq / (n - 1))
        err = np.where(n > min_count, std / np.sqrt(n), np.nan)

    has = n > 0
    out = dict(separation=centres[has], rho=mean[has], rho_err=err[has], n=n[has])
    # the self-pairs (rho = 1, with no spread) fill the zero bin, but the Gaussian is 1 there for any FWHM anyway
    out["fit"] = np.isfinite(out["rho_err"]) & (out["rho_err"] > 0)
    return out


def fit_kernel_width(X, x_axis, bin_width=None, min_count=5, normalise=True, p0=None):
    """Fit the FWHM of the Gaussian kernel to a set of samples drawn from the prior.

    Parameters
    ----------
    X : array_like, shape (N, M)
        The samples from the prior.
    x_axis : array_like, shape (M,)
        The feature positions.
    bin_width : float, optional
        The width of the separation bins, which defaults to the median feature spacing.
    min_count : int, optional
        See `correlation_vs_separation`.
    normalise : bool, optional
        See `correlation_matrix`.
    p0 : float, optional
        A starting guess for the FWHM, which defaults to five bin widths.

    Returns
    -------
    dict
        The ``fwhm`` and ``fwhm_err``, the correlation matrix ``corr``, the binned curve ``binned`` (from
        `correlation_vs_separation`), and the ``bin_width`` and ``x_axis`` used. This is what `plotting.kernel_fit`
        and `plotting.correlation_matrix` expect.
    """
    x_axis = np.asarray(x_axis, float)
    if bin_width is None:
        bin_width = float(np.median(np.diff(np.sort(x_axis))))
    if p0 is None:
        p0 = 5 * bin_width

    corr = correlation_matrix(X, normalise)
    binned = correlation_vs_separation(corr, x_axis, bin_width, min_count)

    def model(separation, fwhm):
        return gaussian_window(separation, 0, fwhm)

    use = binned["fit"]
    popt, pcov = curve_fit(model, binned["separation"][use], binned["rho"][use], p0=[p0],
                           sigma=binned["rho_err"][use], absolute_sigma=True)

    return dict(
        fwhm=abs(float(popt[0])),
        fwhm_err=float(np.sqrt(pcov[0, 0])),
        corr=corr,
        binned=binned,
        bin_width=bin_width,
        x_axis=x_axis,
    )
