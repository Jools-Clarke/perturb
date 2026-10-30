"""Step 3: building the perturbation grid.

For a sample x* with noise sigma, we perturb the region around feature m by adding a Gaussian window w of FWHM eps,
scaled by each magnitude xi_k in turn (in units of sigma_m),

    X'_m = 1_K x* + sigma_m xi w(eps, m).

This gives one perturbed copy of x* per magnitude. Everything here is pure numpy.
"""
import numpy as np

from .kernel import gaussian_window


def magnitude_grid(xi_max=3.0, n=101):
    """Build the grid of perturbation magnitudes.

    Parameters
    ----------
    xi_max : float, optional
        The largest magnitude, in units of sigma. The grid runs from -xi_max to xi_max.
    n : int, optional
        The number of magnitudes. An even n is bumped up by one, and anything below 3 becomes 3, so that xi = 0 is
        always on the grid.

    Returns
    -------
    ndarray, shape (K,)
        The magnitudes, linearly spaced.
    """
    if n % 2 == 0:
        n += 1
    n = max(n, 3)
    return np.round(np.linspace(-xi_max, xi_max, n), 10)


def perturbation_rows(x, sigma, x_axis, xi, fwhm, scale="centre", centres=None, rows=None):
    """Build some or all of the rows of the flattened perturbation grid.

    The full grid has shape (n_centres * K, M), where row ``j * K + k`` is centre j perturbed by magnitude xi_k.
    Asking for only a subset of rows means the grid can be built a batch at a time, rather than all at once.

    Parameters
    ----------
    x : array_like, shape (M,)
        The sample being perturbed.
    sigma : float or array_like, shape (M,)
        The noise on each feature of x.
    x_axis : array_like, shape (M,)
        The position of each feature (e.g. wavelength).
    xi : array_like, shape (K,)
        The perturbation magnitudes, in units of sigma.
    fwhm : float
        The FWHM of the Gaussian window, in the units of x_axis.
    scale : {"centre", "own"}, optional
        How each feature in the window is scaled. "centre" moves every feature by the noise of the centre feature,
        which keeps the smooth shape of the window. "own" moves each feature by its own noise instead, so that the
        perturbation follows the noise from one feature to the next. This is worth using if the noise jumps sharply
        between neighbouring features.
    centres : array_like of int, optional
        The indices of the centre features. By default every feature is used.
    rows : array_like of int, optional
        The flat row indices to build. By default every row is built.

    Returns
    -------
    ndarray, shape (len(rows), M)
        The perturbed samples.

    Raises
    ------
    ValueError
        If scale is not "centre" or "own".
    """
    if scale not in ("centre", "own"):
        raise ValueError(f"scale must be 'centre' or 'own', not {scale!r}")

    x = np.asarray(x, float)
    x_axis = np.asarray(x_axis, float)
    xi = np.atleast_1d(np.asarray(xi, float))
    sigma = np.broadcast_to(np.asarray(sigma, float), x.shape)

    if centres is None:
        centres = np.arange(len(x))
    else:
        centres = np.atleast_1d(centres)

    if rows is None:
        rows = np.arange(len(centres) * len(xi))
    else:
        rows = np.asarray(rows)

    # split each flat row index into its centre and its magnitude
    j, k = np.divmod(rows, len(xi))
    m = centres[j]

    window = gaussian_window(x_axis[None, :], x_axis[m][:, None], fwhm)
    if scale == "centre":
        s = sigma[m][:, None]
    else:
        s = sigma[None, :]
    return x[None, :] + xi[k][:, None] * s * window


def perturbed_samples(x, sigma, x_axis, centre_index, xi, fwhm, scale="centre"):
    """Perturb x around a single centre feature m, giving X'_m.

    Parameters
    ----------
    x, sigma, x_axis, xi, fwhm, scale
        As in `perturbation_rows`.
    centre_index : int
        The index of the centre feature m.

    Returns
    -------
    ndarray, shape (K, M)
        One perturbed copy of x per magnitude.
    """
    return perturbation_rows(x, sigma, x_axis, xi, fwhm, scale, [centre_index])


def perturbation_grid(x, sigma, x_axis, xi, fwhm, scale="centre", centres=None):
    """Perturb x around every centre feature.

    Parameters
    ----------
    x, sigma, x_axis, xi, fwhm, scale, centres
        As in `perturbation_rows`.

    Returns
    -------
    ndarray, shape (n_centres, K, M)
        The output of `perturbed_samples` for each centre in turn.
    """
    if centres is None:
        n_centres = len(x)
    else:
        n_centres = len(np.atleast_1d(centres))
    rows = perturbation_rows(x, sigma, x_axis, xi, fwhm, scale, centres)
    return rows.reshape(n_centres, -1, len(x))
