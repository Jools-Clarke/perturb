"""Step 3: perturbing one sample through a black-box.

The quantities involved are:

    f             any callable mapping inputs (N, M) to outputs (N, P) (or (N,) for one output), e.g. a retrieval
    response      R_m = f(X'_m) - f(x*)                                                       (n_centres, K, P)
    gradient      the slope of the response between neighbouring magnitudes                   (n_centres, K - 1, P)
    importance    the mean of those slopes, per feature and output                            (n_centres, P)
    nonlinearity  how far the response bends from the straight line importance * xi          (n_centres, P)
    crossing(t)   the first magnitude either side of zero at which |response| reaches t       (n_centres, 2, P)
"""
from dataclasses import asdict, dataclass, fields

import h5py
import numpy as np

from .grid import magnitude_grid, perturbation_rows


@dataclass
class PerturbResult:
    """The response of f to every perturbation of one sample x.

    The magnitudes xi are sorted in ascending order, and the centres by their position on x_axis. If f was run
    elsewhere (e.g. on a cluster), you can build the inputs with `grid.perturbation_grid` and fill this in directly,
    with response = f(X') - f(x) reshaped to (n_centres, K, P).

    Attributes
    ----------
    x : ndarray, shape (M,)
        The sample.
    x_axis : ndarray, shape (M,)
        Its feature axis (e.g. wavelength).
    sigma : ndarray, shape (M,)
        Its noise.
    centres : ndarray of int, shape (n_centres,)
        The indices of the perturbed centre features.
    xi : ndarray, shape (K,)
        The magnitudes, in units of the centre feature's sigma.
    baseline : ndarray, shape (P,)
        f(x), the unperturbed answer.
    response : ndarray, shape (n_centres, K, P)
        f(X') - f(x).
    fwhm : float
        The FWHM of the perturbation window.
    scale : {"centre", "own"}
        See `grid.perturbation_rows`.
    """

    x: np.ndarray
    x_axis: np.ndarray
    sigma: np.ndarray
    centres: np.ndarray
    xi: np.ndarray
    baseline: np.ndarray
    response: np.ndarray
    fwhm: float
    scale: str = "centre"

    @property
    def centre_axis(self):
        """ndarray, shape (n_centres,): the position of each centre on x_axis."""
        return self.x_axis[self.centres]

    @property
    def gradient(self):
        """ndarray, shape (n_centres, K - 1, P): the rate of change of the response with xi, between neighbours."""
        step = np.diff(self.xi)[None, :, None]
        return np.diff(self.response, axis=1) / step

    @property
    def importance(self):
        """ndarray, shape (n_centres, P): the mean of the K - 1 slopes, in output units per sigma."""
        return self.gradient.mean(axis=1)

    @property
    def nonlinearity(self):
        """ndarray, shape (n_centres, P): the RMS distance of each response from the line importance * xi.

        This is in output units. It is 0 for a linear response, and is never large for a column that is just noise.
        """
        line = self.importance[:, None, :] * self.xi[None, :, None]
        return np.sqrt(np.mean((self.response - line) ** 2, axis=1))

    def crossing(self, threshold):
        """Find the first magnitude, working out from zero, at which |response| reaches a threshold.

        The answer is always a grid point, so it can overshoot the true crossing by up to one step in xi.

        Parameters
        ----------
        threshold : float or array_like, shape (P,)
            The threshold in output units, either one per output or shared by all of them.

        Returns
        -------
        ndarray, shape (n_centres, 2, P)
            ``[:, 0]`` is the crossing below zero and ``[:, 1]`` the one above. This is NaN wherever the response
            never reaches the threshold.
        """
        hit = np.abs(self.response) >= np.asarray(threshold, float)
        out = np.full((len(self.centres), 2, self.response.shape[-1]), np.nan)

        for s, side in enumerate((self.xi < 0, self.xi > 0)):
            # this side's magnitudes, ordered working out from zero
            idx = np.flatnonzero(side)
            order = idx[np.argsort(np.abs(self.xi[side]), kind="stable")]
            if order.size == 0:
                continue
            h = hit[:, order]
            first = self.xi[order][h.argmax(axis=1)]
            out[:, s] = np.where(h.any(axis=1), first, np.nan)
        return out

    def samples(self, rows=None):
        """Recompute the perturbed inputs X', rather than storing them.

        Parameters
        ----------
        rows : array_like of int, optional
            Flat row indices into the (n_centres * K, M) grid, where row ``j * K + k`` is centre j at xi_k. By
            default every row is returned.

        Returns
        -------
        ndarray, shape (len(rows), M)
        """
        return perturbation_rows(self.x, self.sigma, self.x_axis, self.xi, self.fwhm, self.scale, self.centres, rows)

    def save(self, path):
        """Write the result to an HDF5 file.

        The file can be read without this package, as every field is saved as a dataset (apart from fwhm and scale,
        which are attributes), and the gradient and importance are saved alongside them.

        Parameters
        ----------
        path : str or Path
            The file to write, which is overwritten if it already exists.
        """
        data = asdict(self)
        data["gradient"] = self.gradient
        data["importance"] = self.importance
        with h5py.File(path, "w") as h:
            for key, value in data.items():
                if key in ("fwhm", "scale"):
                    h.attrs[key] = value
                else:
                    h[key] = value

    @classmethod
    def load(cls, path):
        """Read back a file written by `save`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        PerturbResult
        """
        kwargs = {}
        with h5py.File(path, "r") as h:
            for field in fields(cls):
                if field.name in h.attrs:
                    kwargs[field.name] = h.attrs[field.name]
                else:
                    kwargs[field.name] = h[field.name][()]
        kwargs["fwhm"] = float(kwargs["fwhm"])
        kwargs["scale"] = str(kwargs["scale"])
        return cls(**kwargs)


class Perturber:
    """Gaussian perturbation of samples through a black-box f.

    Parameters
    ----------
    f : callable
        The black-box, mapping inputs (N, M) to outputs (N, P) or (N,).
    x_axis : array_like, shape (M,)
        The position of each feature. The kernel FWHM is measured along this axis (see `kernel.fit_kernel_width`).
    sigma : float or array_like, shape (M,)
        The noise on the samples.
    fwhm : float
        The FWHM of the perturbation window.
    scale : {"centre", "own"}, optional
        See `grid.perturbation_rows`.
    batch_size : int, optional
        The maximum number of rows per call to f, which also caps how many perturbed inputs are held in memory at
        once. By default all n_centres * K rows go in one call.
    rerun_zero : bool, optional
        At xi = 0 every centre's perturbed sample is just x itself, so by default f is not run there again. Instead,
        the response is set to 0 and every centre shares the one baseline f(x). This saves n_centres calls, and for
        a stochastic f (e.g. a sampler) it also keeps the run-to-run noise out of the gradient next to xi = 0. Set
        this to True to run f at xi = 0 for every centre anyway.
    """

    def __init__(self, f, x_axis, sigma, fwhm, scale="centre", batch_size=None, rerun_zero=False):
        self.f = f
        self.fwhm = fwhm
        self.scale = scale
        self.batch_size = batch_size
        self.rerun_zero = rerun_zero
        self.x_axis = np.asarray(x_axis, float)
        self.sigma = np.broadcast_to(np.asarray(sigma, float), self.x_axis.shape).copy()

    def evaluate(self, X):
        """Run f over the rows of X, in batches.

        Parameters
        ----------
        X : array_like, shape (N, M)

        Returns
        -------
        ndarray, shape (N, P)
        """
        X = np.asarray(X, float)
        batch = self.batch_size or len(X)
        outputs = []
        for start in range(0, len(X), batch):
            chunk = X[start:start + batch]
            y = np.asarray(self.f(chunk), float)
            outputs.append(y.reshape(len(chunk), -1))
        return np.concatenate(outputs)

    def __call__(self, x, xi=None, centres=None):
        """Perturb one sample at every magnitude, around every centre.

        Parameters
        ----------
        x : array_like, shape (M,)
            The sample.
        xi : array_like, optional
            The magnitudes, in units of sigma. Any duplicates are dropped. By default this is `magnitude_grid()`,
            i.e. 101 values on [-3, 3].
        centres : array_like of int, optional
            The indices of the centre features. By default all M are used.

        Returns
        -------
        PerturbResult

        Raises
        ------
        ValueError
            If fewer than two distinct magnitudes are given.
        """
        x = np.asarray(x, float)
        if xi is None:
            xi = magnitude_grid()
        xi = np.unique(np.asarray(xi, float))
        if len(xi) < 2:
            raise ValueError(f"need at least 2 distinct magnitudes, got xi = {xi} (did max_magnitude() return 0?)")

        if centres is None:
            centres = np.arange(len(x))
        centres = np.atleast_1d(centres)
        centres = centres[np.argsort(self.x_axis[centres], kind="stable")]

        result = PerturbResult(x=x, x_axis=self.x_axis, sigma=self.sigma, centres=centres, xi=xi,
                               baseline=self.evaluate(x[None])[0], response=None, fwhm=self.fwhm, scale=self.scale)

        rows = np.arange(len(centres) * len(xi))
        if not self.rerun_zero:
            rows = rows[xi[rows % len(xi)] != 0]
        batch = self.batch_size or len(rows)
        outputs = []
        for start in range(0, len(rows), batch):
            outputs.append(self.evaluate(result.samples(rows[start:start + batch])))
        response = np.zeros((len(centres) * len(xi), len(result.baseline)))
        response[rows] = np.concatenate(outputs) - result.baseline
        result.response = response.reshape(len(centres), len(xi), -1)
        return result
