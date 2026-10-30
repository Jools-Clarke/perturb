"""The demo problem, recovering the amplitudes of n_freq sine waves of fixed frequency and phase from a noisy signal.

    x(t) = sum_k a_k sin(2 pi f_k t + phi_k) + noise,   a_k ~ U(0, 1),   t in [0, 1] on 100 points

The noise sigma(t) grows along t, standing in for the uncertainty of an instrument. FourierCNN is the black-box, which
we trained on this problem with fresh noise every epoch, and its weights ship as fourier_cnn.pt. It has no dropout, so
it gives the same answer every time it is called.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

T = np.linspace(0.0, 1.0, 100)
FREQUENCIES = np.arange(1, 9)  # cycles per unit t
PHASES = np.random.default_rng(10).uniform(0, 2 * np.pi, len(FREQUENCIES))
BASIS = np.sin(2 * np.pi * FREQUENCIES[:, None] * T[None, :] + PHASES[:, None])  # (n_freq, 100)
SIGMA = 1 + 0.25 * T  # noise per point
NAMES = [f"a{k} ({f} Hz)" for k, f in enumerate(FREQUENCIES, 1)]
WEIGHTS = Path(__file__).resolve().parent / "fourier_cnn.pt"


def sample_amplitudes(n, rng):
    """Draw amplitudes from the prior.

    Parameters
    ----------
    n : int
        The number of draws.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray, shape (n, n_freq)
    """
    return rng.uniform(0, 1, (n, len(FREQUENCIES)))


def contribution(amplitudes):
    """Calculate each sine wave on its own, a_k sin(2 pi f_k t + phi_k).

    Parameters
    ----------
    amplitudes : array_like, shape (..., n_freq)

    Returns
    -------
    ndarray, shape (..., n_freq, 100)
    """
    return np.asarray(amplitudes)[..., :, None] * BASIS


def forward(amplitudes):
    """Calculate the noiseless signal, i.e. the sum of the contributions.

    Parameters
    ----------
    amplitudes : array_like, shape (..., n_freq)

    Returns
    -------
    ndarray, shape (..., 100)
    """
    return contribution(amplitudes).sum(axis=-2)


class FourierCNN(nn.Module):
    """A CNN that maps a signal of 100 points to n_freq amplitudes.

    It has three Conv1d layers (with 32, 32 and 64 filters and a kernel of 5), followed by dense layers of 256 and
    n_freq units.

    Parameters
    ----------
    n_points : int, optional
        The length of the input signal.
    n_out : int, optional
        The number of amplitudes to predict.
    """

    def __init__(self, n_points=len(T), n_out=len(FREQUENCIES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (n_points // 4), 256), nn.ReLU(),  # two pooling layers quarter the length
            nn.Linear(256, n_out),
        )

    def forward(self, x):
        """Map signals (N, n_points) to amplitudes (N, n_out)."""
        return self.head(self.features(x.unsqueeze(1)))


def setup():
    """Describe the problem the network was trained for.

    This is stored alongside the weights, so that they can't be loaded into a different setup by mistake.

    Returns
    -------
    dict of torch.Tensor
        ``frequencies``, ``phases`` and ``sigma``.
    """
    return {
        "frequencies": torch.tensor(FREQUENCIES),
        "phases": torch.tensor(PHASES),
        "sigma": torch.tensor(SIGMA),
    }


def load_black_box(path=WEIGHTS):
    """Load the trained network as a plain numpy function.

    Parameters
    ----------
    path : Path, optional
        The weights file.

    Returns
    -------
    callable
        f, which maps signals (N, 100) to amplitudes (N, n_freq).

    Raises
    ------
    ValueError
        If the weights were trained on a different setup from the one in this module.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in setup().items():
        if not torch.equal(ckpt[key], value):
            raise ValueError(f"{path.name} was trained with a different {key}, "
                             "and the shipped weights only fit the setup in fourier.py")
    model = FourierCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def f(x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        with torch.no_grad():
            return model(x).numpy()

    return f
