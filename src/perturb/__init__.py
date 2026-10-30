"""Perturb: Gaussian perturbation interpretability for any black-box function.

The package follows the four steps of the method, with one module for each (and the perturbation split over two):

    kernel.py    step 1, choosing the kernel width from how quickly the correlation between features falls off
    ood.py       step 2, choosing how far to perturb, by checking how much of each perturbation the retrieval fits
    grid.py      step 3, building the magnitude grid and the perturbed samples
    perturb.py   step 3, running the black-box over the perturbed samples (Perturber) to get the response, its
                 gradient and the importance
    plotting.py  step 4, plotting the results
"""
from .grid import magnitude_grid, perturbation_grid, perturbed_samples
from .kernel import correlation_matrix, fit_kernel_width, gaussian_window
from .ood import fraction_fitted, in_distribution_curve, max_magnitude
from .perturb import PerturbResult, Perturber

__all__ = [
    "magnitude_grid", "perturbation_grid", "perturbed_samples",
    "correlation_matrix", "fit_kernel_width", "gaussian_window",
    "fraction_fitted", "in_distribution_curve", "max_magnitude",
    "PerturbResult", "Perturber",
]
