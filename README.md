![perturb](https://joolsclarke.co.uk/perturb/perturb-full-light.svg)

[Visit the Perturb webpage](https://joolsclarke.co.uk/perturb) 

Perturb is an interpretability framework for understanding the predictions made by black-box regression models, such
as machine learning retrievals. For a single sample, it tells you which input features the black-box's answer depends
on, how strongly, and in which direction. This repo packages the Gaussian perturbation method of Clarke, Yip &
Nikolaou, *Gaussian perturbation interpretability for regression techniques to understand retrieval black-boxes*
(RASTI 2026), so that it can be used on its own, along with a few additions of our own: a way of choosing how far to
perturb (step 2 below), and the non-linearity and threshold crossings of each response. The paper's own code,
including the TauREx-III experiments, is in `gaussian-perturbation-interpretability`.

Perturb is completely black-box agnostic. The black-box can be any function `f` that maps inputs `(N, M)` to outputs
`(N, P)`, whether that is a neural network, a Bayesian retrieval or a wrapper around some external code.

## What can Perturb tell me?

At its core, Perturb gives you the rate-of-change of the black-box's response to perturbing individual input
features, i.e. if I increase wavelength x, what happens to output y? This is useful for understanding which regions
of the input the black-box is sensitive to for each parameter, and so which input features are most important for
predicting which outputs.

As well as importance, the response tells you how robust the answer is to noise at an input feature level. By
reading it backwards, you can estimate a per-wavelength (or per-pixel) noise budget for an observation to achieve a
desired maximum posterior uncertainty (`examples/uncertainty_budget_demo.ipynb`). Importance can also be used to find
the parts of an observation that contribute very little, so that superfluous bins can be identified, which is useful
in both observation planning and instrument design (`examples/instrument_design_demo.ipynb`).

## Can I use Perturb on my data?

Perturb can be used on any 1D regression problem where neighbouring inputs are correlated, not just exoplanet
spectra. The Gaussian assumption may hold better or worse depending on the nature of your problem, but you should
expect at least some added stability over naive perturbation, which varies one input feature at a time.

Perturb does need to be able to sort the input features in order along an axis (e.g. wavelength). If two bins have
very different widths but similar centres, Perturb can still be used, but it won't be able to distinguish between
these bins well, and bins with exactly the same centre can't be told apart at all.

## About this package

I am a researcher, and not a software developer. I did not have the time to develop this library from scratch. This library was refactored
from the code used in the original implementation and documented with the help of a large language model (LLM). The LLM was used to assist
in code refactoring and documentation generation. However, the final implementation, testing, and validation were performed by myself (a 
human, last time I checked) to ensure that the implimentation is faithful to the paper. If you wish to use the code in your research, please
cite the original paper.

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Install

```
pip install perturb-interp    # numpy, scipy, matplotlib, h5py; then `import perturb`
```

The demo notebooks live in `examples/` and aren't part of the PyPI package, so clone the repo to run them:

```
pip install -e ".[examples]"  # + torch, jupyter, tqdm and ipywidgets, for the demo notebooks
```

## The four steps

For a single sample `x_star`, Perturb nudges a small, smooth region of the input up and down by a few times its
noise, and records how the black-box's answer moves. Doing this with the region centred on every feature in turn
gives a map of which parts of the input each output depends on, and in which direction.

1. **Choose the kernel width** (`perturb.fit_kernel_width`). Neighbouring features in real data tend to move
   together, so rather than perturbing one feature at a time, we perturb a Gaussian window of them. The width of this
   window is set from samples drawn from the prior, by calculating the correlation of every pair of features against
   their separation and fitting a Gaussian to it.
2. **Choose how far to perturb** (`perturb.in_distribution_curve`, `perturb.max_magnitude`). If a sample is pushed
   too far, the black-box is no longer seeing anything like the data it was built for, and its answers stop meaning
   much. To check for this, we feed each answer back through the forward model and measure how much of the
   perturbation the best fit reproduces. The largest magnitude at which the black-box still treats a perturbation the
   same way it treats a small one sets `xi_max`.
3. **Perturb the sample** (`perturb.Perturber`). Each centre is perturbed by every magnitude `xi` on a grid, giving
   the perturbed samples `X'_m = x* + sigma_m xi w(eps, m)`. Running these through the black-box gives the response,
   its gradient, and the importance (i.e. the mean of the gradient).
4. **Plot the results** (`perturb.plotting`).

Step 2 needs the forward model `g` (outputs -> inputs), i.e. the model whose parameters `f` retrieves. Each perturbed
sample's answer is fed back through `g`, and the fraction of the perturbation that this best fit reproduces is
compared with its value at the smallest magnitudes. We count a perturbed sample as in distribution while the two agree
to within `tol`. If you don't have a forward model, you can skip step 2 and choose `xi_max` yourself, as in
`examples/no_forward_model_demo.ipynb`.

By default the best fit is `g(f(X'))`, i.e. the forward model evaluated at the black-box's answer. However, if `f`
returns a posterior mean, this isn't necessarily the best-fitting spectrum (the maximum a posteriori spectrum from a
sampler would be, for example). In that case you can pass in your own spectra with `best_fit=`, as an array
`(centres, K, M)` whose `xi = 0` column is the best fit to `x_star`, using either
`pt.fraction_fitted(result, best_fit=fits)` or `pt.in_distribution_curve(result, best_fit=fits)`.

Putting it all together looks like this:

```python
import perturb as pt
from perturb import plotting

fit = pt.fit_kernel_width(X_prior, x_axis)                    # X_prior: (N, M) samples from the prior
wide = pt.Perturber(f, x_axis, sigma, fit["fwhm"])(x_star, pt.magnitude_grid(xi_max=10, n=81))  # explore
curve = pt.in_distribution_curve(wide, g, tol=0.1)
xi_max = pt.max_magnitude(curve)

result = pt.Perturber(f, x_axis, sigma, fit["fwhm"])(x_star, pt.magnitude_grid(xi_max))
result.response, result.gradient, result.importance          # (centres, K, P), (centres, K-1, P), (centres, P)
result.nonlinearity, result.crossing(threshold)              # (centres, P), (centres, 2, P): first xi below/above 0

plotting.kernel_fit(fit); plotting.in_distribution(curve); plotting.ood_map(curve, wide)
plotting.gradient_map(result, output=0); plotting.callouts(result, output=0); plotting.importance(result)
```

`result.nonlinearity` measures how far each response bends away from a straight line, and `result.crossing(threshold)`
finds the first magnitude on either side of zero at which the response reaches `threshold`. `plotting.callouts` draws
both of these on top of the gradient map.

If memory is a concern, `batch_size=` on `Perturber` and `in_distribution_curve` caps the number of rows passed to `f`
or `g` in each call, and with it the number of perturbed samples held in memory at once.

At `xi = 0` every centre's perturbed sample is just `x_star` itself, so `Perturber` doesn't run `f` there again.
Instead, the response is set to 0 and every centre shares the one baseline `f(x_star)`. This saves one call per
centre, and if `f` is stochastic (e.g. a sampler), it also keeps the run-to-run noise out of the gradient next to
`xi = 0`. If you would rather run `f` at `xi = 0` for every centre anyway, pass `rerun_zero=True`.

## Saving results, and black-boxes run elsewhere

`result.save("run.h5")` writes every array in a result to an HDF5 file (`x`, `x_axis`, `sigma`, `centres`, `xi`,
`baseline`, `response`, `gradient` and `importance`, with `fwhm` and `scale` saved as attributes). The file can be
read with plain `h5py` or any other HDF5 tool, so you don't need `perturb` installed to use it, and
`pt.PerturbResult.load("run.h5")` reads it back in.

Perturb can also be used to interpret slower, traditional retrieval codes, but expect inference to take much longer,
and to only be able to afford a lower response resolution (i.e. a coarser grid of magnitudes). If `f` is too slow to
call from Python (e.g. a retrieval running on a cluster), you can build the inputs, run them yourself, and fill in the
result directly:

```python
xi = pt.magnitude_grid(xi_max)
X = pt.perturbation_grid(x_star, sigma, x_axis, xi, fwhm)          # (M, K, M): every centre, every magnitude
nz = xi != 0                                                        # xi = 0 is x_star itself: no need to run it
Y = run_elsewhere(X[:, nz].reshape(-1, M)).reshape(M, nz.sum(), -1) # (M, K-1, P)
y0 = run_elsewhere(x_star[None])[0]                                 # (P,)
response = np.zeros((M, len(xi), len(y0)))
response[:, nz] = Y - y0
result = pt.PerturbResult(x=x_star, x_axis=x_axis, sigma=sigma, centres=np.arange(M), xi=xi, baseline=y0,
                          response=response, fwhm=fwhm)
```

Note that `xi` must be sorted in ascending order, and so must `x_axis` if you want to use the plots.

## Demos

`examples/fourier_demo.ipynb` (demo A) takes a CNN through every step above. The network is trained to recover the
amplitudes of a set of sine waves (of fixed frequency and phase) from a noisy signal, so we know what it *should* be
looking at, and sure enough, the importance of each amplitude comes out as its own sine wave. The network comes
pre-trained, with its weights in `examples/fourier/fourier_cnn.pt`.

`examples/no_forward_model_demo.ipynb` (demo A2) uses the same network, but assumes all you have is one sample and
`f`, so the kernel width and the largest magnitude are set by hand.

`examples/uncertainty_budget_demo.ipynb` (demo B) uses a black-box that isn't machine learning at all, a `scipy`
least-squares retrieval of overlapping Gaussian bands seen by 4 filters and 2 spectrographs, which fits one
observation at a time (`batch_size=1`). Here the perturbation is read backwards as a requirement. To constrain one
amplitude to a target error bar, `result.crossing(target)` gives the magnitude at which each region's response first
reaches the target, and from this we get the largest noise each data point can have, and the number of exposures each
instrument needs. This runs in under a minute.

`examples/instrument_design_demo.ipynb` (demo C) uses the same retrieval and instruments, but this time the importance
of each data point is used to reshape the instrument. Filters are dropped, and the ends of the spectrographs are
cropped and their interior bins merged, one point at a time, until any parameter has lost more than 50 per cent of its
precision. This takes a few minutes to run.
