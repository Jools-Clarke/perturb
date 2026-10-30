"""Step 4: plotting the results, for any problem.

Every function here returns the matplotlib Figure, so you can adjust or save it afterwards.

    kernel_fit           the correlation against feature separation, and the Gaussian fit that sets the FWHM
    correlation_matrix   the correlation of every pair of features across the prior samples
    in_distribution      the fraction of perturbed samples still in distribution at each magnitude, with xi_max
    ood_map              the fraction fitted for every centre and magnitude, with the in-distribution boundary
    gradient_map         the response gradient over every centre and magnitude
    callouts             the gradient map, with the steepest positive, steepest negative and most non-linear columns
                         drawn out as response curves, and optional hatched threshold bands
    response_curves      ceteris paribus-style response curves for single centres
    importance           the importance of every feature, for each output
"""
import matplotlib.pyplot as plt
import numpy as np

from .kernel import gaussian_window
from .ood import max_magnitude


def edges(centres):
    """Find the cell edges around a set of cell centres.

    Parameters
    ----------
    centres : array_like, shape (n,)
        The sorted cell centres, which need not be evenly spaced.

    Returns
    -------
    ndarray, shape (n + 1,)
        Edges halfway between neighbouring centres, with the outer cells mirrored. A single centre is given a cell
        of width 1.
    """
    c = np.asarray(centres, float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    first = 2 * c[0] - mid[0]
    last = 2 * c[-1] - mid[-1]
    return np.concatenate([[first], mid, [last]])


def _label(names, p):
    """The name of output p, or a generic one if no names were given."""
    if names is None:
        return f"output {p}"
    return names[p]


def _colour_limit(values, vmax):
    """A symmetric colour limit, which is vmax if given, or otherwise the 99th percentile of |values| (or 1)."""
    if vmax:
        return vmax
    return np.nanpercentile(np.abs(values), 99) or 1


def kernel_fit(fit, xlabel="separation"):
    """Plot the kernel width fit.

    The top panel shows the binned correlation against separation (with standard errors), the bins excluded from
    the fit, and the fitted Gaussian with its 3 sigma band. The bottom panel shows the residuals.

    Parameters
    ----------
    fit : dict
        From `kernel.fit_kernel_width`.
    xlabel : str, optional
        The name of the separation axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    binned = fit["binned"]
    fwhm = fit["fwhm"]
    fwhm_err = fit["fwhm_err"]
    x = binned["separation"]
    y = binned["rho"]
    yerr = binned["rho_err"]
    ok = binned["fit"]

    xx = np.linspace(x.min(), x.max(), 400)
    model = gaussian_window(xx, 0, fwhm)

    # the 1 sigma band, from propagating the FWHM uncertainty through a central finite difference
    h = 1e-6 * fwhm
    dmodel_dfwhm = (gaussian_window(xx, 0, fwhm + h) - gaussian_window(xx, 0, fwhm - h)) / (2 * h)
    band = np.abs(dmodel_dfwhm) * fwhm_err

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 4.6),
                                 gridspec_kw=dict(height_ratios=[2, 1], hspace=0.06))

    a1.plot(xx, model, color="firebrick", label=f"fit, FWHM = {fwhm:.3g} ± {fwhm_err:.1g}")
    a1.fill_between(xx, model - 3 * band, model + 3 * band, color="firebrick", alpha=0.25, lw=0, label="±3σ")
    a1.errorbar(x[ok], y[ok], yerr[ok], fmt=".", color="0.15", elinewidth=0.8, label="binned mean ± s.e.")
    a1.plot(x[~ok], y[~ok], "x", color="0.6", label="excluded")
    a1.set_ylabel("Pearson ρ")
    a1.legend(fontsize=7, frameon=False)

    residual = y - gaussian_window(x, 0, fwhm)
    a2.fill_between(xx, -3 * band, 3 * band, color="firebrick", alpha=0.25, lw=0)
    a2.axhline(0, color="firebrick", lw=1)
    a2.errorbar(x[ok], residual[ok], yerr[ok], fmt=".", color="0.15", elinewidth=0.8)
    a2.plot(x[~ok], residual[~ok], "x", color="0.6")
    a2.set_ylabel("residual")
    a2.set_xlabel(f"{xlabel} ({fit['bin_width']:.3g} bins)")

    fig.align_ylabels([a1, a2])
    return fig


def correlation_matrix(fit, xlabel="x"):
    """Plot the correlation of every pair of features.

    Parameters
    ----------
    fit : dict
        From `kernel.fit_kernel_width`.
    xlabel : str, optional
        The name of the feature axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    order = np.argsort(fit["x_axis"], kind="stable")  # pcolormesh needs a sorted axis
    e = edges(fit["x_axis"][order])
    corr = fit["corr"][np.ix_(order, order)]
    mesh = ax.pcolormesh(e, e, corr, cmap="viridis", vmin=-1, vmax=1, rasterized=True)
    fig.colorbar(mesh, ax=ax, label="Pearson ρ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(xlabel)
    ax.set_aspect("equal")
    return fig


def in_distribution(curve, level=0.95):
    """Plot the fraction of centres still in distribution against magnitude.

    The required level, and the limit on |xi| that it gives (`ood.max_magnitude`), are both marked.

    Parameters
    ----------
    curve : dict
        From `ood.in_distribution_curve`.
    level : float, optional
        The fraction required to be in distribution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lim = max_magnitude(curve, level)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.axvspan(-lim, lim, color="k", alpha=0.06, lw=0, label=f"|ξ| ≤ {lim:.2g}σ")
    ax.plot(curve["xi"], curve["fraction_in"], color="k", label=f"tol = {curve['tol']:g}")
    ax.axhline(level, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("ξ (σ)")
    ax.set_ylabel("fraction in distribution")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, frameon=False)
    return fig


def ood_map(curve, result, xlabel="x"):
    """Map the fraction fitted over every centre and magnitude.

    The boundary of the in-distribution region is drawn as a black line.

    Parameters
    ----------
    curve : dict
        From `ood.in_distribution_curve`.
    result : PerturbResult
        The run that the curve was calculated from.
    xlabel : str, optional
        The name of the feature axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 2.8))
    values = curve["values"].T
    lo, hi = np.nanpercentile(values, [1, 99])
    mesh = ax.pcolormesh(edges(result.centre_axis), edges(result.xi), values, cmap="viridis", vmin=lo, vmax=hi,
                         rasterized=True)
    fig.colorbar(mesh, ax=ax, label="fraction fitted")

    inside = curve["inside"].T.astype(float)
    ax.contour(result.centre_axis, result.xi, inside, levels=[0.5], colors="k", linewidths=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ξ (σ)")
    ax.set_title(f"in distribution inside the black line (tol = {curve['tol']:g})", fontsize=9)
    return fig


def gradient_map(result, output=0, names=None, xlabel="x", vmax=None):
    """Map the response gradient of one output over every centre and magnitude.

    The gradient is the slope of the response between neighbouring magnitudes, and its mean down each column is the
    importance.

    Parameters
    ----------
    result : PerturbResult
    output : int, optional
        The index of the output to plot.
    names : list of str, optional
        The output names, used for the labels.
    xlabel : str, optional
        The name of the feature axis.
    vmax : float, optional
        The colour limit, which defaults to the 99th percentile of |gradient|.

    Returns
    -------
    matplotlib.figure.Figure
    """
    name = _label(names, output)
    g = result.gradient[..., output]
    v = _colour_limit(g, vmax)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    mesh = ax.pcolormesh(edges(result.centre_axis), result.xi, g.T, cmap="PuOr", vmin=-v, vmax=v, rasterized=True)
    fig.colorbar(mesh, ax=ax, label=f"∂Δ {name} / ∂ξ  (per σ)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ξ (σ)")
    ax.set_title(f"{name}: response gradient, FWHM = {result.fwhm:.3g}", fontsize=9)
    return fig


def callouts(result, output=0, n_positive=2, n_negative=2, n_nonlinear=3, threshold=None, names=None, xlabel="x",
             min_spacing=None, vmax=None):
    """Plot the gradient map of one output, with the most interesting columns drawn out as callouts.

    Above the map are the n_positive columns with the largest importance, and the n_negative with the most negative
    importance. Below it are the n_nonlinear columns whose response bends furthest from a straight line. Each callout
    shows that column's response curve, coloured by its gradient as in the map, against the straight line
    importance * xi (dashed).

    The bend is measured as the RMS distance of the response from that line, in the output's units, so a flat column
    of noise is never picked. The picked columns are at least min_spacing apart on the feature axis, and each kind of
    callout avoids the columns already picked by the others.

    The numbers behind the picks and lines are ``result.importance``, ``result.nonlinearity`` and
    ``result.crossing(threshold)``, if you want to use them directly.

    Parameters
    ----------
    result : PerturbResult
    output : int, optional
        The index of the output to plot.
    n_positive, n_negative, n_nonlinear : int, optional
        The number of callouts of each kind.
    threshold : float, optional
        A change in the output that is worth noticing (e.g. its uncertainty), in the output's units. Responses
        beyond +-threshold are blacked out on the map and shaded in the callouts (hatched above, plain below), and
        a black line in each callout marks where its response first crosses it.
    names : list of str, optional
        The output names, used for the labels.
    xlabel : str, optional
        The name of the feature axis.
    min_spacing : float, optional
        The minimum distance between picked columns. This defaults to the kernel FWHM, i.e. the width of one
        perturbation.
    vmax : float, optional
        The colour limit, which defaults to the 99th percentile of |gradient|.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import ConnectionPatch, Rectangle

    name = _label(names, output)
    r = result.response[..., output]
    g = result.gradient[..., output]
    imp = result.importance[:, output]
    bend = result.nonlinearity[:, output]
    xc = result.centre_axis
    xi = result.xi
    if threshold is None:
        cross = None
    else:
        cross = result.crossing(threshold)[..., output]  # (n_centres, 2)
    if min_spacing is None:
        min_spacing = result.fwhm

    def pick(score, n, taken):
        """Pick up to n columns with the highest score, spaced out from each other and from `taken`."""
        chosen = []
        for j in np.argsort(score)[::-1]:
            if len(chosen) == n:
                break
            if all(abs(xc[j] - xc[k]) >= min_spacing for k in chosen + taken):
                chosen.append(j)
        return chosen

    positive = pick(imp, n_positive, [])
    negative = pick(-imp, n_negative, positive)
    nonlinear = pick(bend, n_nonlinear, positive + negative)

    kind = {}
    for j in positive:
        kind[j] = "positive"
    for j in negative:
        kind[j] = "negative"
    for j in nonlinear:
        kind[j] = "non-linear"

    def by_position(columns):
        return sorted(columns, key=lambda j: xc[j])

    # (columns, gridspec row, title) for the callouts above and below the map
    rows = [
        (by_position(positive + negative), 0, "steepest response"),
        (by_position(nonlinear), 4, "most non-linear response"),
    ]

    ncol = max(n_positive + n_negative, n_nonlinear, 1)
    fig = plt.figure(figsize=(2.3 * ncol + 1, 7.5))
    gs = fig.add_gridspec(5, ncol, height_ratios=[1, 0.45, 1.5, 0.45, 1], hspace=0.05, wspace=0.35)

    ax_map = fig.add_subplot(gs[2, :])
    v = _colour_limit(g, vmax)
    norm = plt.Normalize(-v, v)
    xe = edges(xc)
    mesh = ax_map.pcolormesh(xe, xi, g.T, cmap="PuOr", norm=norm, rasterized=True)
    fig.colorbar(mesh, ax=ax_map, label=f"∂Δ {name} / ∂ξ (per σ)", pad=0.01)

    if threshold is not None:
        big = np.abs(r).max() + threshold
        with plt.rc_context({"hatch.color": "0.6"}):  # hatched above, plain below, as in the callouts
            ax_map.contourf(xc, xi, r.T, levels=[threshold, big], colors="0.1", hatches=["//"], zorder=5)
            ax_map.contourf(xc, xi, r.T, levels=[-big, -threshold], colors="0.1", hatches=[None], zorder=5)
    ax_map.axhline(0, color="k", lw=0.5, ls="--")
    ax_map.set_xlabel(xlabel)
    ax_map.set_ylabel("ξ (σ)")
    ax_map.set_xlim(xc[0], xc[-1])
    ax_map.set_ylim(xi[0], xi[-1])

    for picked, row, title in rows:
        top = row == 0
        axes = [fig.add_subplot(gs[row, i]) for i in range(len(picked))]

        for i, (ax, j) in enumerate(zip(axes, picked)):
            # the response curve as line segments, each coloured by its gradient
            points = np.column_stack([xi, r[j]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap="PuOr", norm=norm, linewidths=2.5)
            lc.set_array(g[j])
            ax.add_collection(lc)

            ax.plot(xi, imp[j] * xi, "k--", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axvline(0, color="0.7", lw=0.5)
            ax.autoscale_view()
            ax.set_title(f"{kind[j]}, {xlabel} = {xc[j]:.3g}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("ξ (σ)", fontsize=8)
            if i == 0:
                ax.set_ylabel(f"Δ {name}", fontsize=8)

            # outline the column on the map and join its corners to the callout's
            outline = Rectangle((xe[j], xi[0]), xe[j + 1] - xe[j], xi[-1] - xi[0], fill=False, ec="firebrick",
                                lw=1, zorder=10)
            ax_map.add_patch(outline)
            if top:
                y_map, y_ax = xi[-1], 0
            else:
                y_map, y_ax = xi[0], 1
            for x_map, x_ax in ((xe[j], 0), (xe[j + 1], 1)):
                link = ConnectionPatch((x_map, y_map), (x_ax, y_ax), ax_map.transData, ax.transAxes,
                                       color="firebrick", lw=0.7)
                fig.add_artist(link)

        if not axes:
            continue

        # share one y scale across each row, so the callouts can be compared at a glance
        ymin = min(a.get_ylim()[0] for a in axes)
        ymax = max(a.get_ylim()[1] for a in axes)
        if threshold is not None:
            ymin = min(ymin, -1.5 * threshold)
            ymax = max(ymax, 1.5 * threshold)

        for a, j in zip(axes, picked):
            a.set_ylim(ymin, ymax)
            a.set_xlim(xi[0], xi[-1])
            if threshold is None:
                continue
            a.fill_between(xi, threshold, ymax, facecolor="0.8", edgecolor="0.9", lw=0, zorder=0, hatch="//")
            a.fill_between(xi, ymin, -threshold, facecolor="0.8", edgecolor="0.9", lw=0, zorder=0)
            for c in cross[j][np.isfinite(cross[j])]:
                a.axvline(c, color="k", lw=1.2)

        if top:
            axes[0].annotate(title, (0, 1.25), xycoords="axes fraction", fontsize=9, weight="bold", va="bottom")
        else:
            axes[0].annotate(title, (0, -0.45), xycoords="axes fraction", fontsize=9, weight="bold", va="top")
    return fig


def response_curves(result, output=0, centres=None, names=None, xlabel="x"):
    """Plot the response against magnitude for a few single centres.

    Parameters
    ----------
    result : PerturbResult
    output : int, optional
        The index of the output to plot.
    centres : array_like of int, optional
        Indices into ``result.centres``. By default, 6 are spread evenly along the axis.
    names : list of str, optional
        The output names, used for the labels.
    xlabel : str, optional
        The name of the feature axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if centres is None:
        centres = np.linspace(0, len(result.centres) - 1, 6).round().astype(int)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(centres)))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    for j, colour in zip(centres, colours):
        label = f"{xlabel} = {result.centre_axis[j]:.3g}"
        ax.plot(result.xi, result.response[j, :, output], color=colour, label=label)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("ξ (σ)")
    ax.set_ylabel(f"Δ {_label(names, output)}")
    ax.legend(fontsize=7, frameon=False)
    return fig


def importance(result, outputs=None, names=None, xlabel="x", ncols=4):
    """Plot the importance of every centre feature, with one panel per output.

    Parameters
    ----------
    result : PerturbResult
    outputs : iterable of int, optional
        The outputs to plot, by default all of them.
    names : list of str, optional
        The output names, used for the panel titles.
    xlabel : str, optional
        The name of the feature axis.
    ncols : int, optional
        The maximum number of panels per row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if outputs is None:
        outputs = range(result.response.shape[-1])
    outputs = list(outputs)
    ncols = min(ncols, len(outputs))
    nrows = int(np.ceil(len(outputs) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2 * nrows), sharex=True, squeeze=False)
    imp = result.importance
    for ax, p in zip(axes.flat, outputs):
        ax.axhline(0, color="0.6", lw=0.6)
        ax.plot(result.centre_axis, imp[:, p], "k", lw=1)
        ax.set_title(_label(names, p), fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.flat[len(outputs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("importance (per σ)", fontsize=8)
    fig.tight_layout()
    return fig
