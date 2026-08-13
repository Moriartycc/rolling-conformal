"""Rolling conformal prediction along a minimum-norm least-squares path.

For observation i (one-based), both the observed pair and candidate pairs are
scored with the predictable estimator fitted to observations 1, ..., i - 1:

    s_i((x, y); Z_<i) = 0.5 * (x^T theta_hat_{i-1} - y)^2.

The experiment follows a single growing data stream from i < d through the
interpolation threshold i = d and into the underparameterized regime i > d.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NOMINAL_LEVELS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
EVOLUTION_ALPHAS = (0.40, 0.20, 0.10, 0.05)


def first_informative_iteration(alpha: float) -> int:
    """Return the first integer i for which alpha * (i + 1) > 1."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")
    iteration = max(1, int(np.floor(1.0 / alpha)))
    while alpha * (iteration + 1) <= 1.0:
        iteration += 1
    return iteration


def first_plotted_iteration(config: "ExperimentConfig") -> int:
    """Apply the finite-sample plotting cutoff used throughout stream plots."""
    alpha = 1.0 - config.set_nominal_coverage
    return first_informative_iteration(alpha)


def stream_tick_label(iteration: int) -> str:
    """Format large stream sizes compactly while keeping labels in LaTeX."""
    if iteration >= 1000:
        return rf"${iteration / 1000:g}\mathrm{{k}}$"
    return rf"${iteration:d}$"


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 40_000
    d: int = 200
    sigma: float = 1.0
    trials: int = 100
    seed: int = 2026
    holdout_size: int = 500
    set_nominal_coverage: float = 0.90
    y_grid_size: int = 401
    y_max: float | None = None
    evaluation_stride: int = 1
    nominal_levels: tuple[float, ...] = DEFAULT_NOMINAL_LEVELS


def prescribed_theta_star(d: int) -> np.ndarray:
    """Return theta_star = e_1 in R^d."""
    if d < 3:
        raise ValueError("d must be at least 3 for the fixed test features.")
    theta_star = np.zeros(d, dtype=float)
    theta_star[0] = 1.0
    return theta_star


def make_test_points(d: int) -> tuple[np.ndarray, list[str]]:
    """Return four sparse fixed features with comparable Euclidean norms."""
    if d < 3:
        raise ValueError("d must be at least 3 for the fixed test features.")

    radius = np.sqrt(float(d))
    test_x = np.zeros((4, d), dtype=float)
    test_x[0, 0] = radius
    test_x[1, 0] = -radius
    test_x[2, 1] = radius
    test_x[3, [0, 1]] = radius / np.sqrt(2.0)
    labels = [
        r"\sqrt{d}\,e_1",
        r"-\sqrt{d}\,e_1",
        r"\sqrt{d}\,e_2",
        r"\sqrt{d/2}\,(e_1+e_2)",
    ]
    return test_x, labels


def sample_linear_data(
    sample_size: int,
    d: int,
    theta_star: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample X ~ N(0, I_d) and Y = X^T theta_star + epsilon."""
    x = rng.standard_normal((sample_size, d))
    noise = sigma * rng.standard_normal(sample_size)
    y = x @ theta_star + noise
    return x, y


def fit_min_norm_checkpoint_path(
    x: np.ndarray,
    y: np.ndarray,
    evaluation_iterations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on every observation but retain scores only at checkpoints.

    ``predictable_thetas[k]`` is fitted through observation ``i_k - 1`` and
    is used to score observation ``i_k``. ``theta_path[k + 1]`` is fitted
    through observation ``i_k`` and is used for the displayed point
    prediction at that checkpoint. Between checkpoints, underparameterized
    least-squares updates are applied in blocks for better BLAS efficiency.
    """
    n, d = x.shape
    evaluation_iterations = np.asarray(evaluation_iterations, dtype=int)
    if y.shape != (n,):
        raise ValueError("y must have shape (n,).")
    if (
        evaluation_iterations.ndim != 1
        or evaluation_iterations.size == 0
        or evaluation_iterations[0] < 1
        or evaluation_iterations[-1] > n
        or np.any(np.diff(evaluation_iterations) <= 0)
    ):
        raise ValueError("evaluation_iterations must be increasing in [1, n].")

    checkpoint_count = evaluation_iterations.size
    predictable_thetas = np.empty((checkpoint_count, d), dtype=float)
    theta_path = np.zeros((checkpoint_count + 1, d), dtype=float)
    calibration_scores = np.empty(checkpoint_count, dtype=float)

    theta = np.zeros(d, dtype=float)
    row_basis = np.zeros((d, d), dtype=float)
    basis_size = 0
    gram_inverse: np.ndarray | None = None
    processed_count = 0
    refactored_after_threshold = False

    def update_through(stop: int) -> None:
        """Update the fit from ``processed_count`` through ``stop - 1``."""
        nonlocal theta, basis_size, gram_inverse, processed_count
        nonlocal refactored_after_threshold

        while processed_count < stop and processed_count < d:
            current_x = x[processed_count]
            current_y = y[processed_count]
            residual = current_x @ theta - current_y
            if basis_size:
                active_basis = row_basis[:, :basis_size]
                orthogonal_component = current_x - active_basis @ (
                    active_basis.T @ current_x
                )
                orthogonal_component -= active_basis @ (
                    active_basis.T @ orthogonal_component
                )
            else:
                orthogonal_component = current_x.copy()

            orthogonal_norm = float(np.linalg.norm(orthogonal_component))
            tolerance = np.finfo(float).eps * d * max(
                1.0, float(np.linalg.norm(current_x))
            )
            prefix_size = processed_count + 1
            if orthogonal_norm <= tolerance:
                theta = np.linalg.lstsq(
                    x[:prefix_size], y[:prefix_size], rcond=None
                )[0]
            else:
                new_direction = orthogonal_component / orthogonal_norm
                theta += new_direction * (-residual / orthogonal_norm)
                row_basis[:, basis_size] = new_direction
                basis_size += 1
            processed_count = prefix_size

        if processed_count < stop and gram_inverse is None:
            # Establish a nonsingular normal matrix immediately after the
            # interpolation threshold.
            prefix_size = processed_count + 1
            prefix_x = x[:prefix_size]
            theta = np.linalg.lstsq(prefix_x, y[:prefix_size], rcond=None)[0]
            gram_inverse = np.linalg.inv(prefix_x.T @ prefix_x)
            processed_count = prefix_size

        if processed_count < stop:
            if gram_inverse is None:  # pragma: no cover - defensive guard
                raise RuntimeError("The underparameterized inverse is unavailable.")
            block_x = x[processed_count:stop]
            block_y = y[processed_count:stop]
            inverse_times_xt = gram_inverse @ block_x.T
            middle = np.eye(block_x.shape[0]) + block_x @ inverse_times_xt
            residuals = block_y - block_x @ theta
            theta += inverse_times_xt @ np.linalg.solve(middle, residuals)
            gram_inverse -= inverse_times_xt @ np.linalg.solve(
                middle, inverse_times_xt.T
            )
            gram_inverse = 0.5 * (gram_inverse + gram_inverse.T)
            processed_count = stop

        # One direct refactorization soon after interpolation removes the
        # numerical sensitivity inherited from the nearly square design.
        if (
            gram_inverse is not None
            and not refactored_after_threshold
            and processed_count >= 2 * d
        ):
            prefix_x = x[:processed_count]
            gram = prefix_x.T @ prefix_x
            theta = np.linalg.solve(gram, prefix_x.T @ y[:processed_count])
            gram_inverse = np.linalg.inv(gram)
            refactored_after_threshold = True

    for checkpoint_index, iteration in enumerate(evaluation_iterations):
        update_through(int(iteration) - 1)
        predictable_thetas[checkpoint_index] = theta
        residual = x[iteration - 1] @ theta - y[iteration - 1]
        calibration_scores[checkpoint_index] = 0.5 * residual**2
        update_through(int(iteration))
        theta_path[checkpoint_index + 1] = theta

    return theta_path, predictable_thetas, calibration_scores


def holdout_coverage_paths(
    predictable_thetas: np.ndarray,
    calibration_scores: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    nominal_levels: np.ndarray,
) -> np.ndarray:
    """Return hold-out coverage at every prefix for each nominal level."""
    n = calibration_scores.size
    if predictable_thetas.shape[0] != n:
        raise ValueError(
            "predictable_thetas and calibration_scores have incompatible lengths."
        )

    predictable_predictions = predictable_thetas @ holdout_x.T
    candidate_scores = 0.5 * (
        predictable_predictions - holdout_y[None, :]
    ) ** 2
    cumulative_counts = np.cumsum(
        candidate_scores > calibration_scores[:, None],
        axis=0,
        dtype=np.int64,
    )
    coverage_paths = np.ones((nominal_levels.size, n + 1), dtype=float)
    thresholds = nominal_levels[:, None] * (
        np.arange(1, n + 1, dtype=float)[None, :] + 1.0
    )
    coverage_paths[:, 1:] = np.mean(
        cumulative_counts[None, :, :] < thresholds[:, :, None],
        axis=2,
    )
    return coverage_paths


def conditional_membership_path(
    theta_path: np.ndarray,
    predictable_thetas: np.ndarray,
    calibration_scores: np.ndarray,
    test_x: np.ndarray,
    y_grid: np.ndarray,
    nominal_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RoCP membership over T, fixed test features, and candidate y."""
    n = calibration_scores.size
    if predictable_thetas.shape[0] != n or theta_path.shape[0] != n + 1:
        raise ValueError("Checkpoint paths have incompatible lengths.")
    predictable_predictions = predictable_thetas @ test_x.T
    candidate_scores = 0.5 * (
        predictable_predictions[:, :, None] - y_grid[None, None, :]
    ) ** 2
    comparisons = candidate_scores > calibration_scores[:, None, None]
    cumulative_counts = np.cumsum(comparisons, axis=0, dtype=np.int64)

    membership = np.ones(
        (n + 1, test_x.shape[0], y_grid.size), dtype=bool
    )
    thresholds = nominal_coverage * (
        np.arange(1, n + 1, dtype=float) + 1.0
    )
    membership[1:] = cumulative_counts < thresholds[:, None, None]
    point_predictions = theta_path @ test_x.T
    return membership, point_predictions


def _setup_matplotlib():
    """Import Matplotlib with a cache outside the figure directory."""
    matplotlib_cache = Path(tempfile.gettempdir()) / "rolling-conformal-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Plotting requires matplotlib. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return matplotlib, plt


def plot_coverage_calibration(
    output_stem: Path,
    nominal_levels: np.ndarray,
    mean_coverage: np.ndarray,
    ci95: np.ndarray,
    config: ExperimentConfig,
) -> None:
    """Plot empirical marginal coverage against nominal coverage."""
    _, plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(13.2, 8.4))

    lower = max(0.0, float(np.min(nominal_levels)) - 0.04)
    upper = 1.005
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.8,
        color="#555555",
        label="Nominal coverage",
        zorder=1,
    )
    bounded_error = np.vstack(
        [np.minimum(ci95, mean_coverage), np.minimum(ci95, 1.0 - mean_coverage)]
    )
    ax.errorbar(
        nominal_levels,
        mean_coverage,
        yerr=bounded_error,
        color="#0072B2",
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        elinewidth=1.4,
        capsize=4.0,
        label="RoCP empirical coverage",
        zorder=3,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal coverage", fontsize=20)
    ax.set_ylabel("Empirical hold-out coverage", fontsize=20)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", fontsize=15, frameon=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_evolution_panels(
    output_stem: Path,
    alphas: np.ndarray,
    mean_paths: np.ndarray,
    ci95_paths: np.ndarray,
    iterations: np.ndarray,
    config: ExperimentConfig,
) -> None:
    """Plot empirical hold-out coverage over the stream for four alphas."""
    matplotlib, plt = _setup_matplotlib()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 10.2),
        sharex=True,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(2, 2)
    # On the linear scale, retain the full fitted path i=1,...,n.
    plot_start = 1
    displayed_mask = iterations >= plot_start
    displayed_iterations = iterations[displayed_mask]
    stream_ticks = np.unique(
        np.concatenate(
            (
                np.asarray([plot_start], dtype=int),
                np.rint(np.linspace(0, config.n, 5)[1:]).astype(int),
            )
        )
    )
    stream_ticks = np.unique(
        stream_ticks[
            (stream_ticks >= displayed_iterations[0])
            & (stream_ticks <= config.n)
        ]
    )

    for panel_index, ax in enumerate(axes.flat):
        alpha = float(alphas[panel_index])
        nominal_coverage = 1.0 - alpha
        panel_mask = displayed_mask
        panel_iterations = iterations[panel_mask]
        mean = mean_paths[panel_index, panel_mask]
        interval = ci95_paths[panel_index, panel_mask]
        error_every = max(1, panel_iterations.size // 15)
        bounded_error = np.vstack(
            [np.minimum(interval, mean), np.minimum(interval, 1.0 - mean)]
        )
        ax.axhline(
            nominal_coverage,
            color="#555555",
            linestyle="--",
            linewidth=1.8,
            label=r"$\mathrm{Nominal\ coverage}\ 1-\alpha$",
            zorder=1,
        )
        ax.axvline(
            config.d,
            color="#777777",
            linestyle=":",
            linewidth=1.6,
            label=r"$i=d$",
            zorder=1,
        )
        ax.errorbar(
            panel_iterations,
            mean,
            yerr=bounded_error,
            color="#0072B2",
            marker="o",
            markersize=5.0,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=3.0,
            markevery=error_every,
            errorevery=error_every,
            label=r"$\mathrm{RoCP\ empirical\ coverage}$",
            zorder=3,
        )
        ax.set_xscale("linear")
        ax.set_xlim(displayed_iterations[0], config.n)
        ax.set_ylim(1.0 - 2.0 * alpha, 1.0)
        ax.set_xticks(stream_ticks)
        ax.set_xticklabels(
            [stream_tick_label(int(tick)) for tick in stream_ticks]
        )
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(rf"$\alpha={alpha:g}$", fontsize=22, pad=10)
        ax.tick_params(axis="both", labelsize=18)
        if panel_index % 2 == 1:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True)
        ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.8)

    axes[0, 0].legend(loc="lower right", fontsize=15, frameon=True)
    fig.supxlabel(
        rf"$\mathrm{{Data\ stream\ size}}\ i\ "
        rf"\mathrm{{(from\ {plot_start}\ to}}\ n\mathrm{{)}}$",
        fontsize=22,
        y=0.025,
    )
    fig.supylabel(
        r"$\mathrm{Empirical\ hold\!\!-\!out\ coverage}$",
        fontsize=22,
        x=0.025,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.94,
        bottom=0.10,
        top=0.95,
        wspace=0.12,
        hspace=0.18,
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_conditional_panels(
    output_stem: Path,
    inclusion_rates: np.ndarray,
    mean_point_predictions: np.ndarray,
    point_prediction_ci95: np.ndarray,
    iterations: np.ndarray,
    y_grid: np.ndarray,
    test_labels: list[str],
    ground_truths: np.ndarray,
    y_max: float,
    config: ExperimentConfig,
) -> None:
    """Plot four fixed-feature RoCP paths with point and oracle predictions."""
    matplotlib, plt = _setup_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    membership_cmap = LinearSegmentedColormap.from_list(
        "rocP_membership",
        ["#F0F2F4", "#B8D9EC", "#0072B2"],
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.8),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(2, 2)
    # Only show stream sizes for which alpha * (i + 1) > 1, so the finite-
    # sample conformal cutoff is nontrivial.  At nominal coverage 0.9 this
    # starts the displayed path at i=10.
    plot_start = first_plotted_iteration(config)
    displayed_mask = iterations >= plot_start
    displayed_iterations = iterations[displayed_mask]
    displayed_rates = inclusion_rates[displayed_mask]
    clipped_predictions = np.clip(
        mean_point_predictions[displayed_mask], -y_max, y_max
    )
    displayed_prediction_ci95 = point_prediction_ci95[displayed_mask]
    clipped_prediction_lower = np.clip(
        mean_point_predictions[displayed_mask] - displayed_prediction_ci95,
        -y_max,
        y_max,
    )
    clipped_prediction_upper = np.clip(
        mean_point_predictions[displayed_mask] + displayed_prediction_ci95,
        -y_max,
        y_max,
    )
    prediction_error_indices = np.unique(
        np.searchsorted(
            displayed_iterations,
            np.geomspace(
                displayed_iterations[0], displayed_iterations[-1], 20
            ),
        ).clip(0, displayed_iterations.size - 1)
    )
    image = None

    for test_index, ax in enumerate(axes.flat):
        image = ax.pcolormesh(
            displayed_iterations,
            y_grid,
            displayed_rates[:, test_index, :].T,
            shading="auto",
            cmap=membership_cmap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
            zorder=0,
        )
        ax.axvline(
            config.d,
            color="#666666",
            linestyle=":",
            linewidth=1.7,
            zorder=2,
        )
        ax.plot(
            displayed_iterations,
            clipped_predictions[:, test_index],
            color="#CC0077",
            linewidth=2.2,
            label=(r"$\mathrm{Averaged}\ X_{n+1}^{\top}\widehat{\theta}_i"
                   r"\ \mathrm{(95\%\ CI)}$"),
            zorder=4,
        )
        prediction_yerr = np.vstack(
            (
                clipped_predictions[:, test_index]
                - clipped_prediction_lower[:, test_index],
                clipped_prediction_upper[:, test_index]
                - clipped_predictions[:, test_index],
            )
        )
        ax.errorbar(
            displayed_iterations[prediction_error_indices],
            clipped_predictions[prediction_error_indices, test_index],
            yerr=prediction_yerr[:, prediction_error_indices],
            fmt="none",
            ecolor="#CC0077",
            elinewidth=1.25,
            capsize=2.5,
            capthick=1.1,
            zorder=5,
        )
        ax.axhline(
            ground_truths[test_index],
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            label=r"$X_{n+1}^{\top}\theta^\star$",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_xlim(displayed_iterations[0], config.n)
        stream_ticks = np.asarray(
            [plot_start, 50, config.d, 1000, 10000, config.n], dtype=int
        )
        stream_ticks = np.unique(
            stream_ticks[
                (stream_ticks >= displayed_iterations[0])
                & (stream_ticks <= config.n)
            ]
        )
        ax.set_xticks(stream_ticks)
        ax.set_xticklabels(
            [stream_tick_label(int(tick)) for tick in stream_ticks]
        )
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-y_max, y_max)
        ax.set_title(
            rf"$X_{{n+1}}={test_labels[test_index]},\quad "
            rf"X_{{n+1}}^\top\theta^\star={ground_truths[test_index]:.2f}$",
            fontsize=20,
            pad=11,
        )
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(False)

    axes[0, 0].legend(loc="lower left", fontsize=14.5, frameon=True)
    fig.supxlabel(
        rf"$\mathrm{{Data\ stream\ size}}\ i\ "
        rf"\mathrm{{(from\ {plot_start}\ to}}\ n\mathrm{{)}}$",
        fontsize=22,
        y=0.03,
    )
    fig.supylabel(
        r"$\mathrm{Rolling\!\!-\!conformal\ prediction\ set}\ "
        r"\widehat{C}_i(X_{n+1})\ \mathrm{and\ least\ square\ prediction}\ "
        r"X_{n+1}^{\top}\widehat{\theta}_i$",
        fontsize=19,
        x=0.012,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.09,
        top=0.95,
        wspace=0.08,
        hspace=0.18,
    )
    if image is None:  # pragma: no cover - four panels are always present
        raise RuntimeError("No conditional panel was rendered.")
    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label(
        r"$\mathrm{Inclusion\ frequency\ of}\ "
        r"\widehat{C}_i(X_{n+1})$",
        fontsize=18,
    )
    colorbar.ax.tick_params(labelsize=16)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _validate_config(config: ExperimentConfig) -> None:
    if config.d < 3:
        raise ValueError("d must be at least 3.")
    if config.n <= config.d:
        raise ValueError("n must exceed d so the stream crosses interpolation.")
    if config.sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if config.trials < 2:
        raise ValueError("At least two trials are needed for confidence intervals.")
    if config.holdout_size < 1:
        raise ValueError("holdout_size must be positive.")
    if config.evaluation_stride < 1:
        raise ValueError("evaluation_stride must be positive.")
    if config.n % config.evaluation_stride != 0:
        raise ValueError("evaluation_stride must divide n.")
    if config.y_grid_size < 51 or config.y_grid_size % 2 == 0:
        raise ValueError("y_grid_size must be an odd integer at least 51.")
    if not (0.0 < config.set_nominal_coverage < 1.0):
        raise ValueError("set_nominal_coverage must lie strictly between 0 and 1.")
    levels = np.asarray(config.nominal_levels, dtype=float)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("Provide at least two nominal coverage levels.")
    if np.any(np.diff(levels) <= 0) or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("nominal_levels must be increasing and lie in (0, 1).")


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Run all Monte Carlo trials and write the three paper figures."""
    _validate_config(config)
    theta_star = prescribed_theta_star(config.d)
    test_x, test_labels = make_test_points(config.d)
    ground_truths = test_x @ theta_star
    y_max = (
        float(config.y_max)
        if config.y_max is not None
        else np.sqrt(2.0 * float(config.d))
    )
    if y_max <= float(np.max(np.abs(ground_truths))):
        raise ValueError("y_max must exceed every fixed test point's oracle mean.")

    nominal_levels = np.asarray(config.nominal_levels, dtype=float)
    evolution_alphas = np.asarray(EVOLUTION_ALPHAS, dtype=float)
    evolution_nominal_levels = 1.0 - evolution_alphas
    all_coverage_levels = np.concatenate(
        [nominal_levels, evolution_nominal_levels]
    )
    y_grid = np.linspace(-y_max, y_max, config.y_grid_size)
    evaluation_iterations = np.arange(
        config.evaluation_stride,
        config.n + 1,
        config.evaluation_stride,
        dtype=int,
    )
    iterations = np.concatenate(([0], evaluation_iterations))
    checkpoint_count = evaluation_iterations.size
    trial_coverages = np.empty(
        (config.trials, nominal_levels.size), dtype=float
    )
    trial_coverage_paths = np.empty(
        (config.trials, evolution_alphas.size, checkpoint_count + 1),
        dtype=float,
    )
    membership_sums = np.zeros(
        (checkpoint_count + 1, test_x.shape[0], y_grid.size), dtype=np.int32
    )
    point_predictions = np.empty(
        (config.trials, checkpoint_count + 1, test_x.shape[0]), dtype=float
    )
    child_seeds = np.random.SeedSequence(config.seed).spawn(config.trials)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    report_every = max(1, config.trials // 20)
    for trial_index, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        train_x, train_y = sample_linear_data(
            config.n, config.d, theta_star, config.sigma, rng
        )
        theta_path, predictable_thetas, calibration_scores = (
            fit_min_norm_checkpoint_path(
                train_x,
                train_y,
                evaluation_iterations,
            )
        )
        holdout_x, holdout_y = sample_linear_data(
            config.holdout_size,
            config.d,
            theta_star,
            config.sigma,
            rng,
        )
        all_trial_coverage_paths = holdout_coverage_paths(
            predictable_thetas,
            calibration_scores,
            holdout_x,
            holdout_y,
            all_coverage_levels,
        )
        trial_coverages[trial_index] = all_trial_coverage_paths[
            : nominal_levels.size, -1
        ]
        trial_coverage_paths[trial_index] = all_trial_coverage_paths[
            nominal_levels.size :
        ]
        membership, trial_predictions = conditional_membership_path(
            theta_path,
            predictable_thetas,
            calibration_scores,
            test_x,
            y_grid,
            config.set_nominal_coverage,
        )
        membership_sums += membership
        point_predictions[trial_index] = trial_predictions

        completed = trial_index + 1
        if completed == config.trials or completed % report_every == 0:
            elapsed = time.perf_counter() - run_started
            remaining = elapsed * (config.trials - completed) / completed
            print(
                f"Completed trial {completed}/{config.trials} | "
                f"elapsed {elapsed:.1f}s | ETA {remaining:.1f}s",
                flush=True,
            )

    if not np.all(np.isfinite(trial_coverages)):
        raise RuntimeError("Non-finite empirical coverage was produced.")
    if not np.all(np.isfinite(trial_coverage_paths)):
        raise RuntimeError("Non-finite empirical coverage paths were produced.")
    if not np.all(np.isfinite(point_predictions)):
        raise RuntimeError("Non-finite point predictions were produced.")

    mean_coverage = np.mean(trial_coverages, axis=0)
    coverage_se = np.std(trial_coverages, axis=0, ddof=1) / np.sqrt(
        config.trials
    )
    coverage_ci95 = 1.96 * coverage_se
    mean_coverage_paths = np.mean(trial_coverage_paths, axis=0)
    coverage_path_se = np.std(
        trial_coverage_paths, axis=0, ddof=1
    ) / np.sqrt(config.trials)
    coverage_path_ci95 = 1.96 * coverage_path_se
    inclusion_rates = membership_sums / float(config.trials)
    mean_point_predictions = np.mean(point_predictions, axis=0)
    point_prediction_se = np.std(
        point_predictions, axis=0, ddof=1
    ) / np.sqrt(config.trials)
    point_prediction_ci95 = 1.96 * point_prediction_se

    coverage_stem = output_dir / "fig1-1"
    coverage_evolution_stem = output_dir / "fig1-2"
    conditional_stem = output_dir / "fig1-3"
    plot_coverage_calibration(
        coverage_stem,
        nominal_levels,
        mean_coverage,
        coverage_ci95,
        config,
    )
    plot_coverage_evolution_panels(
        coverage_evolution_stem,
        evolution_alphas,
        mean_coverage_paths,
        coverage_path_ci95,
        iterations,
        config,
    )
    plot_conditional_panels(
        conditional_stem,
        inclusion_rates,
        mean_point_predictions,
        point_prediction_ci95,
        iterations,
        y_grid,
        test_labels,
        ground_truths,
        y_max,
        config,
    )


def _parse_nominal_levels(value: str) -> tuple[float, ...]:
    try:
        levels = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Nominal levels must be comma-separated numbers."
        ) from exc
    if len(levels) < 2:
        raise argparse.ArgumentTypeError("Provide at least two nominal levels.")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoCP along a growing minimum-norm linear-regression path."
    )
    parser.add_argument("--n", type=int, default=40_000)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--trials", "--M", dest="trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--holdout-size", "--N", dest="holdout_size", type=int, default=500
    )
    parser.add_argument("--set-nominal-coverage", type=float, default=0.90)
    parser.add_argument("--y-grid-size", type=int, default=401)
    parser.add_argument(
        "--evaluation-stride",
        type=int,
        default=1,
        help="Score and report every k-th stream size; OLS still uses all data.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Symmetric plotting limit; default is sqrt(2*d).",
    )
    parser.add_argument(
        "--nominal-levels",
        type=_parse_nominal_levels,
        default=DEFAULT_NOMINAL_LEVELS,
        help="Comma-separated levels for the coverage calibration curve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n=args.n,
        d=args.d,
        sigma=args.sigma,
        trials=args.trials,
        seed=args.seed,
        holdout_size=args.holdout_size,
        set_nominal_coverage=args.set_nominal_coverage,
        y_grid_size=args.y_grid_size,
        y_max=args.y_max,
        evaluation_stride=args.evaluation_stride,
        nominal_levels=args.nominal_levels,
    )
    run_experiment(config, args.output_dir)
    print(f"\nFigures written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
