"""Compare rolling conformal and split conformal for minimum-norm OLS.

The default configuration reproduces Figure 2 of the paper.  At each partial
sample size, rolling conformal aggregates predictable residual scores along
the OLS training path, while split conformal refits at a prescribed training
ratio and calibrates on the remaining observations.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ALPHAS = np.asarray((0.40, 0.20, 0.10, 0.05), dtype=float)
BURN_INS = (100, 200, 500, 1_000)
SPLIT_RATIOS = (0.20, 0.40, 0.60, 0.80)
ROLLING_COLORS = ("#009E73", "#0072B2", "#56B4E9", "#CC79A7")
SPLIT_COLORS = ("#E69F00", "#D55E00", "#A65628", "#6A3D9A")


@dataclass(frozen=True)
class Config:
    max_n: int = 5_000
    d: int = 200
    sigma: float = 0.2
    trials: int = 100
    checkpoint_step: int = 20
    seed: int = 2026


def sample_linear_data(
    sample_size: int,
    d: int,
    theta_star: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((sample_size, d))
    y = x @ theta_star + sigma * rng.standard_normal(sample_size)
    return x, y


def fit_min_norm_checkpoint_path(
    x: np.ndarray,
    y: np.ndarray,
    evaluation_iterations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictable OLS fits and residual radii at selected indices.

    For one-based index ``i``, the stored fit uses observations ``1,...,i-1``
    and the radius is ``abs(X_i^T theta_hat_{i-1} - Y_i)``.
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
        raise ValueError("evaluation_iterations must increase within [1,n].")

    predictable_thetas = np.empty((evaluation_iterations.size, d), dtype=float)
    residual_radii = np.empty(evaluation_iterations.size, dtype=float)
    theta = np.zeros(d, dtype=float)
    row_basis = np.zeros((d, d), dtype=float)
    basis_size = 0
    gram_inverse: np.ndarray | None = None
    processed_count = 0
    refactored_after_threshold = False

    def update_through(stop: int) -> None:
        nonlocal theta, basis_size, gram_inverse, processed_count
        nonlocal refactored_after_threshold

        while processed_count < stop and processed_count < d:
            current_x = x[processed_count]
            current_y = y[processed_count]
            residual = current_x @ theta - current_y
            if basis_size:
                active_basis = row_basis[:, :basis_size]
                orthogonal = current_x - active_basis @ (
                    active_basis.T @ current_x
                )
                orthogonal -= active_basis @ (active_basis.T @ orthogonal)
            else:
                orthogonal = current_x.copy()
            orthogonal_norm = float(np.linalg.norm(orthogonal))
            tolerance = np.finfo(float).eps * d * max(
                1.0, float(np.linalg.norm(current_x))
            )
            prefix_size = processed_count + 1
            if orthogonal_norm <= tolerance:
                theta = np.linalg.lstsq(
                    x[:prefix_size], y[:prefix_size], rcond=None
                )[0]
            else:
                direction = orthogonal / orthogonal_norm
                theta += direction * (-residual / orthogonal_norm)
                row_basis[:, basis_size] = direction
                basis_size += 1
            processed_count = prefix_size

        if processed_count < stop and gram_inverse is None:
            prefix_size = processed_count + 1
            prefix_x = x[:prefix_size]
            theta = np.linalg.lstsq(prefix_x, y[:prefix_size], rcond=None)[0]
            gram_inverse = np.linalg.inv(prefix_x.T @ prefix_x)
            processed_count = prefix_size

        if processed_count < stop:
            if gram_inverse is None:
                raise RuntimeError("Underparameterized inverse is unavailable.")
            block_x = x[processed_count:stop]
            block_y = y[processed_count:stop]
            inverse_times_xt = gram_inverse @ block_x.T
            middle = np.eye(block_x.shape[0]) + block_x @ inverse_times_xt
            solved_residuals = np.linalg.solve(
                middle, block_y - block_x @ theta
            )
            theta += inverse_times_xt @ solved_residuals
            gram_inverse -= inverse_times_xt @ np.linalg.solve(
                middle, inverse_times_xt.T
            )
            gram_inverse = 0.5 * (gram_inverse + gram_inverse.T)
            processed_count = stop

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
        residual_radii[checkpoint_index] = abs(
            x[iteration - 1] @ theta - y[iteration - 1]
        )
        update_through(int(iteration))

    return predictable_thetas, residual_radii


def rolling_interval_results(
    centers: np.ndarray,
    radii: np.ndarray,
    test_y: float,
    prefix_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact RoCP coverage and Lebesgue lengths at checkpoints."""
    maximum_size = int(prefix_sizes[-1])
    lower = centers[:maximum_size] - radii[:maximum_size]
    upper = centers[:maximum_size] + radii[:maximum_size]
    coordinates = np.unique(np.concatenate((lower, upper)))
    widths = np.diff(coordinates)
    lower_indices = np.searchsorted(coordinates, lower)
    upper_indices = np.searchsorted(coordinates, upper)
    endpoint_differences = np.zeros(coordinates.size, dtype=np.int32)
    cumulative_test_depth = np.cumsum(
        (lower <= test_y) & (test_y <= upper), dtype=np.int32
    )
    covered = np.empty((prefix_sizes.size, ALPHAS.size), dtype=bool)
    lengths = np.empty((prefix_sizes.size, ALPHAS.size), dtype=float)
    previous_size = 0

    for checkpoint_index, prefix_size_value in enumerate(prefix_sizes):
        prefix_size = int(prefix_size_value)
        new_slice = slice(previous_size, prefix_size)
        np.add.at(endpoint_differences, lower_indices[new_slice], 1)
        np.add.at(endpoint_differences, upper_indices[new_slice], -1)
        segment_depths = np.cumsum(endpoint_differences[:-1], dtype=np.int32)
        required_depths = np.floor(ALPHAS * (prefix_size + 1)).astype(int)
        if np.any(required_depths < 1):
            raise ValueError("The RoCP calibration sequence is too short.")
        covered[checkpoint_index] = (
            cumulative_test_depth[prefix_size - 1] >= required_depths
        )
        for alpha_index, required_depth in enumerate(required_depths):
            lengths[checkpoint_index, alpha_index] = widths[
                segment_depths >= required_depth
            ].sum()
        previous_size = prefix_size

    return covered, lengths


def split_results_for_stream(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: float,
    n_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute split-conformal results for all horizons and split ratios."""
    split_points = np.asarray(
        [int(ratio * n) for n in n_values for ratio in SPLIT_RATIOS],
        dtype=int,
    ).reshape(n_values.size, len(SPLIT_RATIOS))
    unique_points = np.unique(split_points)
    split_thetas, _ = fit_min_norm_checkpoint_path(
        train_x, train_y, unique_points + 1
    )
    theta_lookup = {
        int(split_point): split_thetas[index]
        for index, split_point in enumerate(unique_points)
    }
    covered = np.empty(
        (n_values.size, len(SPLIT_RATIOS), ALPHAS.size), dtype=bool
    )
    lengths = np.empty(covered.shape, dtype=float)

    for n_index, n_value in enumerate(n_values):
        n = int(n_value)
        for ratio_index, split_point_value in enumerate(split_points[n_index]):
            split_point = int(split_point_value)
            theta = theta_lookup[split_point]
            radii = np.abs(
                train_x[split_point:n] @ theta - train_y[split_point:n]
            )
            calibration_size = radii.size
            required_depths = np.floor(
                ALPHAS * (calibration_size + 1)
            ).astype(int)
            if np.any(required_depths < 1):
                raise ValueError(
                    f"Calibration set is too short at n={n}, rho={SPLIT_RATIOS[ratio_index]}."
                )
            sorted_radii = np.sort(radii)
            quantiles = sorted_radii[calibration_size - required_depths]
            test_residual = abs(test_y - float(test_x @ theta))
            covered[n_index, ratio_index] = test_residual <= quantiles
            lengths[n_index, ratio_index] = 2.0 * quantiles

    return covered, lengths


def setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "rolling-conformal-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_panel(ax, alpha: float, max_n: int) -> None:
    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=22, pad=7)
    ax.set_xlim(100, max_n)
    ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.72)
    ax.tick_params(labelsize=15)


def plot_lengths(
    output_stem: Path,
    config: Config,
    n_values: np.ndarray,
    rolling_n: dict[int, np.ndarray],
    rolling_lengths: dict[int, np.ndarray],
    split_lengths: np.ndarray,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.7, 9.3), sharex=True, constrained_layout=True
    )
    for alpha_index, (ax, alpha) in enumerate(zip(axes.flat, ALPHAS)):
        for burn_in, color in zip(BURN_INS, ROLLING_COLORS):
            x_values = rolling_n[burn_in]
            means = rolling_lengths[burn_in][:, :, alpha_index].mean(axis=0)
            ax.plot(
                x_values,
                means,
                color=color,
                linewidth=2.0,
                label=rf"RoCP, $m={burn_in:,}$",
            )
            ax.scatter(
                x_values[0],
                means[0],
                s=55,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
        for ratio_index, (ratio, color) in enumerate(
            zip(SPLIT_RATIOS, SPLIT_COLORS)
        ):
            means = split_lengths[:, :, ratio_index, alpha_index].mean(axis=0)
            ax.plot(
                n_values,
                means,
                color=color,
                linestyle="--",
                linewidth=1.9,
                label=rf"Split-CP, $\rho={ratio:g}$",
            )
        style_panel(ax, float(alpha), config.max_n)
        ax.set_yscale("log")
        ax.set_ylim(top=10.0)
        if alpha_index == 0:
            ax.legend(loc="upper right", fontsize=15, frameon=True, ncol=2)

    fig.supxlabel(r"Stream size $n_i$", fontsize=18)
    fig.supylabel("Average prediction-set length", fontsize=18)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage(
    output_stem: Path,
    config: Config,
    n_values: np.ndarray,
    rolling_n: dict[int, np.ndarray],
    rolling_covered: dict[int, np.ndarray],
    split_covered: np.ndarray,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.7, 9.3), sharex=True, constrained_layout=True
    )
    for alpha_index, (ax, alpha) in enumerate(zip(axes.flat, ALPHAS)):
        for burn_in, color in zip(BURN_INS, ROLLING_COLORS):
            x_values = rolling_n[burn_in]
            means = rolling_covered[burn_in][:, :, alpha_index].mean(axis=0)
            ax.plot(
                x_values,
                means,
                color=color,
                linewidth=2.0,
                label=rf"RoCP, $m={burn_in:,}$",
            )
            ax.scatter(
                x_values[0],
                means[0],
                s=55,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
        for ratio_index, (ratio, color) in enumerate(
            zip(SPLIT_RATIOS, SPLIT_COLORS)
        ):
            means = split_covered[:, :, ratio_index, alpha_index].mean(axis=0)
            ax.plot(
                n_values,
                means,
                color=color,
                linestyle="--",
                linewidth=1.9,
                label=rf"Split-CP, $\rho={ratio:g}$",
            )
        ax.axhline(
            1.0 - alpha,
            color="#666666",
            linestyle=(0, (2, 3)),
            linewidth=1.25,
            label="Nominal coverage",
        )
        style_panel(ax, float(alpha), config.max_n)
        ax.set_ylim(max(0.0, 1.0 - 2.0 * alpha - 0.05), 1.01)
        if alpha_index == 0:
            ax.legend(loc="best", fontsize=15, frameon=True, ncol=2)

    fig.supxlabel(r"Stream size $n_i$", fontsize=18)
    fig.supylabel("Empirical coverage", fontsize=18)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate(config: Config) -> None:
    if config.max_n <= config.d or config.d < 1:
        raise ValueError("Require max_n > d >= 1.")
    if config.trials < 2 or config.checkpoint_step < 1:
        raise ValueError("Require at least two trials and a positive step.")
    if config.sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if any(burn_in >= config.max_n for burn_in in BURN_INS):
        raise ValueError("Every burn-in must be smaller than max_n.")


def run_experiment(config: Config, output_dir: Path) -> None:
    validate(config)
    n_values = np.arange(
        100, config.max_n + 1, config.checkpoint_step, dtype=int
    )
    rolling_n = {
        burn_in: np.arange(
            burn_in + config.checkpoint_step,
            config.max_n + 1,
            config.checkpoint_step,
            dtype=int,
        )
        for burn_in in BURN_INS
    }
    rolling_covered = {
        burn_in: np.empty(
            (config.trials, rolling_n[burn_in].size, ALPHAS.size), dtype=bool
        )
        for burn_in in BURN_INS
    }
    rolling_lengths = {
        burn_in: np.empty(rolling_covered[burn_in].shape, dtype=float)
        for burn_in in BURN_INS
    }
    split_shape = (
        config.trials,
        n_values.size,
        len(SPLIT_RATIOS),
        ALPHAS.size,
    )
    split_covered = np.empty(split_shape, dtype=bool)
    split_lengths = np.empty(split_shape, dtype=float)
    theta_star = np.zeros(config.d, dtype=float)
    theta_star[0] = 1.0
    child_seeds = np.random.SeedSequence(config.seed).spawn(config.trials)
    rolling_iterations = np.arange(BURN_INS[0], config.max_n + 1, dtype=int)
    started = time.perf_counter()

    for trial_index, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        train_x, train_y = sample_linear_data(
            config.max_n, config.d, theta_star, config.sigma, rng
        )
        test_x_array, test_y_array = sample_linear_data(
            1, config.d, theta_star, config.sigma, rng
        )
        test_x = test_x_array[0]
        test_y = float(test_y_array[0])
        rolling_thetas, rolling_radii = fit_min_norm_checkpoint_path(
            train_x, train_y, rolling_iterations
        )
        rolling_centers = rolling_thetas @ test_x

        for burn_in in BURN_INS:
            start = burn_in - BURN_INS[0]
            prefix_sizes = rolling_n[burn_in] - burn_in + 1
            (
                rolling_covered[burn_in][trial_index],
                rolling_lengths[burn_in][trial_index],
            ) = rolling_interval_results(
                rolling_centers[start:],
                rolling_radii[start:],
                test_y,
                prefix_sizes,
            )

        (
            split_covered[trial_index],
            split_lengths[trial_index],
        ) = split_results_for_stream(
            train_x, train_y, test_x, test_y, n_values
        )

        completed = trial_index + 1
        elapsed = time.perf_counter() - started
        remaining = elapsed * (config.trials - completed) / completed
        print(
            f"Completed stream {completed}/{config.trials} | "
            f"elapsed {elapsed:.1f}s | ETA {remaining:.1f}s",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lengths(
        output_dir / "fig4-1",
        config,
        n_values,
        rolling_n,
        rolling_lengths,
        split_lengths,
    )
    plot_coverage(
        output_dir / "fig4-2",
        config,
        n_values,
        rolling_n,
        rolling_covered,
        split_covered,
    )
    print(
        f"Completed in {time.perf_counter() - started:.1f}s. "
        f"Figures written to: {output_dir.resolve()}"
    )


def parse_args() -> tuple[Config, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=5_000)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--trials", "--M", dest="trials", type=int, default=100)
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    config = Config(
        max_n=args.max_n,
        d=args.d,
        sigma=args.sigma,
        trials=args.trials,
        checkpoint_step=args.checkpoint_step,
        seed=args.seed,
    )
    return config, args.output_dir


if __name__ == "__main__":
    experiment_config, experiment_output = parse_args()
    run_experiment(experiment_config, experiment_output)
