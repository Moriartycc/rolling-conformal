"""Section 4.2 experiments for rolling conformal multiclass logistic SGD.

The script compares gamma in {0.6, 0.8, 1.0} for the cross-entropy score
s_i^Ent and the running signed-margin score s_i^Mar.  It writes six
figures: two end-of-stream calibration plots, two coverage-evolution plots,
and two fixed-feature conditional-membership plots.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_COUNT = 5
SCORE_NAMES = ("cross_entropy", "running_margin")
DEFAULT_GAMMAS = (0.6, 0.8, 1.0)
DEFAULT_NOMINAL_LEVELS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
EVOLUTION_ALPHAS = (0.40, 0.20, 0.10, 0.05)

GAMMA_COLORS = ("#0072B2", "#D55E00", "#009E73")
GAMMA_MARKERS = ("o", "s", "^")
CLASS_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)


def prescribed_theta_star(d: int) -> np.ndarray:
    """Return the five oracle coefficient vectors as matrix rows."""
    if d < 5:
        raise ValueError("d must be at least 5.")
    theta_star = np.zeros((CLASS_COUNT, d), dtype=float)
    theta_star[0, 0] = 1.0
    theta_star[1, 1] = 1.0
    theta_star[2, 2] = 1.0
    theta_star[3, [0, 1]] = 0.5
    theta_star[4, [1, 2, 3, 4]] = 0.5
    return theta_star


def sample_multiclass_logistic_data(
    n: int,
    d: int,
    theta_star: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample X ~ N(0, I_d) and Y | X from the softmax model."""
    x = rng.standard_normal((n, d))
    probabilities = softmax(x @ theta_star.T)
    uniforms = rng.random(n)
    y = (uniforms[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    y = np.minimum(y, CLASS_COUNT - 1).astype(np.int64)
    return x, y


def train_sgd_path(
    x: np.ndarray,
    y: np.ndarray,
    eta0: float,
    t0: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the online SGD path and predictable cross-entropy scores."""
    n, d = x.shape
    weights_path = np.zeros((n + 1, CLASS_COUNT, d), dtype=float)
    calibration_scores = np.empty(n, dtype=float)
    weights = np.zeros((CLASS_COUNT, d), dtype=float)

    for obs_index in range(n):
        probabilities = softmax(weights @ x[obs_index])
        calibration_scores[obs_index] = -np.log(
            np.clip(probabilities[y[obs_index]], 1e-300, 1.0)
        )
        eta = eta0 / (t0 + obs_index + 1) ** gamma
        gradient_factor = probabilities.copy()
        gradient_factor[y[obs_index]] -= 1.0
        weights -= eta * gradient_factor[:, None] * x[obs_index][None, :]
        weights_path[obs_index + 1] = weights

    return weights_path, calibration_scores


def candidate_score_paths(
    weights_path: np.ndarray,
    test_x: np.ndarray,
    margin_window: int,
) -> dict[str, np.ndarray]:
    """Return predictable scores for every fixed feature and class."""
    n = weights_path.shape[0] - 1
    logits = np.einsum("tcd,qd->tqc", weights_path[:n], test_x, optimize=True)
    cross_entropy = np.logaddexp.reduce(logits, axis=2, keepdims=True) - logits

    snapshot_margins = np.empty_like(logits)
    for class_index in range(CLASS_COUNT):
        competitors = np.arange(CLASS_COUNT) != class_index
        snapshot_margins[:, :, class_index] = (
            np.max(logits[:, :, competitors], axis=2)
            - logits[:, :, class_index]
        )
    cumulative = np.cumsum(snapshot_margins, axis=0)
    prefix = np.concatenate((np.zeros_like(cumulative[:1]), cumulative), axis=0)
    paper_indices = np.arange(1, n + 1)
    starts = np.maximum(0, paper_indices - margin_window)
    running_margin = (prefix[paper_indices] - prefix[starts]) / np.minimum(
        paper_indices, margin_window
    )[:, None, None]
    return {
        "cross_entropy": cross_entropy,
        "running_margin": running_margin,
    }


def rolling_conformal_membership(
    candidate_scores: np.ndarray,
    calibration_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute prediction-set membership at stream sizes 0 through n."""
    comparisons = candidate_scores > calibration_scores[:, None, None]
    cumulative = np.cumsum(comparisons, axis=0, dtype=np.int64)
    n = candidate_scores.shape[0]
    counts = np.zeros(
        (n + 1, candidate_scores.shape[1], candidate_scores.shape[2]),
        dtype=np.int64,
    )
    counts[1:] = cumulative
    thresholds = (1.0 - alpha) * (np.arange(n + 1) + 1)
    return counts < thresholds[:, None, None]


def stream_tick_label(iteration: int) -> str:
    """Format large stream sizes compactly while keeping labels in LaTeX."""
    if iteration >= 1000:
        return rf"${iteration / 1000:g}\mathrm{{k}}$"
    return rf"${iteration:d}$"


def stream_ticks(n: int, include_zero: bool) -> np.ndarray:
    """Use the same compact landmark ticks as the Section 4.1 figures."""
    candidates = [0 if include_zero else 1, 1000, 5000, 10_000, n]
    return np.unique(
        np.asarray([tick for tick in candidates if 0 <= tick <= n], dtype=int)
    )


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 10_000
    d: int = 10
    trials: int = 100
    holdout_size: int = 500
    seed: int = 2026
    eta0: float = 1.0
    t0: float = 10.0
    gammas: tuple[float, ...] = DEFAULT_GAMMAS
    membership_alpha: float = 0.30
    nominal_levels: tuple[float, ...] = DEFAULT_NOMINAL_LEVELS
    evolution_alphas: tuple[float, ...] = EVOLUTION_ALPHAS
    margin_window: int = 100
    holdout_chunk_size: int = 100
    margin_block_size: int = 64


def make_test_points(d: int) -> tuple[np.ndarray, list[str]]:
    """Return four compact fixed features with distinct oracle behavior."""
    if d < 5:
        raise ValueError("d must be at least 5 for the fixed test points.")

    test_x = np.zeros((4, d), dtype=float)
    test_x[0, 0] = 4.0
    test_x[1, 0] = -4.0
    test_x[2, 2] = 4.0
    test_x[3, [3, 4]] = 3.0
    labels = [r"4e_1", r"-4e_1", r"4e_3", r"3(e_4+e_5)"]
    return test_x, labels


def running_margin_calibration_scores_blocked(
    weights_path: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    window: int,
) -> np.ndarray:
    """Compute the exact windowed signed logit-margin score in BLAS blocks.

    At observation i, the statistic averages competitor-minus-true-class
    margins under W_max(0,i-window),...,W_{i-1}.  Blocking limits temporary
    memory while preserving the exact score.
    """
    if window < 1:
        raise ValueError("window must be at least one.")
    n = x.shape[0]
    scores = np.empty(n, dtype=float)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        block_weight_start = max(0, start + 1 - window)
        available_weights = weights_path[block_weight_start:stop]
        logits = np.einsum(
            "tcd,bd->tbc",
            available_weights,
            x[start:stop],
            optimize=True,
        )
        for local_index, obs_index in enumerate(range(start, stop)):
            weight_start = max(0, obs_index + 1 - window)
            local_weight_start = weight_start - block_weight_start
            local_weight_stop = obs_index + 1 - block_weight_start
            available_logits = logits[
                local_weight_start:local_weight_stop, local_index
            ]
            true_class = y[obs_index]
            true_logits = available_logits[:, true_class]
            competitor_logits = np.max(
                available_logits[:, np.arange(CLASS_COUNT) != true_class],
                axis=1,
            )
            scores[obs_index] = np.mean(competitor_logits - true_logits)
    return scores


def true_label_score_paths(
    weights_path: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    margin_window: int,
) -> dict[str, np.ndarray]:
    """Return only the two score paths needed for true-label coverage.

    All K logits are still used for the softmax normalizer and argmax, but the
    K candidate-score paths are not materialized for hold-out observations.
    """
    if margin_window < 1:
        raise ValueError("margin_window must be at least one.")
    n = weights_path.shape[0] - 1
    logits = np.einsum(
        "tcd,qd->tqc", weights_path[:n], x, optimize=True
    )
    true_logits = np.take_along_axis(
        logits,
        np.broadcast_to(y[None, :, None], (n, y.size, 1)),
        axis=2,
    ).squeeze(axis=2)
    cross_entropy = np.logaddexp.reduce(logits, axis=2) - true_logits

    true_index = np.broadcast_to(y[None, :, None], (n, y.size, 1))
    np.put_along_axis(logits, true_index, -np.inf, axis=2)
    snapshot_margins = np.max(logits, axis=2) - true_logits
    cumulative_margins = np.cumsum(snapshot_margins, axis=0)
    prefix_margins = np.concatenate(
        (np.zeros_like(cumulative_margins[:1]), cumulative_margins), axis=0
    )
    paper_indices = np.arange(1, n + 1)
    window_starts = np.maximum(0, paper_indices - margin_window)
    window_sums = prefix_margins[paper_indices] - prefix_margins[window_starts]
    running_margin = window_sums / np.minimum(
        paper_indices, margin_window
    )[:, None]
    return {
        "cross_entropy": cross_entropy,
        "running_margin": running_margin,
    }


def _coverage_paths_for_holdout(
    weights_path: np.ndarray,
    calibration_scores: dict[str, np.ndarray],
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    coverage_levels: np.ndarray,
    chunk_size: int,
    margin_window: int,
) -> dict[str, np.ndarray]:
    """Return hold-out coverage paths without materializing all candidates."""
    n = weights_path.shape[0] - 1
    iterations = np.arange(n + 1, dtype=float)
    thresholds = coverage_levels[:, None] * (iterations[None, :] + 1.0)
    covered_counts = {
        score_name: np.zeros((coverage_levels.size, n + 1), dtype=np.int64)
        for score_name in SCORE_NAMES
    }
    covered_counts["cross_entropy"][:, 0] = holdout_x.shape[0]
    covered_counts["running_margin"][:, 0] = holdout_x.shape[0]

    for start in range(0, holdout_x.shape[0], chunk_size):
        stop = min(holdout_x.shape[0], start + chunk_size)
        chunk_x = holdout_x[start:stop]
        chunk_y = holdout_y[start:stop]
        true_score_paths = true_label_score_paths(
            weights_path, chunk_x, chunk_y, margin_window=margin_window
        )
        for score_name in SCORE_NAMES:
            comparisons = (
                true_score_paths[score_name]
                > calibration_scores[score_name][:, None]
            )
            cumulative = np.cumsum(comparisons, axis=0, dtype=np.int32)
            covered_counts[score_name][:, 1:] += np.sum(
                cumulative[None, :, :] < thresholds[:, 1:, None],
                axis=2,
            )

    return {
        score_name: covered_counts[score_name] / float(holdout_x.shape[0])
        for score_name in SCORE_NAMES
    }


def _mean_and_ci95(
    sums: np.ndarray,
    sum_squares: np.ndarray,
    trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = sums / float(trials)
    if trials < 2:
        return mean, np.zeros_like(mean)
    variance = np.maximum(
        (sum_squares - trials * mean**2) / float(trials - 1),
        0.0,
    )
    standard_error = np.sqrt(variance / float(trials))
    return mean, 1.96 * standard_error


def _setup_matplotlib():
    matplotlib_cache = Path(tempfile.gettempdir()) / "rolling-conformal-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save_figure(fig, output_stem: Path, plt) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_calibration(
    output_stem: Path,
    nominal_levels: np.ndarray,
    means: np.ndarray,
    ci95: np.ndarray,
    gammas: np.ndarray,
) -> None:
    """Plot end-of-stream coverage against nominal coverage for one score."""
    _, plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 7.1))
    ax.plot(
        [0.46, 1.0],
        [0.46, 1.0],
        color="#555555",
        linestyle="--",
        linewidth=1.9,
        label="Nominal coverage",
        zorder=1,
    )
    for gamma_index, gamma in enumerate(gammas):
        mean = means[gamma_index]
        interval = ci95[gamma_index]
        bounded_error = np.vstack(
            (np.minimum(interval, mean), np.minimum(interval, 1.0 - mean))
        )
        ax.errorbar(
            nominal_levels,
            mean,
            yerr=bounded_error,
            color=GAMMA_COLORS[gamma_index],
            marker=GAMMA_MARKERS[gamma_index],
            markersize=5.7,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=3.0,
            label=rf"RoCP empirical coverage when $\gamma={gamma:g}$",
            zorder=3,
        )
    ax.set_xlim(0.46, 1.0)
    ax.set_ylim(0.46, 1.005)
    ax.set_xlabel("Nominal coverage", fontsize=18)
    ax.set_ylabel("Empirical hold-out coverage", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", fontsize=11.5, frameon=True)
    fig.tight_layout()
    _save_figure(fig, output_stem, plt)


def plot_coverage_evolution(
    output_stem: Path,
    alphas: np.ndarray,
    means: np.ndarray,
    ci95: np.ndarray,
    gammas: np.ndarray,
    iterations: np.ndarray,
) -> None:
    """Plot four nominal levels, each with one curve per gamma."""
    _, plt = _setup_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 10.2), sharex=True, sharey=False
    )
    axes = np.asarray(axes).reshape(2, 2)
    displayed_iterations = iterations[1:]
    displayed_ticks = stream_ticks(int(displayed_iterations[-1]), include_zero=False)
    error_every = max(1, displayed_iterations.size // 15)
    for panel_index, ax in enumerate(axes.flat):
        alpha = float(alphas[panel_index])
        nominal_coverage = 1.0 - alpha
        ax.axhline(
            nominal_coverage,
            color="#555555",
            linestyle="--",
            linewidth=1.8,
            label="Nominal coverage",
            zorder=1,
        )
        for gamma_index, gamma in enumerate(gammas):
            mean = means[gamma_index, panel_index, 1:]
            interval = ci95[gamma_index, panel_index, 1:]
            bounded_error = np.vstack(
                (np.minimum(interval, mean), np.minimum(interval, 1.0 - mean))
            )
            ax.errorbar(
                displayed_iterations,
                mean,
                yerr=bounded_error,
                color=GAMMA_COLORS[gamma_index],
                marker=GAMMA_MARKERS[gamma_index],
                markersize=4.8,
                linewidth=1.9,
                elinewidth=1.0,
                capsize=2.5,
                markevery=error_every,
                errorevery=error_every,
                label=rf"RoCP empirical coverage when $\gamma={gamma:g}$",
                zorder=3,
            )
        ax.set_xlim(1, displayed_iterations[-1])
        ax.set_xticks(displayed_ticks)
        ax.set_xticklabels(
            [stream_tick_label(int(tick)) for tick in displayed_ticks]
        )
        ax.set_ylim(1.0 - 2.0 * alpha, 1.0)
        ax.set_title(rf"$\alpha={alpha:g}$", fontsize=28, pad=10)
        ax.tick_params(axis="both", labelsize=19)
        if panel_index % 2 == 1:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelleft=False, labelright=True)
        ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.8)

    axes[0, 0].legend(loc="lower right", fontsize=14.5, frameon=True)
    fig.supxlabel(
        r"$\mathrm{Data\ stream\ size}\ i\ \mathrm{(from\ 1\ to}\ n\mathrm{)}$",
        fontsize=30,
        y=0.025,
    )
    fig.supylabel(
        r"$\mathrm{Empirical\ hold\!\!-\!out\ coverage}$",
        fontsize=30,
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
    _save_figure(fig, output_stem, plt)


def plot_membership_by_gamma(
    output_stem: Path,
    score_name: str,
    inclusion_rates: np.ndarray,
    gammas: np.ndarray,
    test_labels: list[str],
    oracle_probabilities: np.ndarray,
) -> None:
    """Plot four test-point rows and three gamma columns for one score."""
    _, plt = _setup_matplotlib()
    import matplotlib.patheffects as path_effects
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker

    row_count = len(test_labels)
    fig, axes = plt.subplots(
        row_count,
        gammas.size,
        figsize=(13.8, 3.55 * row_count),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(row_count, gammas.size)
    n = inclusion_rates.shape[1] - 1
    displayed_ticks = stream_ticks(n, include_zero=True)
    time_edges = np.arange(n + 2, dtype=float) - 0.5
    for row_index in range(row_count):
        for gamma_index, gamma in enumerate(gammas):
            ax = axes[row_index, gamma_index]
            rates = inclusion_rates[gamma_index, :, row_index, :]
            for class_index in range(CLASS_COUNT):
                center = class_index + 1
                ax.axhspan(
                    center - 0.36,
                    center + 0.36,
                    facecolor="#F0F2F4",
                    edgecolor="none",
                    zorder=0,
                )
                step_rates = np.append(rates[:, class_index], rates[-1, class_index])
                half_width = 0.36 * step_rates
                ax.fill_between(
                    time_edges,
                    center - half_width,
                    center + half_width,
                    step="post",
                    facecolor=CLASS_COLORS[class_index],
                    edgecolor="none",
                    zorder=2,
                )
            ax.set_xlim(-0.5, n + 0.5)
            ax.set_xticks(displayed_ticks)
            ax.set_xticklabels(
                [stream_tick_label(int(tick)) for tick in displayed_ticks]
            )
            ax.set_ylim(0.5, CLASS_COUNT + 0.5)
            ax.set_yticks(np.arange(1, CLASS_COUNT + 1))
            ax.tick_params(axis="both", labelsize=15)
            ax.grid(False)
            if row_index == 0:
                ax.set_title(rf"$\gamma={gamma:g}$", fontsize=22, pad=9)

    xlabel_y = 0.004
    fig.supxlabel(r"Data stream size $i$", fontsize=28, y=xlabel_y)
    fig.supylabel(r"Included class $y$", fontsize=28, x=0.010)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.15,
        top=0.95,
        wspace=0.07,
        hspace=0.72,
    )

    # Place one compact two-line caption beneath each three-panel row.
    # Probability glyph thickness encodes absolute oracle probability on the
    # same scale in every row.
    first_row_bottom = axes[0, 0].get_position().y0
    second_row_top = axes[1, 0].get_position().y1
    standard_caption_gap = 0.5 * (first_row_bottom - second_row_top)
    tick_label_clearance = (15.0 / 72.0) / fig.get_figheight() + 0.010
    caption_fontsize = 22
    for row_index in range(row_count):
        left_box = axes[row_index, 0].get_position()
        right_box = axes[row_index, -1].get_position()
        caption_center_x = 0.5 * (left_box.x0 + right_box.x1)
        if row_index < row_count - 1:
            lower_panel_top = axes[row_index + 1, 0].get_position().y1
            caption_y = 0.5 * (left_box.y0 + lower_panel_top)
        else:
            caption_y = (
                left_box.y0 - standard_caption_gap - tick_label_clearance
            )

        probability_row = oracle_probabilities[row_index]
        point_caption = TextArea(
            rf"Test point $X_{{n+1}}={test_labels[row_index]}$ "
            "with oracle probabilities",
            textprops={"fontsize": caption_fontsize},
        )
        probability_items = [
            TextArea(
                r"$p^\star=[$", textprops={"fontsize": caption_fontsize}
            ),
        ]
        for probability_index, probability in enumerate(probability_row):
            # Use one absolute scale across all rows: a probability of 0.32
            # should not be rendered as strongly as a probability of 0.84.
            stroke_width = 0.10 + 1.35 * float(probability)
            probability_items.append(
                TextArea(
                    rf"${probability:.2f}$",
                    textprops={
                        "fontsize": caption_fontsize,
                        "path_effects": [
                            path_effects.withStroke(
                                linewidth=stroke_width,
                                foreground="black",
                            )
                        ],
                    },
                )
            )
            if probability_index < len(probability_row) - 1:
                probability_items.append(
                    TextArea(
                        r"$,$ ", textprops={"fontsize": caption_fontsize}
                    )
                )
        probability_items.append(
            TextArea(
                r"$]^\top$", textprops={"fontsize": caption_fontsize}
            )
        )
        probability_caption = HPacker(
            children=probability_items,
            align="center",
            pad=0,
            sep=0,
        )
        stacked_caption = VPacker(
            children=[point_caption, probability_caption],
            align="center",
            pad=0,
            sep=4,
        )
        fig.add_artist(
            AnnotationBbox(
                stacked_caption,
                (caption_center_x, caption_y),
                xycoords=fig.transFigure,
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )
    _save_figure(fig, output_stem, plt)


def _validate_config(config: ExperimentConfig) -> None:
    if config.n < 1 or config.d < 5:
        raise ValueError("Need n >= 1 and d >= 5.")
    if config.trials < 2 or config.holdout_size < 1:
        raise ValueError("Need at least two trials and one hold-out point.")
    if config.margin_window < 1:
        raise ValueError("margin_window must be at least one.")
    if config.eta0 <= 0 or config.t0 < 0:
        raise ValueError("eta0 must be positive and t0 must be nonnegative.")
    if config.holdout_chunk_size < 1 or config.margin_block_size < 1:
        raise ValueError("Chunk and block sizes must be positive.")
    gammas = np.asarray(config.gammas, dtype=float)
    if gammas.shape != (3,) or np.any((gammas <= 0.5) | (gammas > 1.0)):
        raise ValueError("Provide exactly three gamma values in (0.5, 1].")
    if not (0.0 < config.membership_alpha < 1.0):
        raise ValueError("membership_alpha must lie in (0,1).")
    nominal_levels = np.asarray(config.nominal_levels, dtype=float)
    evolution_alphas = np.asarray(config.evolution_alphas, dtype=float)
    if (
        nominal_levels.ndim != 1
        or nominal_levels.size < 2
        or np.any(np.diff(nominal_levels) <= 0)
        or np.any((nominal_levels <= 0) | (nominal_levels >= 1))
    ):
        raise ValueError("nominal_levels must be increasing and lie in (0,1).")
    if evolution_alphas.shape != (4,) or np.any(
        (evolution_alphas <= 0) | (evolution_alphas >= 1)
    ):
        raise ValueError("Provide exactly four evolution alpha values in (0,1).")


def run_experiment(config: ExperimentConfig, output_dir: Path) -> None:
    """Run the complete Section 4.2 experiment and write six figures."""
    _validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_star = prescribed_theta_star(config.d)
    test_x, test_labels = make_test_points(config.d)
    oracle_probabilities = softmax(test_x @ theta_star.T)
    gammas = np.asarray(config.gammas, dtype=float)
    nominal_levels = np.asarray(config.nominal_levels, dtype=float)
    evolution_alphas = np.asarray(config.evolution_alphas, dtype=float)
    evolution_levels = 1.0 - evolution_alphas
    coverage_levels = np.unique(np.concatenate((nominal_levels, evolution_levels)))
    nominal_indices = np.searchsorted(coverage_levels, nominal_levels)
    evolution_indices = np.searchsorted(coverage_levels, evolution_levels)
    iterations = np.arange(config.n + 1)

    end_sums = {
        score: np.zeros((gammas.size, nominal_levels.size), dtype=float)
        for score in SCORE_NAMES
    }
    end_sum_squares = {
        score: np.zeros_like(end_sums[score]) for score in SCORE_NAMES
    }
    evolution_sums = {
        score: np.zeros(
            (gammas.size, evolution_alphas.size, config.n + 1), dtype=float
        )
        for score in SCORE_NAMES
    }
    evolution_sum_squares = {
        score: np.zeros_like(evolution_sums[score]) for score in SCORE_NAMES
    }
    membership_sums = {
        score: np.zeros(
            (gammas.size, config.n + 1, test_x.shape[0], CLASS_COUNT),
            dtype=np.int32,
        )
        for score in SCORE_NAMES
    }

    child_seeds = np.random.SeedSequence(config.seed).spawn(config.trials)
    started = time.perf_counter()
    report_every = max(1, config.trials // 100)
    for trial_index, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        train_x, train_y = sample_multiclass_logistic_data(
            config.n, config.d, theta_star, rng
        )
        holdout_x, holdout_y = sample_multiclass_logistic_data(
            config.holdout_size, config.d, theta_star, rng
        )

        for gamma_index, gamma in enumerate(gammas):
            weights_path, calibration_ce = train_sgd_path(
                train_x,
                train_y,
                eta0=config.eta0,
                t0=config.t0,
                gamma=float(gamma),
            )
            calibration_margin = running_margin_calibration_scores_blocked(
                weights_path,
                train_x,
                train_y,
                config.margin_block_size,
                config.margin_window,
            )
            calibration_scores = {
                "cross_entropy": calibration_ce,
                "running_margin": calibration_margin,
            }
            holdout_paths = _coverage_paths_for_holdout(
                weights_path,
                calibration_scores,
                holdout_x,
                holdout_y,
                coverage_levels,
                config.holdout_chunk_size,
                config.margin_window,
            )
            fixed_candidate_scores = candidate_score_paths(
                weights_path, test_x, margin_window=config.margin_window
            )
            for score_name in SCORE_NAMES:
                end_values = holdout_paths[score_name][nominal_indices, -1]
                evolution_values = holdout_paths[score_name][evolution_indices]
                end_sums[score_name][gamma_index] += end_values
                end_sum_squares[score_name][gamma_index] += end_values**2
                evolution_sums[score_name][gamma_index] += evolution_values
                evolution_sum_squares[score_name][gamma_index] += (
                    evolution_values**2
                )
                membership = rolling_conformal_membership(
                    fixed_candidate_scores[score_name],
                    calibration_scores[score_name],
                    config.membership_alpha,
                )
                membership_sums[score_name][gamma_index] += membership

        completed = trial_index + 1
        if completed == config.trials or completed % report_every == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed * (config.trials - completed) / completed
            print(
                f"Completed trial {completed}/{config.trials} | "
                f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s",
                flush=True,
            )

    end_means: dict[str, np.ndarray] = {}
    end_ci95: dict[str, np.ndarray] = {}
    evolution_means: dict[str, np.ndarray] = {}
    evolution_ci95: dict[str, np.ndarray] = {}
    inclusion_rates: dict[str, np.ndarray] = {}
    for score_name in SCORE_NAMES:
        end_means[score_name], end_ci95[score_name] = _mean_and_ci95(
            end_sums[score_name], end_sum_squares[score_name], config.trials
        )
        evolution_means[score_name], evolution_ci95[score_name] = _mean_and_ci95(
            evolution_sums[score_name],
            evolution_sum_squares[score_name],
            config.trials,
        )
        inclusion_rates[score_name] = (
            membership_sums[score_name] / float(config.trials)
        )

    figure_stems = {
        "cross_entropy": (
            output_dir / "fig2-1",
            output_dir / "fig2-3",
            output_dir / "fig2-5",
        ),
        "running_margin": (
            output_dir / "fig2-2",
            output_dir / "fig2-4",
            output_dir / "fig2-6",
        ),
    }
    for score_name in SCORE_NAMES:
        calibration_stem, evolution_stem, membership_stem = figure_stems[score_name]
        plot_coverage_calibration(
            calibration_stem,
            nominal_levels,
            end_means[score_name],
            end_ci95[score_name],
            gammas,
        )
        plot_coverage_evolution(
            evolution_stem,
            evolution_alphas,
            evolution_means[score_name],
            evolution_ci95[score_name],
            gammas,
            iterations,
        )
        plot_membership_by_gamma(
            membership_stem,
            score_name,
            inclusion_rates[score_name],
            gammas,
            test_labels,
            oracle_probabilities,
        )


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Section 4.2 RoCP experiments for multiclass logistic SGD."
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--trials", "--M", dest="trials", type=int, default=100)
    parser.add_argument(
        "--holdout-size", "--N", dest="holdout_size", type=int, default=500
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eta0", type=float, default=1.0)
    parser.add_argument("--t0", type=float, default=10.0)
    parser.add_argument("--gammas", type=_parse_float_tuple, default=DEFAULT_GAMMAS)
    parser.add_argument("--membership-alpha", type=float, default=0.30)
    parser.add_argument(
        "--nominal-levels",
        type=_parse_float_tuple,
        default=DEFAULT_NOMINAL_LEVELS,
    )
    parser.add_argument(
        "--evolution-alphas",
        type=_parse_float_tuple,
        default=EVOLUTION_ALPHAS,
    )
    parser.add_argument("--margin-window", type=int, default=100)
    parser.add_argument("--holdout-chunk-size", type=int, default=100)
    parser.add_argument("--margin-block-size", type=int, default=64)
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
        trials=args.trials,
        holdout_size=args.holdout_size,
        seed=args.seed,
        eta0=args.eta0,
        t0=args.t0,
        gammas=args.gammas,
        membership_alpha=args.membership_alpha,
        nominal_levels=args.nominal_levels,
        evolution_alphas=args.evolution_alphas,
        margin_window=args.margin_window,
        holdout_chunk_size=args.holdout_chunk_size,
        margin_block_size=args.margin_block_size,
    )
    run_experiment(config, args.output_dir)
    print(f"\nFigures written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
