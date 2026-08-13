"""One-pass MNIST experiment for rolling conformal prediction.

The network sees each training observation exactly once.  At iteration i, the
cross-entropy calibration score and all hold-out scores are evaluated using the
network trained on Z_{<i}; the network is then updated by one pure-SGD step on
Z_i.  Coverage is estimated by averaging inclusion indicators over M fixed
MNIST test observations.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NOMINAL_LEVELS = tuple(np.arange(0.50, 1.00, 0.05).round(2))
EVOLUTION_ALPHAS = (0.40, 0.20, 0.10, 0.05)
ROCP_COLOR = "#0072B2"
NOMINAL_COLOR = "#555555"


@dataclass(frozen=True)
class Config:
    n: int = 60_000
    M: int = 1_000
    seed: int = 2026
    eta0: float = 50.0
    t0: float = 5_000.0
    gamma: float = 1.0
    nominal_levels: tuple[float, ...] = DEFAULT_NOMINAL_LEVELS
    evolution_alphas: tuple[float, ...] = EVOLUTION_ALPHAS
    device: str = "auto"
    num_threads: int = 0
    download: bool = True


def stream_tick_label(iteration: int) -> str:
    if iteration >= 1000:
        return rf"${iteration / 1000:g}\mathrm{{k}}$"
    return rf"${iteration:d}$"


def stream_ticks(n: int) -> np.ndarray:
    candidates = [1, 1000, 5000, 10_000, 30_000, 60_000, n]
    return np.unique(
        np.asarray([tick for tick in candidates if 1 <= tick <= n], dtype=int)
    )


def _setup_matplotlib():
    matplotlib_cache = Path(tempfile.gettempdir()) / "rolling-conformal-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, output_stem: Path, plt) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_end_coverage(
    output_stem: Path,
    nominal_levels: np.ndarray,
    empirical_coverage: np.ndarray,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 7.1))
    ax.plot(
        [0.46, 1.0],
        [0.46, 1.0],
        color=NOMINAL_COLOR,
        linestyle="--",
        linewidth=1.9,
        label="Nominal coverage",
        zorder=1,
    )
    ax.plot(
        nominal_levels,
        empirical_coverage,
        color=ROCP_COLOR,
        marker="o",
        markersize=6.2,
        linewidth=2.2,
        label="RoCP empirical coverage",
        zorder=3,
    )
    ax.set_xlim(0.46, 1.0)
    ax.set_ylim(0.46, 1.005)
    ax.set_xlabel("Nominal coverage", fontsize=18)
    ax.set_ylabel("Empirical hold-out coverage", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, color="#D9DDE1", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", fontsize=13, frameon=True)
    fig.tight_layout()
    _save_figure(fig, output_stem, plt)


def plot_coverage_evolution(
    output_stem: Path,
    alphas: np.ndarray,
    coverage_paths: np.ndarray,
    iterations: np.ndarray,
) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 10.2), sharex=True, sharey=False
    )
    axes = np.asarray(axes).reshape(2, 2)
    displayed_iterations = iterations[1:]
    displayed_ticks = stream_ticks(int(displayed_iterations[-1]))

    for panel_index, ax in enumerate(axes.flat):
        alpha = float(alphas[panel_index])
        nominal_coverage = 1.0 - alpha
        ax.axhline(
            nominal_coverage,
            color=NOMINAL_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Nominal coverage",
            zorder=1,
        )
        ax.plot(
            displayed_iterations,
            coverage_paths[panel_index, 1:],
            color=ROCP_COLOR,
            linewidth=2.0,
            label="RoCP empirical coverage",
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


def _resolve_device(requested: str, torch):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _set_reproducibility(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_model(torch):
    nn = torch.nn

    class SmallMNISTCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(8, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x):
            return self.classifier(self.features(x))

    return SmallMNISTCNN()


def _load_mnist(config: Config, data_dir: Path, torch, torchvision):
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=config.download,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=config.download,
    )
    if config.n > len(train_dataset):
        raise ValueError(
            f"n={config.n} exceeds the {len(train_dataset)} MNIST training examples."
        )
    if config.M > len(test_dataset):
        raise ValueError(
            f"M={config.M} exceeds the {len(test_dataset)} MNIST test examples."
        )

    generator = torch.Generator().manual_seed(config.seed)
    train_order = torch.randperm(len(train_dataset), generator=generator)[: config.n]
    test_order = torch.randperm(len(test_dataset), generator=generator)[: config.M]
    # Keep the full training stream in compact uint8 form.  Converting the
    # entire 60K stream to float up front is slower than normalizing the single
    # image used by each batch-size-one SGD update.
    train_images = train_dataset.data.unsqueeze(1)
    holdout_images = test_dataset.data[test_order].unsqueeze(1).float().div_(255.0)
    holdout_images.sub_(0.1307).div_(0.3081)
    train_labels = train_dataset.targets
    holdout_labels = test_dataset.targets[test_order].clone()
    return train_images, train_labels, train_order, holdout_images, holdout_labels


def _validate_config(config: Config) -> None:
    if not (1 <= config.n <= 60_000):
        raise ValueError("n must lie in [1, 60000].")
    if not (1 <= config.M <= 10_000):
        raise ValueError("M must lie in [1, 10000].")
    if config.eta0 <= 0 or config.t0 <= 0 or config.gamma <= 0:
        raise ValueError("eta0, t0, and gamma must be positive.")
    if len(config.evolution_alphas) != 4:
        raise ValueError("Exactly four evolution alpha values are required.")


def run_experiment(config: Config, output_dir: Path, data_dir: Path) -> None:
    _validate_config(config)
    try:
        import torch
        import torch.nn.functional as F
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires PyTorch and torchvision. "
            "Install the packages listed in requirements.txt."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_reproducibility(config.seed, torch)
    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    device = _resolve_device(config.device, torch)
    (
        train_images,
        train_labels,
        train_order,
        holdout_images,
        holdout_labels,
    ) = _load_mnist(config, data_dir, torch, torchvision)

    model = _build_model(torch).to(device)
    holdout_images = holdout_images.to(device)
    holdout_labels = holdout_labels.to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    nominal_levels = np.asarray(config.nominal_levels, dtype=float)
    evolution_alphas = np.asarray(config.evolution_alphas, dtype=float)
    evolution_levels = 1.0 - evolution_alphas
    evolution_paths = np.ones((evolution_alphas.size, config.n + 1), dtype=float)
    exceedance_counts = torch.zeros(config.M, dtype=torch.int64, device=device)
    started = time.perf_counter()
    report_every = max(1, config.n // 100)
    for zero_index in range(config.n):
        iteration = zero_index + 1
        train_index = int(train_order[zero_index])
        eta_i = config.eta0 / (config.t0 + iteration) ** config.gamma

        model.zero_grad(set_to_none=True)
        train_image = (
            train_images[train_index : train_index + 1]
            .float()
            .div(255.0)
            .sub(0.1307)
            .div(0.3081)
        )
        train_logits = model(train_image)
        train_loss = F.cross_entropy(
            train_logits,
            train_labels[train_index : train_index + 1],
            reduction="mean",
        )
        with torch.no_grad():
            holdout_logits = model(holdout_images)
            holdout_scores = F.cross_entropy(
                holdout_logits,
                holdout_labels,
                reduction="none",
            )
            exceedance_counts += holdout_scores > train_loss.detach()
            for alpha_index, coverage_level in enumerate(evolution_levels):
                evolution_paths[alpha_index, iteration] = float(
                    (
                        exceedance_counts
                        < coverage_level * float(iteration + 1)
                    )
                    .float()
                    .mean()
                    .cpu()
                )

        train_loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(parameter.grad, alpha=-eta_i)

        if iteration == 1 or iteration % report_every == 0 or iteration == config.n:
            elapsed = time.perf_counter() - started
            rate = iteration / elapsed
            remaining = (config.n - iteration) / rate if rate > 0 else float("inf")
            print(
                f"[{iteration:>6d}/{config.n}] "
                f"{100.0 * iteration / config.n:6.2f}% | "
                f"elapsed {elapsed:8.1f}s | ETA {remaining:8.1f}s",
                flush=True,
            )

    final_counts = exceedance_counts.detach().cpu().numpy()
    end_coverage = np.asarray(
        [
            np.mean(final_counts < level * float(config.n + 1))
            for level in nominal_levels
        ],
        dtype=float,
    )
    iterations = np.arange(config.n + 1)
    plot_end_coverage(
        output_dir / "fig3-1",
        nominal_levels,
        end_coverage,
    )
    plot_coverage_evolution(
        output_dir / "fig3-2",
        evolution_alphas,
        evolution_paths,
        iterations,
    )

    print(f"Figures written to {output_dir.resolve()}", flush=True)


def parse_args() -> tuple[Config, Path, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60_000)
    parser.add_argument("--M", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eta0", type=float, default=50.0)
    parser.add_argument("--t0", type=float, default=5_000.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require MNIST to exist locally rather than downloading it.",
    )
    args = parser.parse_args()
    config = Config(
        n=args.n,
        M=args.M,
        seed=args.seed,
        eta0=args.eta0,
        t0=args.t0,
        gamma=args.gamma,
        device=args.device,
        num_threads=args.num_threads,
        download=not args.no_download,
    )
    return config, args.output_dir, args.data_dir


if __name__ == "__main__":
    experiment_config, experiment_output, experiment_data = parse_args()
    run_experiment(experiment_config, experiment_output, experiment_data)
