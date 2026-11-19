"""Rebuild TensorBoard learning curves + paper figures from completed COMS4705 runs.

NO training or Test evaluation is performed. Uses per-epoch metrics_history.json.

Usage:
  pip install tensorboard matplotlib
  python visualize_training.py                 # 2k and 10k, 3 seeds, 10 epochs
  tensorboard --logdir outputs/training_visualizations/tensorboard_means --port 6006
  python visualize_training.py --sizes 2k 10k 40k
  python visualize_training.py --no-std-shading  # mean curves only

Exports a 2x3 validation metric grid by default (2K/10K x 3 metrics).

The script follows outputs/experiments/<size>/<strategy>/seed_<seed>/completed.json
so an older attempt_001 is not mistaken for a newer 10-epoch experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGIES = ("candidate_random", "hard", "global_random")
LABELS = {"candidate_random": "Candidate Random", "hard": "Hard Negative",
          "global_random": "Global Random"}
METRICS = {"recall_at_2": "Recall@2", "paragraph_em_at_2": "Paragraph EM@2",
           "ndcg_at_5": "nDCG@5", "train_bce_loss": "Training BCE Loss"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect(model_root: Path, sizes: list[str], seeds: list[int], epochs: int):
    runs = {}
    failures = []
    for size in sizes:
        for strategy in STRATEGIES:
            for seed in seeds:
                key = (size, strategy, seed)
                run_dir = model_root / size / strategy / f"seed_{seed}"
                try:
                    marker = load_json(run_dir / "completed.json")
                    attempt_name = marker["attempt_dir"]
                    if not (isinstance(attempt_name, str) and attempt_name.startswith("attempt_")
                            and Path(attempt_name).name == attempt_name):
                        raise ValueError("invalid attempt directory in completed.json")
                    attempt = run_dir / attempt_name
                    summary = load_json(attempt / "training_summary.json")
                    history = load_json(attempt / "metrics_history.json")
                    for field, target in (("size", size), ("strategy", strategy), ("seed", seed), ("epochs", epochs)):
                        if summary.get(field) != target:
                            raise ValueError(f"{field}={summary.get(field)!r}; expected {target!r}")
                    if summary.get("debug_subset") is not False:
                        raise ValueError("this is a debug run or missing debug_subset flag")
                    if not isinstance(history, list) or len(history) != epochs:
                        raise ValueError(f"expected {epochs} epoch records, got {len(history) if isinstance(history, list) else type(history)}")
                    if [r.get("epoch") for r in history] != list(range(1, epochs + 1)):
                        raise ValueError("epochs must be exactly 1..N")
                    for rec in history:
                        for metric in METRICS:
                            if not isinstance(rec.get(metric), (int, float)) or not math.isfinite(rec[metric]):
                                raise ValueError(f"missing or non-finite {metric} in epoch {rec['epoch']}")
                    runs[key] = history
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    failures.append(f"  {size}/{strategy}/seed_{seed}: {exc}")
    if failures:
        raise SystemExit("ERROR: incomplete or mismatched experiment runs:\n" + "\n".join(failures) +
                         "\nEnsure completed.json points to the 10-epoch attempt. No output was written.")
    return runs


def summarize(runs):
    groups = defaultdict(lambda: defaultdict(list))
    for (size, strategy, seed), hist in runs.items():
        for record in hist:
            groups[(size, strategy)][record["epoch"]].append(record)
    aggregates = {}
    for key, epochs in groups.items():
        agg_hist = []
        for epoch in sorted(epochs):
            values = epochs[epoch]
            row = {"epoch": epoch, "n_seeds": len(values)}
            for metric in METRICS:
                seq = [v[metric] for v in values]
                row[metric] = statistics.mean(seq)
                row[metric + "_sd"] = statistics.stdev(seq) if len(seq) > 1 else 0.0
            agg_hist.append(row)
        aggregates[key] = agg_hist
    return aggregates


def save_csv(runs, aggregates, output: Path):
    fields = ["size", "strategy", "seed", "epoch", *METRICS]
    with (output / "epoch_metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (size, strategy, seed), hist in sorted(runs.items()):
            for record in hist:
                writer.writerow({"size": size, "strategy": strategy, "seed": seed,
                                 "epoch": record["epoch"], **{m: record[m] for m in METRICS}})
    fields = ["size", "strategy", "epoch", "n_seeds"] + [f"{m}{suffix}" for m in METRICS for suffix in ("", "_sd")]
    with (output / "epoch_metrics_mean_sd.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (size, strategy), hist in sorted(aggregates.items()):
            for record in hist:
                writer.writerow({"size": size, "strategy": strategy, **record})


def save_tensorboard(aggregates, out: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter  # alternative event writer
        except ImportError as exc:
            raise RuntimeError("TensorBoard writer is missing. Run: python -m pip install tensorboard") from exc

    tag = {"recall_at_2": "Validation/Recall@2", "paragraph_em_at_2": "Validation/EM@2",
           "ndcg_at_5": "Validation/nDCG@5", "train_bce_loss": "Training/BCE_Loss"}
    # ONLY mean curves are written into this dedicated TensorBoard folder.
    # Using a fresh path keeps old per-seed logs from appearing in this dashboard.
    for (size, strategy), hist in sorted(aggregates.items()):
        with SummaryWriter(str(out / size / strategy)) as w:
            for record in hist:
                for metric, tb_tag in tag.items():
                    w.add_scalar(tb_tag, record[metric], record["epoch"])


def _mpl():
    """Use a noninteractive Matplotlib backend for reproducible paper figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is not installed. Run: python -m pip install matplotlib") from exc
    return plt


# Consistent strategy colors in every panel (Matplotlib's standard tab colors).
STRATEGY_COLORS = {
    "candidate_random": "tab:blue",
    "hard": "tab:orange",
    "global_random": "tab:green",
}
VAL_METRICS = (
    ("recall_at_2", "Recall@2"),
    ("paragraph_em_at_2", "Paragraph EM@2"),
    ("ndcg_at_5", "nDCG@5"),
)
# A shared y-scale within each metric column makes 2K and 10K comparable.
VAL_LIMITS = {
    "recall_at_2": (65, 95),
    "paragraph_em_at_2": (40, 90),
    "ndcg_at_5": (80, 100),
}


def draw_mean_curve(ax, aggregates, size, strategy, metric, show_std=True):
    """Plot one strategy's mean across seeds, plus optional ±1 sample SD."""
    rows = aggregates[(size, strategy)]
    x = [r["epoch"] for r in rows]
    factor = 1 if metric == "train_bce_loss" else 100
    means = [r[metric] * factor for r in rows]
    sds = [r[metric + "_sd"] * factor for r in rows]
    color = STRATEGY_COLORS[strategy]
    ax.plot(x, means, color=color, marker="o", markersize=2.4,
            linewidth=1.7, label=LABELS[strategy])
    if show_std:
        lo = [max(0, m - sd) for m, sd in zip(means, sds)]
        hi = [m + sd for m, sd in zip(means, sds)]
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)


def plot_metrics_grid(aggregates, sizes, output, show_std=True):
    """Paper-ready N x 3 grid; N=2 produces the requested 2K/10K 2x3 figure."""
    plt = _mpl()
    fig, axes = plt.subplots(len(sizes), 3, figsize=(12.6, 3.15 * len(sizes)),
                             squeeze=False, sharex=True)
    for row_idx, size in enumerate(sizes):
        for col_idx, (metric, metric_title) in enumerate(VAL_METRICS):
            ax = axes[row_idx][col_idx]
            for strategy in STRATEGIES:
                draw_mean_curve(ax, aggregates, size, strategy, metric, show_std)
            ax.set_ylim(*VAL_LIMITS[metric])
            ax.set_title(metric_title if row_idx == 0 else "", fontsize=11, pad=9)
            ax.set_xticks([r["epoch"] for r in aggregates[(size, STRATEGIES[0])]])
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=0.23, linewidth=0.7)
            if col_idx == 0:
                ax.set_ylabel(f"{size.upper()} training\nScore (%)", fontsize=10)
            if row_idx == len(sizes) - 1:
                ax.set_xlabel("Epoch", fontsize=9)
            # Distinct row annotations for small double-column paper figures.
            if col_idx == 2:
                ax.text(1.02, 0.5, size.upper(), rotation=270, va="center",
                        ha="left", transform=ax.transAxes, fontsize=10,
                        fontweight="bold", color="0.3")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.93, top=0.92, bottom=0.12,
                        wspace=0.25, hspace=0.28)
    stem = "validation_metrics_grid"
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(aggregates, sizes, metric, output, show_std=True):
    """Keep the original Recall/Loss figures and export standalone EM/nDCG too."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.4 * len(sizes), 3.6),
                             sharey=metric != "train_bce_loss", squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for strategy in STRATEGIES:
            draw_mean_curve(ax, aggregates, size, strategy, metric, show_std)
        ax.set_title(f"{size.upper()} training questions", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_xticks([r["epoch"] for r in aggregates[(size, STRATEGIES[0])]])
        if ax is axes[0][0]:
            ax.set_ylabel("Training BCE loss" if metric == "train_bce_loss" else METRICS[metric] + " (%)")
        if metric in VAL_LIMITS:
            ax.set_ylim(*VAL_LIMITS[metric])
        ax.grid(alpha=0.23)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    filenames = {
        "recall_at_2": "validation_recall_at_2",
        "paragraph_em_at_2": "validation_em_at_2",
        "ndcg_at_5": "validation_ndcg_at_5",
        "train_bce_loss": "training_bce_loss",
    }
    stem = filenames[metric]
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-root", type=Path, default=ROOT / "outputs" / "experiments")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "training_visualizations")
    p.add_argument("--sizes", nargs="+", choices=("2k", "10k", "40k"), default=["2k", "10k"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--no-tensorboard", action="store_true", help="Only create PDF/PNG/CSV; skip TensorBoard events")
    p.add_argument("--no-std-shading", action="store_true", help="Paper figures: only 3 seed-mean lines, no SD shading")
    args = p.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.seeds)) != len(args.seeds):
        p.error("Sizes/seeds cannot contain duplicates")
    if args.epochs < 1:
        p.error("Epochs must be >= 1")
    runs = collect(args.model_root, args.sizes, args.seeds, args.epochs)
    aggregates = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(runs, aggregates, args.output_dir)
    # The main 2x3 paper figure: 2K / 10K rows and Recall, EM, nDCG columns.
    plot_metrics_grid(aggregates, args.sizes, args.output_dir, show_std=not args.no_std_shading)
    for metric in ("recall_at_2", "paragraph_em_at_2", "ndcg_at_5", "train_bce_loss"):
        plot_metric(aggregates, args.sizes, metric, args.output_dir, show_std=not args.no_std_shading)
    if not args.no_tensorboard:
        save_tensorboard(aggregates, args.output_dir / "tensorboard_means")
    print(f"Read {len(runs)} completed runs ({len(args.sizes)} training sizes x 3 strategies x {len(args.seeds)} seeds)")
    print("Every plotted/logged line is the arithmetic mean across seeds, never an individual seed.")
    print(f"Main paper figure: {args.output_dir / 'validation_metrics_grid.pdf'}")
    print("Also exported standalone Recall, EM, nDCG and Training Loss figures (PDF + PNG).")
    if not args.no_tensorboard:
        print(f"TensorBoard mean-only logs: {args.output_dir / 'tensorboard_means'}")
        print("Open with: tensorboard --logdir outputs/training_visualizations/tensorboard_means --port 6006")
    print("No models were trained and no Test results were used.")


if __name__ == "__main__":
    main()
