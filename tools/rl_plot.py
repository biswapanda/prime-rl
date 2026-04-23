#!/usr/bin/env python3
"""
rl_plot -- Extract metrics from Prime-RL logs and display step-vs-metric charts.

Reads orchestrator and trainer log files, extracts per-step metrics, and renders
them as terminal sparkline/bar charts. Optionally exports to CSV, JSON, or PNG.

Usage:
    # Terminal charts (default paths from smoke test):
    python tools/rl_plot.py

    # Custom log paths:
    python tools/rl_plot.py --orch-log /tmp/smoke_long_orchestrator.log \
                            --trainer-log /tmp/smoke_long_trainer.log

    # Export to CSV:
    python tools/rl_plot.py --export csv --out /tmp/metrics.csv

    # Export to JSON:
    python tools/rl_plot.py --export json --out /tmp/metrics.json

    # Export to PNG (one image per metric + a combined summary):
    python tools/rl_plot.py --export png --out /tmp/dynamo_smoke_long/plots

    # Show specific metrics only:
    python tools/rl_plot.py --metrics loss,reward,entropy
"""
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

RE_ORCH_STEP = re.compile(
    r"Step (\d+) \| Time: ([\d.]+)s \| Reward: (-?[\d.]+) \| Seq\. Length: ([\d.]+) tokens/sample"
    r"(?: \| Async Level: (\d+))?(?: \| Max\. Off-Policy Level: (\d+))?"
)

RE_TRAINER_STEP = re.compile(
    r"Step (\d+) \| Time: ([\d.]+)s \| Loss: (-?[\d.]+)"
    r" \| Entropy: (-?[\d.]+)"
    r" \| Mismatch KL: (-?[\d.]+)"
    r" \| Grad\. Norm: ([\d.]+)"
    r" \| LR: ([\d.e+-]+)"
    r" \| Throughput: ([\d.]+) tokens/s"
    r" \| MFU: ([\d.]+)%"
    r" \| Peak Mem\.: ([\d.]+) GiB"
)


def parse_orch_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for m in RE_ORCH_STEP.finditer(path.read_text(errors="replace")):
        rows.append({
            "step": int(m.group(1)),
            "time_s": float(m.group(2)),
            "reward": float(m.group(3)),
            "seq_len": float(m.group(4)),
            "async_level": int(m.group(5)) if m.group(5) else 0,
            "off_policy": int(m.group(6)) if m.group(6) else 0,
        })
    return rows


def parse_trainer_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for m in RE_TRAINER_STEP.finditer(path.read_text(errors="replace")):
        rows.append({
            "step": int(m.group(1)),
            "time_s": float(m.group(2)),
            "loss": float(m.group(3)),
            "entropy": float(m.group(4)),
            "mismatch_kl": float(m.group(5)),
            "grad_norm": float(m.group(6)),
            "lr": float(m.group(7)),
            "throughput": float(m.group(8)),
            "mfu": float(m.group(9)),
            "peak_mem_gib": float(m.group(10)),
        })
    return rows


# ---------------------------------------------------------------------------
# Terminal chart rendering
# ---------------------------------------------------------------------------

SPARK_CHARS = " " + "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
BAR_FULL = "\u2588"
BAR_PARTIAL = [" ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "\u2588"]


def sparkline(values: list[float], width: int = 0) -> str:
    """Render a sparkline string from values."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    chars = []
    for v in values:
        idx = int((v - mn) / rng * 7)
        idx = max(0, min(7, idx))
        chars.append(SPARK_CHARS[idx + 1])
    return "".join(chars)


def bar_chart(label: str, values: list[float], steps: list[int],
              width: int = 50, color: str = "", reset: str = "") -> list[str]:
    """Render a horizontal bar chart, one line per step."""
    if not values:
        return [f"  {label}: (no data)"]

    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    lines = []
    for step, val in zip(steps, values):
        bar_len = (val - mn) / rng * width if rng > 0 else width / 2
        full = int(bar_len)
        partial_idx = int((bar_len - full) * 8)
        partial_idx = max(0, min(8, partial_idx))
        bar_str = BAR_FULL * full + BAR_PARTIAL[partial_idx]
        lines.append(f"  {step:>3} | {color}{bar_str:<{width+1}}{reset} {val:.6f}")
    return lines


def render_metric_chart(title: str, steps: list[int], values: list[float],
                        fmt: str = ".4f", use_color: bool = True) -> str:
    """Render a single metric as sparkline + summary + detailed bars."""
    if not values:
        return f"\n  {title}: (no data)\n"

    c_title = "\033[1m" if use_color else ""
    c_spark = "\033[36m" if use_color else ""
    c_bar = "\033[32m" if use_color else ""
    c_dim = "\033[2m" if use_color else ""
    c_reset = "\033[0m" if use_color else ""

    mn, mx, avg = min(values), max(values), sum(values) / len(values)
    spark = sparkline(values)

    lines = [
        f"",
        f"  {c_title}{title}{c_reset}  "
        f"min={mn:{fmt}}  avg={avg:{fmt}}  max={mx:{fmt}}  "
        f"last={values[-1]:{fmt}}  n={len(values)}",
        f"  {c_spark}{spark}{c_reset}",
        f"",
    ]

    # Only show detailed bars if <=30 steps (otherwise just sparkline)
    if len(values) <= 30:
        lines += bar_chart(title, values, steps, width=40, color=c_bar, reset=c_reset)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(orch_rows: list[dict], trainer_rows: list[dict], out: Path):
    """Export metrics to CSV with one row per step, merging orch + trainer on step number."""
    merged = {}
    for r in orch_rows:
        merged.setdefault(r["step"], {}).update({f"orch_{k}": v for k, v in r.items()})
    for r in trainer_rows:
        merged.setdefault(r["step"], {}).update({f"train_{k}": v for k, v in r.items()})

    if not merged:
        print("No data to export.")
        return

    all_keys = set()
    for row in merged.values():
        all_keys.update(row.keys())
    fieldnames = ["step"] + sorted(all_keys - {"step", "orch_step", "train_step"})

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(merged.keys()):
            row = {"step": step}
            row.update(merged[step])
            row.pop("orch_step", None)
            row.pop("train_step", None)
            writer.writerow(row)

    print(f"Exported {len(merged)} steps to {out}")


def export_json(orch_rows: list[dict], trainer_rows: list[dict], out: Path):
    """Export metrics to JSON."""
    data = {
        "orchestrator": orch_rows,
        "trainer": trainer_rows,
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported to {out}")


# ---------------------------------------------------------------------------
# PNG export (matplotlib)
# ---------------------------------------------------------------------------

# Metric display config: (key_in_row, title, y_label, source, fmt)
METRIC_DEFS = [
    ("reward",      "Reward per Step",         "Reward",           "orch",    ".4f"),
    ("seq_len",     "Sequence Length",          "Tokens / Sample",  "orch",    ".0f"),
    ("loss",        "Training Loss",           "Loss",             "trainer", ".6f"),
    ("entropy",     "Entropy",                 "Entropy",          "trainer", ".4f"),
    ("mismatch_kl", "Mismatch KL Divergence",  "KL",              "trainer", ".4f"),
    ("grad_norm",   "Gradient Norm",           "Norm",             "trainer", ".4f"),
    ("throughput",  "Throughput",              "Tokens / s",       "trainer", ".0f"),
]

# Consistent color palette
METRIC_COLORS = {
    "reward": "#2196F3",
    "seq_len": "#9C27B0",
    "loss": "#F44336",
    "entropy": "#FF9800",
    "mismatch_kl": "#4CAF50",
    "grad_norm": "#795548",
    "throughput": "#00BCD4",
}


def _plot_single_metric(steps: list[int], values: list[float],
                        title: str, ylabel: str, color: str,
                        out_path: Path):
    """Plot one metric and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, values, marker="o", markersize=4, linewidth=1.8, color=color)
    ax.fill_between(steps, values, alpha=0.15, color=color)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Annotate min / max / last
    mn_val, mx_val, avg_val = min(values), max(values), sum(values) / len(values)
    mn_idx = values.index(mn_val)
    mx_idx = values.index(mx_val)
    stats_text = f"min={mn_val:.4g}  avg={avg_val:.4g}  max={mx_val:.4g}  last={values[-1]:.4g}"
    ax.text(0.01, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(orch_rows: list[dict], trainer_rows: list[dict],
                  selected: set[str], out_path: Path):
    """Plot a combined summary grid of all selected metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for key, title, ylabel, source, fmt in METRIC_DEFS:
        if key not in selected:
            continue
        rows = orch_rows if source == "orch" else trainer_rows
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        values = [r[key] for r in rows]
        panels.append((key, title, ylabel, steps, values))

    if not panels:
        print("No metrics to plot.")
        return

    n = len(panels)
    cols = min(n, 2)
    rows_grid = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(7 * cols, 4 * rows_grid), squeeze=False)

    for idx, (key, title, ylabel, steps, values) in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        color = METRIC_COLORS.get(key, "#333333")
        ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, color=color)
        ax.fill_between(steps, values, alpha=0.12, color=color)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        mn_val, mx_val = min(values), max(values)
        avg_val = sum(values) / len(values)
        ax.text(0.01, 0.97, f"min={mn_val:.4g}  avg={avg_val:.4g}  max={mx_val:.4g}",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Hide unused axes
    for idx in range(n, rows_grid * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Prime-RL + Dynamo -- Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_png(orch_rows: list[dict], trainer_rows: list[dict],
               out_dir: Path, selected: set[str]):
    """Export metrics as individual PNGs + a combined summary PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for key, title, ylabel, source, fmt in METRIC_DEFS:
        if key not in selected:
            continue
        rows = orch_rows if source == "orch" else trainer_rows
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        values = [r[key] for r in rows]
        color = METRIC_COLORS.get(key, "#333333")
        out_path = out_dir / f"{key}.png"
        _plot_single_metric(steps, values, title, ylabel, color, out_path)
        saved.append(out_path)
        print(f"  Saved {out_path}")

    # Combined summary
    summary_path = out_dir / "summary.png"
    _plot_summary(orch_rows, trainer_rows, selected, summary_path)
    saved.append(summary_path)
    print(f"  Saved {summary_path}")

    print(f"\nExported {len(saved)} PNGs to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_METRICS = ["reward", "loss", "entropy", "mismatch_kl", "grad_norm", "throughput", "seq_len", "time_s"]


def main():
    parser = argparse.ArgumentParser(description="Plot RL training metrics from log files")
    parser.add_argument("--orch-log", type=Path, default=Path("/tmp/smoke_long_orchestrator.log"),
                        help="Orchestrator log file")
    parser.add_argument("--trainer-log", type=Path, default=Path("/tmp/smoke_long_trainer.log"),
                        help="Trainer log file")
    parser.add_argument("--metrics", type=str, default=None,
                        help=f"Comma-separated metrics to show (default: all). Options: {','.join(ALL_METRICS)}")
    parser.add_argument("--export", choices=["csv", "json", "png"], default=None,
                        help="Export format instead of terminal display")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file/directory for export (for png: directory for images)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors")
    args = parser.parse_args()

    orch_rows = parse_orch_log(args.orch_log)
    trainer_rows = parse_trainer_log(args.trainer_log)

    if not orch_rows and not trainer_rows:
        print(f"No data found. Check log paths:")
        print(f"  --orch-log    {args.orch_log}  {'(exists)' if args.orch_log.exists() else '(NOT FOUND)'}")
        print(f"  --trainer-log {args.trainer_log}  {'(exists)' if args.trainer_log.exists() else '(NOT FOUND)'}")
        sys.exit(1)

    # Export mode
    selected = set(args.metrics.split(",")) if args.metrics else set(ALL_METRICS)
    if args.export:
        if args.export == "png":
            out_dir = args.out or Path("/tmp/rl_plots")
            export_png(orch_rows, trainer_rows, out_dir, selected)
        elif args.export == "csv":
            out = args.out or Path("/tmp/rl_metrics.csv")
            export_csv(orch_rows, trainer_rows, out)
        else:
            out = args.out or Path("/tmp/rl_metrics.json")
            export_json(orch_rows, trainer_rows, out)
        return

    # Terminal display mode
    use_color = not args.no_color and sys.stdout.isatty()
    c_header = "\033[1;34m" if use_color else ""
    c_reset = "\033[0m" if use_color else ""

    print(f"\n{c_header}{'=' * 70}{c_reset}")
    print(f"{c_header}  Prime-RL + Dynamo -- Training Metrics{c_reset}")
    print(f"{c_header}{'=' * 70}{c_reset}")

    if orch_rows:
        orch_steps = [r["step"] for r in orch_rows]
        print(f"\n  Orchestrator: {len(orch_rows)} steps "
              f"({args.orch_log.name})")

        if "reward" in selected:
            print(render_metric_chart("Reward (per step)",
                                      orch_steps, [r["reward"] for r in orch_rows]))
        if "seq_len" in selected:
            print(render_metric_chart("Seq Length (tokens/sample)",
                                      orch_steps, [r["seq_len"] for r in orch_rows],
                                      fmt=".1f", use_color=use_color))

    if trainer_rows:
        train_steps = [r["step"] for r in trainer_rows]
        print(f"\n  Trainer: {len(trainer_rows)} steps "
              f"({args.trainer_log.name})")

        if "loss" in selected:
            print(render_metric_chart("Loss",
                                      train_steps, [r["loss"] for r in trainer_rows],
                                      fmt=".6f", use_color=use_color))
        if "entropy" in selected:
            print(render_metric_chart("Entropy",
                                      train_steps, [r["entropy"] for r in trainer_rows],
                                      use_color=use_color))
        if "mismatch_kl" in selected:
            print(render_metric_chart("Mismatch KL",
                                      train_steps, [r["mismatch_kl"] for r in trainer_rows],
                                      use_color=use_color))
        if "grad_norm" in selected:
            print(render_metric_chart("Gradient Norm",
                                      train_steps, [r["grad_norm"] for r in trainer_rows],
                                      use_color=use_color))
        if "throughput" in selected:
            print(render_metric_chart("Throughput (tokens/s)",
                                      train_steps, [r["throughput"] for r in trainer_rows],
                                      fmt=".0f", use_color=use_color))

    print(f"\n{c_header}{'=' * 70}{c_reset}")
    print(f"  Export:  python tools/rl_plot.py --export csv --out /tmp/metrics.csv")
    print(f"  Plots:   python tools/rl_plot.py --export png --out /tmp/plots")
    print(f"  Filter:  python tools/rl_plot.py --metrics loss,reward")
    print(f"{c_header}{'=' * 70}{c_reset}\n")


if __name__ == "__main__":
    main()
