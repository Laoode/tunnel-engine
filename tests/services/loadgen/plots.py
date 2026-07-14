"""Render analysis plots from loadgen results (not pytest).

Reads results/results.jsonl (successful requests only) and writes five PNGs
to results/: ISLxOSL 2D histogram, TTFT box plot by mix, time-to-completion
box plot by mix, inter-token latency vs token position, TTFT vs ISL scatter.

  make loadtest-plots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# Categorical slots in fixed order (validated palette; identity never cycles).
SERIES = {"LISO": "#2a78d6", "LILO": "#1baf7a", "SILO": "#eda100", "SISO": "#008300"}
TIER_SERIES = {"none": "#2a78d6", "enterprise": "#1baf7a",
               "pro": "#eda100", "free": "#008300"}
INK = "#3d3d3a"
GRID = {"color": "#d9d9d4", "linewidth": 0.6}


def _load() -> list[dict]:
    """Load successful rows from results.jsonl (exit if none)."""
    path = RESULTS_DIR / "results.jsonl"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run `make loadtest` first.")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    ok = [r for r in rows if r["status"] == 200 and r["ttft_s"] is not None]
    if not ok:
        sys.exit("ERROR: no successful requests in results.jsonl")
    print(f"plots: {len(ok)} successful of {len(rows)} total requests",
          file=sys.stderr)
    return ok


def _style(ax, title, xlabel, ylabel):
    """Recessive grid/axes, titles in ink."""
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def _save(fig, name):
    out = RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}", file=sys.stderr)


def plot_isl_osl(rows):
    """2D histogram of input vs output sequence length (single-hue ramp)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    isl = [r["isl"] for r in rows if r["isl"] and r["osl"]]
    osl = [r["osl"] for r in rows if r["isl"] and r["osl"]]
    h = ax.hist2d(isl, osl, bins=24, cmap="Blues", cmin=1)
    fig.colorbar(h[3], ax=ax, label="requests")
    _style(ax, "Traffic shape: ISL vs OSL", "input tokens", "output tokens")
    _save(fig, "isl_osl_hist2d.png")


def _box_by_mix(rows, field, title, ylabel, fname):
    """Box plot of `field` grouped by mix, boxes tinted per mix slot."""
    mixes = [m for m in SERIES if any(r["mix"] == m for r in rows)]
    data = [[r[field] for r in rows if r["mix"] == m and r[field] is not None]
            for m in mixes]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    boxes = ax.boxplot(data, labels=mixes, patch_artist=True, widths=0.5)
    for patch, m in zip(boxes["boxes"], mixes):
        patch.set_facecolor(SERIES[m])
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK)
    for part in ("medians",):
        for line in boxes[part]:
            line.set_color(INK)
    _style(ax, title, "traffic mix", ylabel)
    _save(fig, fname)


def plot_itl_vs_position(rows):
    """Median and p90 inter-token latency vs output token position."""
    by_pos: dict[int, list[float]] = {}
    for r in rows:
        for pos, itl in enumerate(r["itl_s"]):
            by_pos.setdefault(pos, []).append(itl * 1000)
    positions = sorted(p for p, v in by_pos.items() if len(v) >= 5)
    med = [np.median(by_pos[p]) for p in positions]
    p90 = [np.percentile(by_pos[p], 90) for p in positions]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(positions, med, color=SERIES["LISO"], linewidth=2, label="median")
    ax.plot(positions, p90, color=SERIES["SILO"], linewidth=2, label="p90")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    _style(ax, "Inter-token latency vs token position",
           "output token position", "inter-token latency (ms)")
    _save(fig, "itl_vs_position.png")


def plot_ttft_vs_isl(rows):
    """TTFT vs ISL scatter, colored by tier (prefill efficiency slope)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tiers = [t for t in TIER_SERIES if any(r["tier"] == t for r in rows)]
    for tier in tiers:
        pts = [(r["isl"], r["ttft_s"]) for r in rows
               if r["tier"] == tier and r["isl"]]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=14, color=TIER_SERIES[tier], alpha=0.7,
                       label=tier, edgecolors="white", linewidths=0.4)
    if len(tiers) > 1:
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, title="tier")
    _style(ax, "TTFT vs input length (prefill performance)",
           "input tokens", "TTFT (s)")
    _save(fig, "ttft_vs_isl.png")


def main() -> None:
    rows = _load()
    plot_isl_osl(rows)
    _box_by_mix(rows, "ttft_s", "TTFT by traffic mix", "TTFT (s)", "ttft_box.png")
    _box_by_mix(rows, "total_s", "Time to completion by traffic mix",
                "total time (s)", "completion_box.png")
    plot_itl_vs_position(rows)
    plot_ttft_vs_isl(rows)


if __name__ == "__main__":
    main()
