#!/usr/bin/env python3
"""Render the one-group common-hinge mechanism used in the manuscript example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = np.linspace(0.0, 2.0, 401)
    prefix = np.full_like(theta, 5.0)
    suffix = 4.0 + theta
    envelope = np.maximum(prefix, suffix)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )
    fig, axis = plt.subplots(figsize=(5.4, 2.55), constrained_layout=True)
    axis.axvspan(0.0, 1.0, color="#0072B2", alpha=0.07)
    axis.axvspan(1.0, 2.0, color="#D55E00", alpha=0.07)
    axis.plot(theta, prefix, color="#0072B2", linestyle="--", linewidth=1.5,
              label=r"prefix constant $A=5$")
    axis.plot(theta, suffix, color="#D55E00", linestyle=":", linewidth=1.7,
              label=r"suffix line $Q+\theta=4+\theta$")
    axis.plot(theta, envelope, color="#111111", linewidth=2.3,
              label=r"group value $\max\{A,Q+\theta\}$")
    axis.axvline(1.0, color="#666666", linewidth=0.9)
    axis.scatter([0.0, 1.0, 2.0], [5.0, 5.0, 6.0], color="#111111", s=22, zorder=5)
    axis.annotate("one crossover", xy=(1.0, 5.0), xytext=(1.08, 4.72),
                  arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8})
    axis.text(0.48, 5.88, "prefix controls", ha="center", color="#005B8F")
    axis.text(1.52, 5.88, "suffix controls", ha="center", color="#A74400")
    axis.set_xlim(0.0, 2.0)
    axis.set_ylim(3.9, 6.15)
    axis.set_xticks([0.0, 1.0, 2.0])
    axis.set_xlabel(r"threshold $\theta$ within one local deviation block")
    axis.set_ylabel("group contribution")
    axis.grid(True, color="#E5E5E5", linewidth=0.55)
    axis.legend(loc="lower right", frameon=False)

    fig.savefig(output_dir / "common-hinge-mechanism.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
