"""Тайлангийн зургууд (results/cv-ээс). Ажиллуулах: python experiments/figures.py <out_dir>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CV = ROOT / "results" / "cv"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODELS = ["LR", "RF", "ET", "XGB"]
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def effort_curve():
    oof = pd.read_csv(CV / "oof.csv.gz")
    d = oof[(oof.model == "RF") & (oof.repeat == 0)]
    y, e, p = d.y.to_numpy(), d.effort.to_numpy(), d.prob.to_numpy()
    e1 = np.maximum(e, 1)

    def curve(score):
        o = np.argsort(-score, kind="mergesort")
        x = np.concatenate([[0], np.cumsum(e[o]) / e.sum()])
        f = np.concatenate([[0], np.cumsum(y[o]) / y.sum()])
        return x, f

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for b in (0.1, 0.2, 0.5):
        ax.axvline(b, color=MUTED, lw=0.8, ls=":")
        ax.text(b + 0.008, 0.03, f"PofB{int(b*100)}", color=MUTED, fontsize=8)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--", label="Санамсаргүй ranking")
    x, f = curve(y / e1 + 1e-9 * y)
    ax.plot(x, f, color=INK, lw=1.2, label="Оновчтой (бодит density)")
    x, f = curve(p)
    ax.plot(x, f, color=BLUE, lw=2, label="RF, probability ranking")
    x, f = curve(p / e1)
    ax.plot(x, f, color=ORANGE, lw=2, label="RF, density ranking")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Шалгасан effort-ийн хувь (хуримтлагдсан CSM / ΣCSM)")
    ax.set_ylabel("Олдсон defective модулийн хувь")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(OUT / "effort_curve.pdf"); plt.close(fig)


def ranking_dumbbell():
    s = pd.read_csv(CV / "summary.csv", header=[0, 1], index_col=[0, 1]).xs("mean", axis=1, level=1)
    metrics = ["ROC-AUC", "PR-AUC", "PofB20", "IFA"]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.6), sharey=True)
    ypos = np.arange(len(MODELS))[::-1]
    for ax, m in zip(axes, metrics):
        pv = [s.loc[(mm, "probability"), m] for mm in MODELS]
        dv = [s.loc[(mm, "density"), m] for mm in MODELS]
        for yy, a, b in zip(ypos, pv, dv):
            ax.plot([a, b], [yy, yy], color=GRID, lw=2.5, zorder=1)
        ax.scatter(pv, ypos, s=42, color=BLUE, marker="o", zorder=2,
                   edgecolor="white", linewidth=1, label="probability")
        ax.scatter(dv, ypos, s=46, color=ORANGE, marker="s", zorder=2,
                   edgecolor="white", linewidth=1, label="density")
        ax.set_title(m + (" (бага нь сайн)" if m == "IFA" else ""), fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
        lo, hi = min(pv + dv), max(pv + dv)
        pad = (hi - lo) * 0.15
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(ypos, MODELS)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, -0.18), ncol=2, fontsize=9)
    fig.savefig(OUT / "ranking_dumbbell.pdf"); plt.close(fig)


def bootstrap_forest():
    d = pd.read_csv(CV / "bootstrap_diff.csv")
    d = d[(d.comparison == "RF − XGB") & (d.metric != "IFA")]
    order = ["Accuracy", "Precision", "Recall", "F1", "MCC", "G-mean",
             "ROC-AUC", "PR-AUC", "PofB10", "PofB20", "PofB50"]
    d = d.set_index("metric").loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    yy = np.arange(len(order))[::-1]
    for y_, (_, r) in zip(yy, d.iterrows()):
        c = BLUE if r.excludes_zero else MUTED
        ax.plot([r.ci_low, r.ci_high], [y_, y_], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r.diff_mean], [y_], color=c, s=36, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(yy, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RF − XGB ялгаа (дундаж ба 95% bootstrap CI)")
    fig.savefig(OUT / "bootstrap_rf_xgb.pdf"); plt.close(fig)


def size_box():
    df = pd.read_csv(ROOT / "data" / "advisor" / "ant-1.7_nodes.csv")
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    data = [df.loc[df.label == 0, "CSM"], df.loc[df.label == 1, "CSM"]]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                    markeredgecolor="none", alpha=0.5),
                    medianprops=dict(color=INK, lw=1.5))
    for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    ax.set_yticks([1, 2], ["Clean (579)", "Defective (165)"])
    ax.set_xlabel("CSM (method + attribute-ын тоо)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "csm_box.pdf"); plt.close(fig)


if __name__ == "__main__":
    effort_curve(); ranking_dumbbell(); bootstrap_forest(); size_box()
    print("saved to", OUT)
