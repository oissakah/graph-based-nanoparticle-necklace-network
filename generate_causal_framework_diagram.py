"""Generate the single causal diagram for the nanonecklace framework."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTDIR = Path("framework_diagram")


def box(ax, xy, width, height, title, body, color, title_size=15, body_size=11):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8, edgecolor=color, facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.30, body, ha="center", va="center",
            fontsize=body_size, color="#263238", linespacing=1.35)
    return patch


def arrow(ax, start, end, label=None, color="#455A64", rad=0.0, size=18):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=size,
        linewidth=2.0, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3,
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.027, label, ha="center", va="bottom",
                fontsize=10.5, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.955, "Causal framework for voltage-driven transport in a nanonecklace",
            ha="center", va="center", fontsize=24, fontweight="bold", color="#17212B")
    ax.text(0.5, 0.915,
            r"Applied voltage changes the available graph; Kirchhoff redistribution on that graph produces $I(V)$",
            ha="center", va="center", fontsize=13, color="#455A64")

    y, h = 0.48, 0.21
    boxes = [
        (0.025, 0.13, "Applied bias", r"$V$", "#1565C0"),
        (0.195, 0.18, "Junction activation",
         r"$A_i(V)=\mathbb{1}[V_{a,i}\leq V]$" + "\nedge active if both ends are active", "#7B1FA2"),
        (0.415, 0.19, "Percolating graph",
         r"first source-to-drain cluster at $V_{perc}$" + "\nnew parallel routes appear as $V$ rises", "#00897B"),
        (0.645, 0.19, "Current redistribution",
         r"$\mathbf{L}(V)\,\boldsymbol{\phi}=\mathbf{b}(V)$" + "\n" +
         r"$I_{ij}=g_{ij}(\phi_i-\phi_j)$", "#EF6C00"),
        (0.875, 0.105, "Response", r"$I(V)$", "#C62828"),
    ]
    for x, w, title, body, color in boxes:
        box(ax, (x, y), w, h, title, body, color,
            title_size=12.5 if w < 0.14 else 13.5,
            body_size=12 if w < 0.14 else 9.5)

    arrow(ax, (0.155, 0.585), (0.195, 0.585))
    arrow(ax, (0.375, 0.585), (0.415, 0.585))
    arrow(ax, (0.605, 0.585), (0.645, 0.585))
    arrow(ax, (0.835, 0.585), (0.875, 0.585))

    box(ax, (0.175, 0.755), 0.22, 0.13, r"Microscopic activation scale $V_a$",
        r"sample $V_{a,i}$ from a specified distribution" + "\nphenomenological and not directly measured",
        "#7B1FA2", title_size=11.5, body_size=8.7)
    arrow(ax, (0.285, 0.755), (0.285, 0.695), color="#7B1FA2")

    box(ax, (0.395, 0.245), 0.23, 0.13, "Structural controls",
        r"junction count $N$, connection radius $r_c$," + "\n" +
        r"void fraction $f_v$ and random realization",
        "#00897B", title_size=12.5, body_size=10)
    arrow(ax, (0.51, 0.375), (0.51, 0.48), color="#00897B")

    box(ax, (0.65, 0.245), 0.23, 0.13, "Electrical controls",
        r"$R_{edge,ij}=k_e d_{ij}$; fixed $R_j$" + "\n" +
        r"$R_j$ is independent of $V_a$",
        "#EF6C00", title_size=12.5, body_size=10)
    arrow(ax, (0.765, 0.375), (0.765, 0.48), color="#EF6C00")

    box(ax, (0.73, 0.755), 0.25, 0.13, "Macroscopic fit (post-processing)",
        r"$I=A(V-V_T)^{\zeta}$" + "\n" +
        r"$V_T$: fitted onset; $\zeta$: fitted growth exponent",
        "#C62828", title_size=11.5, body_size=9.5)
    arrow(ax, (0.93, 0.69), (0.86, 0.755), color="#C62828", rad=-0.08)

    ax.text(0.51, 0.145,
            r"$V_{a,i}$ is microscopic and prescribed; $V_{perc}$ is observed from connectivity; "
            r"$V_T$ and $\zeta$ are inferred from the macroscopic curve.",
            ha="center", va="center", fontsize=12.5, color="#263238",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#F5F7FA", edgecolor="#B0BEC5"))
    ax.text(0.51, 0.085,
            r"Therefore $V_{perc}$ and $V_T$ are related transport-onset descriptors, but they are not interchangeable.",
            ha="center", va="center", fontsize=11.5, color="#546E7A")

    fig.savefig(OUTDIR / "nanonecklace_causal_framework.png", dpi=220,
                facecolor="white")
    fig.savefig(OUTDIR / "nanonecklace_causal_framework.svg",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
