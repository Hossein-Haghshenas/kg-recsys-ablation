#!/usr/bin/env python3
"""Rebuild fig2_pipeline.pdf with clean math text and no variable interpolation issues."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BOX_FC, BOX_EC = "#dce6f1", "#4a6a8a"
INK, GREY, ANN = "#333340", "#8a8ab0", "#4a6a8a"
W, H = 10.6, 5.9
FS, FSA = 9.2, 8.2

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
PITCH = (FS * 1.32) / (H * 72 / 100)

def box(x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.7,rounding_size=1.8",
        fc=BOX_FC, ec=BOX_EC, lw=0.9, zorder=3))
    n = len(lines)
    top = y + h/2 + (n-1)*PITCH/2
    for i, txt in enumerate(lines):
        ax.text(x + w/2, top - i*PITCH, txt, ha="center", va="center",
                fontsize=FS, color=INK, zorder=4)
    return (x, y, w, h)

def arrow(a, b, side="r", rad=0.0):
    ax_, ay, aw, ah = a; bx, by, bw, bh = b
    if side == "r":   p0, p1 = (ax_+aw, ay+ah/2), (bx, by+bh/2)
    elif side == "d": p0, p1 = (ax_+aw/2, ay), (bx+bw/2, by+bh)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
        lw=0.9, color=GREY, zorder=2,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))

def ann(x, y, lines, ha="center"):
    for i, t in enumerate(lines):
        ax.text(x, y - i*(FSA*1.32)/(H*72/100), t, ha=ha, va="center",
                fontsize=FSA, color=ANN, style="italic", zorder=5)

ui   = box( 2, 63, 18, 17, ["User–item", "interaction graph", r"$\mathcal{G}_{ui}$"])
cenc = box(27, 63, 20, 17, ["Collaborative", r"encoder $f_{ui}$", "(GCN / LightGCN)"])
kg   = box( 2, 20, 18, 13, ["Knowledge graph", r"$\mathcal{G}_{kg}$"])
kenc = box(27, 20, 20, 17, ["Knowledge", r"encoder $f_{kg}$", "(KGE / attention)"])
fus  = box(55, 42, 18, 17, ["Fusion", r"$\phi(\cdot)$", "sum / concat / gate"])
sc   = box(80, 66, 16, 13, ["Scoring", r"$\hat{y}_{ui}$"])
obj  = box(80, 43, 16, 13, ["Objective", r"$\mathcal{L}$"])
upd  = box(80, 20, 16, 13, ["Update", r"$\Theta$"])

arrow(ui, cenc); arrow(kg, kenc)
arrow(cenc, fus, rad=-0.14); arrow(kenc, fus, rad=0.14)
arrow(fus, sc, rad=-0.14); arrow(sc, obj, side="d"); arrow(obj, upd, side="d")

YB = 9.0
ax.plot([88, 88, 37, 37], [20, YB, YB, 20], color=GREY, lw=0.9,
        zorder=1, solid_joinstyle="round", linestyle=(0, (4, 2.2)))
ax.add_patch(FancyArrowPatch((37, YB + 3.0), (37, 20), arrowstyle="-|>",
    mutation_scale=11, lw=0.9, color=GREY, zorder=1, shrinkA=0, shrinkB=2,
    linestyle=(0, (4, 2.2))))
ax.text(62.5, YB - 4.0, "parameter update", ha="center", fontsize=FSA,
        color=GREY, style="italic", zorder=5)

ann(37, 95, [r"Paradigm II: replaces $f_{kg}$", "by explicit paths"])
ann(75, 95, [r"Paradigm V: rewrites $\mathcal{G}_{kg}$ /", r"initializes $f_{kg}$ with an LM"])
ann(64, 33.5, [r"Paradigm III: makes $\phi$", "layer-wise & attentive"])
ann(11, 46, [r"Paradigm I: constrains $f_{kg}$", r"and adds $\mathcal{L}_{kg}$"])
ann(37, 51, [r"Paradigm IV: adds contrastive", r"$\mathcal{L}_{ssl}$ over views of $\mathcal{G}$"])

out_paths = [
    Path("d:/projects/RS-Survey/40_paper/fig2_pipeline.pdf"),
    Path("D:/projects/current-git-project/kg-recsys-ablation/30_analysis/figures/fig2_pipeline.pdf"),
    Path("D:/claude-projects/RS-Survey-Article/manuscript/fig2_pipeline.pdf")
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.06)
    print(f"Saved: {p}")

plt.close(fig)
