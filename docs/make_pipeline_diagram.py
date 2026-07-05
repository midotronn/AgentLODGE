"""Compact, abstract box-and-arrow diagram of the AgentLODGE pipeline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.5, 11.5))
ax.set_xlim(0, 100)
ax.set_ylim(9, 100)
ax.axis("off")

C = {
    "in": "#5B8FF9", "pre": "#61DDAA", "lodge": "#F6BD16", "edge": "#F08BB4",
    "met": "#9270CA", "agent": "#FF9D4D", "sel": "#7DD3C0", "vid": "#65789B",
    "cos": "#E8684A", "out": "#D9D9D9",
}
COST_C = "#C0392B"


def box(x, y, w, h, title, sub="", color="#ddd", tc="black", fs=11, subfs=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
        linewidth=1.4, edgecolor="#333", facecolor=color, alpha=0.93, zorder=2))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color=tc, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=3)
    return (x, y, w, h)


def arr(p1, p2, color="#333", lw=1.9, style="-", rad=0.0, label=""):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, linewidth=lw, color=color,
        linestyle=style, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, label, ha="center",
                va="center", fontsize=7.5, style="italic", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9), zorder=4)


def ortho(pts, color, lw=1.8, style="--", label="", label_pt=None):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
            linestyle=style, lw=lw, zorder=1, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=style, mutation_scale=16), zorder=1)
    if label and label_pt:
        ax.text(*label_pt, label, ha="center", va="center", fontsize=7.5,
                style="italic", color=color, rotation=90,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9), zorder=4)


def bc(b):
    return (b[0] + b[2] / 2, b[1])


def tc_(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


ax.text(50, 97.5, "AgentLODGE Pipeline", ha="center", va="center",
        fontsize=16, fontweight="bold")

b_in = box(38, 89, 24, 5.5, "Song", "audio in", color=C["in"], tc="white")
b_pre = box(24, 76, 52, 8, "Audio Preprocessing",
            "features + AudioDescriptor", color=C["pre"])
b_lodge = box(13, 62, 30, 8, "LODGE", "diffusion \u2192 dance", color=C["lodge"])
b_edge = box(57, 62, 30, 8, "EDGE", "long-form \u2192 dance", color=C["edge"])
b_met = box(23, 49, 54, 7.5, "Coherence Metrics",
            "beat sync, seams, foot stability, trends", color=C["met"])
b_sel = box(29, 37, 42, 7.5, "Selection Agent  (LLM)",
            "picks the more coherent dance", color=C["agent"], tc="white")
b_fmt = box(34, 28.5, 32, 5, "Selected Dance", color=C["sel"], fs=10)
b_vid = box(9, 13, 34, 9, "Stick-figure Video",
            "FK \u2192 3D render + audio", color=C["vid"], tc="white")
b_cos = box(57, 13, 34, 9, "Costume  (LLM \u2192 image)",
            "from audio features", color=C["cos"], tc="white")

arr(bc(b_in), tc_(b_pre))
arr((36, 76), tc_(b_lodge), rad=0.1)
arr((64, 76), tc_(b_edge), rad=-0.1)
arr(bc(b_lodge), (40, 56.5), rad=0.1)
arr(bc(b_edge), (60, 56.5), rad=-0.1)
arr(bc(b_met), tc_(b_sel))
arr(bc(b_sel), tc_(b_fmt))
arr(bc(b_fmt), tc_(b_vid), rad=0.12)
# audio features drive costume (clean right-margin channel)
ortho([(76, 79), (95, 79), (95, 22), (91, 22)], color=COST_C,
      label="audio \u2192 costume", label_pt=(95, 50))

fig.savefig("pipeline_diagram.png", dpi=130, bbox_inches="tight", facecolor="white")
print("saved pipeline_diagram.png")
