import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

# ── Load & normalize ──────────────────────────────────────────────────────────

with open("data.json") as f:
    data = json.load(f)

for row in data:
    if row["Metric"] == "Accuracy":
        for col in ("FFT_Paper", "FFT_Ours", "LoRA_Paper", "LoRA_Ours"):
            if row[col] is not None and row[col] <= 1.0:
                row[col] = row[col] * 100


# ── Chart 1: Grouped Bar Chart (ViT, MNLI, SST-2) ────────────────────────────

acc_rows = [r for r in data if r["Metric"] == "Accuracy"]

group_labels = ["ViT / CIFAR-10", "RoBERTa / MNLI", "RoBERTa / SST-2"]

COLS = ("FFT_Paper", "FFT_Ours", "LoRA_Paper", "LoRA_Ours")
DISPLAY = ("FFT – Paper Baseline", "FFT – Our Results",
           "LoRA – Paper Baseline", "LoRA – Our Results")
COLORS = ("#AED6F1", "#1A5276", "#FAD7A0", "#E67E22")

bar_w = 0.17
gap = 0.08
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w + np.array([-gap / 2, -gap / 2, gap / 2, gap / 2])

x = np.arange(len(acc_rows))

fig, ax = plt.subplots(figsize=(13, 7))

for b, (col, label, color) in enumerate(zip(COLS, DISPLAY, COLORS)):
    for i, row in enumerate(acc_rows):
        v = row[col]
        if v is not None:
            bar = ax.bar(
                x[i] + offsets[b], v, width=bar_w,
                color=color, edgecolor="white", linewidth=0.6,
                label=label if i == 0 else "_nolegend_",
            )
            ax.text(
                x[i] + offsets[b], v + 0.15,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

# Dashed reference line at LoRA Paper per group
for i, row in enumerate(acc_rows):
    lo = row["LoRA_Paper"]
    if lo is not None:
        span = 2 * bar_w + gap + 0.02
        ax.hlines(
            lo, x[i] - span, x[i] + span,
            colors="#FAD7A0", linestyles="dashed", linewidth=1.8, alpha=0.8,
            label="LoRA Paper Baseline (ref.)" if i == 0 else "_nolegend_",
            zorder=5,
        )

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.text(
    0.5, 0.97,
    "Model Performance: NLP & ViT Tasks\n(FFT vs. LoRA — Paper Baselines vs. Our Results)",
    fontsize=14, fontweight="bold",
    ha="center", va="top",
    transform=ax.transAxes,
    bbox=dict(facecolor="white", edgecolor="none", pad=4),
    zorder=10,
)
ax.set_ylim(80, 102)
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

legend_handles = [
    mpatches.Patch(color=COLORS[0], label="FFT – Paper Baseline"),
    mpatches.Patch(color=COLORS[1], label="FFT – Our Results"),
    mpatches.Patch(color=COLORS[2], label="LoRA – Paper Baseline"),
    mpatches.Patch(color=COLORS[3], label="LoRA – Our Results"),
    plt.Line2D([0], [0], color="#FAD7A0", linestyle="dashed",
               linewidth=1.8, label="LoRA Paper Baseline (reference line)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9.5, framealpha=0.95)

sns.despine(left=False, bottom=False)
plt.tight_layout()
plt.savefig("chart1_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart1_grouped_bar.png")


# ── Chart 2: Radar Chart — GPT-2 Medium / E2E ────────────────────────────────

gpt2_rows = [r for r in data if r["Model"] == "GPT-2 Medium / E2E"]
metrics = [r["Metric"] for r in gpt2_rows]
N = len(metrics)

conditions = ("FFT_Paper", "FFT_Ours", "LoRA_Paper", "LoRA_Ours")
cond_labels = (
    "FFT – Paper Baseline",
    "FFT – Our Results",
    "LoRA – Paper Baseline",
    "LoRA – Our Results",
)
cond_colors = ("#AED6F1", "#1A5276", "#FAD7A0", "#E67E22")
cond_ls = ("--", "-", "--", "-")
cond_lw = (2.0, 2.5, 2.0, 2.5)

# Raw values matrix: conditions × metrics
raw = {c: [r[c] for r in gpt2_rows] for c in conditions}

# Normalize each metric to its maximum across all four conditions
max_per_metric = []
for j in range(N):
    vals = [raw[c][j] for c in conditions if raw[c][j] is not None]
    max_per_metric.append(max(vals) if vals else 1.0)

norm = {
    c: [(raw[c][j] / max_per_metric[j] if raw[c][j] is not None else 0.0)
        for j in range(N)]
    for c in conditions
}

angles = [n / N * 2 * np.pi for n in range(N)] + [0]  # closed

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for cond, label, color, ls, lw in zip(conditions, cond_labels, cond_colors, cond_ls, cond_lw):
    vals = norm[cond] + [norm[cond][0]]
    ax.plot(angles, vals, color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
    ax.fill(angles, vals, color=color, alpha=0.08)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")

ax.set_ylim(0, 1.12)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], fontsize=8, color="grey")
ax.yaxis.set_tick_params(pad=6)

ax.set_title(
    "GPT-2 Medium / E2E — Generative Metric Comparison\n"
    "(Each axis normalized to its max value across all conditions)",
    fontsize=13, fontweight="bold", pad=22,
)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.08),
    ncols=2, fontsize=10, framealpha=0.95,
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("chart2_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart2_radar.png")
