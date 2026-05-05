import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
})

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = '#F5F7FA'
HDR_BG   = '#1E3A5F'
ROBERTA  = '#B03A2E'
GPT2C    = '#1A7A5E'
VITC     = '#6C3483'
ROW_BG   = '#FFFFFF'
ROW_BDR  = '#D1D9E6'
BADGE_BG = '#EBF0F8'
BADGE_FG = '#1E3A5F'
VS_C     = '#C0622A'

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 7)
ax.set_ylim(0, 5.2)
ax.axis('off')

# ── Layout ────────────────────────────────────────────────────────────────────
# 4 equal-width columns: [row label | RoBERTa | GPT-2 | ViT]
LX  = 0.2
CW  = 1.5
GAP = 0.15
COL_X   = [LX + i * (CW + GAP) for i in range(4)]   # [0.2, 1.85, 3.5, 5.15]
TOTAL_W = COL_X[-1] + CW - LX                        # 6.45

# 4 rows: [header | Hardware | Experiment Type | Benchmark Dataset]
# Row definitions: (label_text, bottom_y, height)
HDR_Y  = 4.10;  HDR_H = 0.85
ROW_DEFS = [
    ('Hardware',          3.20, 0.80),
    ('Experiment\nType',  1.40, 1.70),   # taller: stacked badges
    ('Benchmark\nDataset',0.20, 1.10),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def rrect(x, y, w, h, fc, ec='none', lw=0, z=2, pad=0.08):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def label(x, y, s, size=13, color='white', bold=True, z=5, bg=None, bgec='none', bglw=0):
    kw = dict(ha='center', va='center', fontsize=size, zorder=z,
              fontweight='bold' if bold else 'normal', color=color)
    if bg:
        kw['bbox'] = dict(boxstyle='round,pad=0.45', facecolor=bg,
                          edgecolor=bgec, linewidth=bglw)
    ax.text(x, y, s, **kw)

# ── Header row: model name columns ───────────────────────────────────────────
rrect(LX-0.1, HDR_Y, TOTAL_W+0.2, HDR_H, fc=ROW_BG, ec=ROW_BDR, lw=1.5, z=1, pad=0.07)

for i, (name, color) in enumerate(
        zip(['', 'RoBERTa', 'GPT-2', 'ViT'], [None, ROBERTA, GPT2C, VITC])):
    cx = COL_X[i] + CW / 2
    cy = HDR_Y + HDR_H / 2
    if color:
        rrect(COL_X[i]+0.12, HDR_Y+0.10, CW-0.24, HDR_H-0.20, fc=color, z=3, pad=0.1)
        label(cx, cy, name, size=16, color='white', z=4)

# ── Data rows ────────────────────────────────────────────────────────────────
ROW_DATA = [
    # Hardware
    [None, 'A100', 'T4', 'T4'],
    # Experiment Type: (p1, p2) tuples for vs comparisons
    [None, ('Full FT', 'LoRA'), ('Small', 'Medium'), ('Generalization\nTest', None)],
    # Benchmark Dataset
    [None, 'SST-2 / MNLI', 'E2E NLG', 'CIFAR-10'],
]

for (row_name, row_y, row_h), row_vals in zip(ROW_DEFS, ROW_DATA):
    cy = row_y + row_h / 2

    # Row background
    rrect(LX-0.1, row_y, TOTAL_W+0.2, row_h, fc=ROW_BG, ec=ROW_BDR, lw=1.5, z=1, pad=0.07)

    # Row label (col 0)
    lbx = COL_X[0] - 0.05
    lbw = CW - 0.1
    ax.text(lbx + lbw/2, cy, row_name, ha='center', va='center',
            fontsize=13, fontweight='bold', color=HDR_BG, zorder=5)

    # Data cells (cols 1–3)
    for col_x, val in zip(COL_X[1:], row_vals[1:]):
        cx = col_x + CW / 2

        if isinstance(val, tuple):
            p1, p2 = val
            if p2:
                # Stacked: badge / vs / badge
                label(cx, cy + 0.48, p1, size=13, color=BADGE_FG, bg=BADGE_BG)
                label(cx, cy,        'vs', size=13, color=VS_C)
                label(cx, cy - 0.48, p2, size=13, color=BADGE_FG, bg=BADGE_BG)
            else:
                label(cx, cy, p1, size=13, color=BADGE_FG, bg=BADGE_BG)
        else:
            label(cx, cy, val, size=14, color=BADGE_FG, bg=BADGE_BG)

    # Vertical dividers: after col 0 (label) and between data columns
    for i in range(0, 3):
        dx = COL_X[i+1] - GAP/2
        ax.plot([dx, dx], [row_y+0.05, row_y+row_h-0.05],
                color=ROW_BDR, lw=1.8, zorder=3, solid_capstyle='butt')

plt.tight_layout(pad=0)
plt.savefig('experiment_table.png', dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print('Saved → experiment_table.png')
