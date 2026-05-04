import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
})

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = '#F5F7FA'
HDR_BG    = '#1E3A5F'
HDR_FG    = '#FFFFFF'
HDR_ACNT  = '#E8A838'     # warm gold for "Model" label
ROBERTA   = '#B03A2E'     # deep crimson
GPT2C     = '#1A7A5E'     # forest teal
VITC      = '#6C3483'     # deep purple
ROW_BG    = '#FFFFFF'
ROW_BDR   = '#D1D9E6'
BADGE_BG  = '#EBF0F8'
BADGE_FG  = '#1E3A5F'
VS_C      = '#C0622A'     # dark burnt orange

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 7.5)
ax.axis('off')

# ── Layout ────────────────────────────────────────────────────────────────────
LX      = 0.4
COL_X   = [LX, 3.2, 7.0, 12.0]
COL_W   = [2.5, 3.4, 4.6, 4.8]
HDR_Y   = 6.2;  HDR_H = 1.0
ROW_H   = 1.8
ROW_Y   = [4.28, 2.36, 0.44]
TOTAL_W = COL_X[-1] + COL_W[-1] - LX   # 16.4

# ── Helpers ───────────────────────────────────────────────────────────────────
def rrect(x, y, w, h, fc, ec='none', lw=0, z=2, pad=0.08):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def label(x, y, s, size=13, color='#1E3A5F', bold=True, z=5,
          bg=None, bgec='none', bglw=0):
    kw = dict(ha='center', va='center', fontsize=size, zorder=z,
              fontweight='bold' if bold else 'normal', color=color)
    if bg:
        kw['bbox'] = dict(boxstyle='round,pad=0.5',
                          facecolor=bg, edgecolor=bgec, linewidth=bglw)
    ax.text(x, y, s, **kw)

# ── Header ────────────────────────────────────────────────────────────────────
rrect(LX-0.1, HDR_Y, TOTAL_W+0.2, HDR_H, fc=HDR_BG, z=1, pad=0.1)

HEADERS = ['Model', 'Hardware', 'Experiment Type', 'Benchmark Dataset']
for cx_base, cw, head in zip(COL_X, COL_W, HEADERS):
    label(cx_base + cw/2, HDR_Y + HDR_H/2, head,
          size=19, color=HDR_ACNT if head == 'Model' else HDR_FG, z=4)

for i in range(1, 4):
    dx = COL_X[i] - 0.18
    ax.plot([dx, dx], [HDR_Y+0.2, HDR_Y+HDR_H-0.2], color='#4A6A8A', lw=1.5, zorder=4)

# ── Data rows ─────────────────────────────────────────────────────────────────
ROWS = [
    dict(name='RoBERTa', color=ROBERTA, gpu='A100', p1='Full FT',  p2='LoRA',   goal='SST-2 / MNLI'),
    dict(name='GPT-2',   color=GPT2C,   gpu='T4',   p1='Small',    p2='Medium', goal='E2E NLG'),
    dict(name='ViT',     color=VITC,    gpu='T4',   p1='Generalization Test', p2=None, goal='CIFAR-10'),
]

for row, ry in zip(ROWS, ROW_Y):
    c  = row['color']
    cy = ry + ROW_H / 2

    # Row: white background, light border
    rrect(LX-0.1, ry, TOTAL_W+0.2, ROW_H, fc=ROW_BG, ec=ROW_BDR, lw=1.5, z=1, pad=0.07)
    # Col 0: colored model name box
    box_x = LX - 0.04
    box_w = COL_X[0] + COL_W[0] - box_x - 0.2
    rrect(box_x, ry + 0.1, box_w, ROW_H - 0.2, fc=c, z=2, pad=0.12)
    label(box_x + box_w/2, cy, row['name'], size=24, color='white', z=5)

    # Col 1: Hardware
    cx1 = COL_X[1] + COL_W[1] / 2
    label(cx1, cy, row['gpu'], size=21, color=BADGE_FG, bg=BADGE_BG)

    # Col 2: Experiment Type
    cx2 = COL_X[2] + COL_W[2] / 2
    if row['p2']:
        label(cx2 - 1.2, cy, row['p1'], size=19, color=BADGE_FG, bg=BADGE_BG)
        label(cx2,        cy, 'vs',      size=20, color=VS_C)
        label(cx2 + 1.2, cy, row['p2'], size=19, color=BADGE_FG, bg=BADGE_BG)
    else:
        label(cx2, cy, row['p1'], size=19, color=BADGE_FG, bg=BADGE_BG)

    # Col 3: Benchmark Dataset
    cx3 = COL_X[3] + COL_W[3] / 2
    label(cx3, cy, row['goal'], size=21, color=BADGE_FG, bg=BADGE_BG)

plt.tight_layout(pad=0)
plt.savefig('experiment_table.png', dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print('Saved → experiment_table.png')
