"""
Figure 1: schematic of the model. Collective layer (group phenotypic composition, GPC:
diversity D, coherence C, spatial structure S, collective function F), the two therapy
channels T_C and T_S, and the clonal layer (sensitive N_s and resistant N_r cells) with the
two couplings phi and omega. Drawing only, no simulation.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(11.5, 7.2))
ax.set_xlim(0, 11.2); ax.set_ylim(-1.35, 6.25); ax.axis('off')
BLUE, GREEN, ORANGE, RED, GREY, PURPLE = '#2f6db5', '#2e8b57', '#d9822b', '#c0392b', '#707070', '#7d3c98'

def box(x, y, title, sub, col, w=1.9, h=0.8, fs=13):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.04,rounding_size=0.14',
                                fc='white', ec=col, lw=2.2, zorder=3))
    ax.text(x, y + 0.14, title, ha='center', va='center', fontsize=fs, weight='bold', color=col, zorder=4)
    ax.text(x, y - 0.2, sub, ha='center', va='center', fontsize=8.5, color='#333333', zorder=4)

def arr(p, q, col, style='-|>', cs='arc3,rad=0', lw=1.9):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=cs, arrowstyle=style, mutation_scale=17,
                                 color=col, lw=lw, shrinkA=2, shrinkB=2, zorder=3.5))

def lab(x, y, t, col, fs=9.5, **k):
    ax.text(x, y, t, fontsize=fs, color=col, ha='center', va='center', zorder=5,
            bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.9), **k)

# panels
ax.add_patch(FancyBboxPatch((0.2, 0.45), 6.6, 4.15, boxstyle='round,pad=0.02,rounding_size=0.2', fc='#f3f6fb', ec='#c9d6e8', lw=1))
ax.add_patch(FancyBboxPatch((7.55, 0.45), 3.4, 4.15, boxstyle='round,pad=0.02,rounding_size=0.2', fc='#f6f2f9', ec='#dccfe8', lw=1))
ax.text(2.0, 4.85, 'Collective layer: group phenotypic composition (GPC)', fontsize=11, weight='bold', ha='left')
ax.text(9.25, 4.85, 'Clonal layer', fontsize=11, weight='bold', ha='center')

# collective variables
box(1.35, 3.1, 'S', 'spatial structure', BLUE)
box(5.65, 3.1, 'D', 'phenotypic diversity', BLUE)
box(3.5, 1.2, 'C', 'functional coherence', BLUE)
ax.add_patch(Circle((3.5, 3.1), 0.55, fc='#fff4e6', ec=ORANGE, lw=2.4, zorder=3))
ax.text(3.5, 3.2, 'F', fontsize=15, weight='bold', color=ORANGE, ha='center', va='center', zorder=4)
ax.text(3.5, 2.88, 'C·S·g(D)', fontsize=8, color=ORANGE, ha='center', va='center', zorder=4)
lab(2.55, 3.95, 'collective function', ORANGE, fs=8.5)

# inputs to F
arr((2.3, 3.1), (2.93, 3.1), GREY); arr((4.7, 3.1), (4.07, 3.1), GREY); arr((3.5, 1.6), (3.5, 2.53), GREY)
lab(4.4, 3.33, 'g(D)', GREY, fs=8)
# feedbacks
arr((3.98, 2.82), (5.55, 2.68), ORANGE, cs='arc3,rad=0.25'); lab(4.85, 2.25, 'μ₀F', ORANGE)
arr((3.05, 2.72), (3.05, 1.62), ORANGE, cs='arc3,rad=0.35'); lab(2.55, 2.1, 'αF(1 − C)', ORANGE)
arr((2.55, 1.15), (1.2, 2.68), BLUE, cs='arc3,rad=-0.3');     lab(1.25, 1.75, 'γC(1 − S)', BLUE)
lab(5.55, 0.68, 'each variable decays (δ_D, δ_C, δ_S)', GREY, fs=8.5)

# therapy
box(1.35, 5.75, 'T_S', 'agent acting on spatial structure', RED, w=2.3, h=0.72, fs=12)
box(3.5, -0.75, 'T_C', 'agent disrupting cooperative signaling', RED, w=2.6, h=0.72, fs=12)
arr((1.35, 5.38), (1.35, 3.52), RED, style='-[, widthB=0.6, lengthB=0.25'); lab(0.8, 4.15, '− κT_S', RED)
arr((3.5, -0.38), (3.5, 0.8), RED, style='-[, widthB=0.6, lengthB=0.25'); lab(3.05, 0.2, '− βT_C', RED)

# clonal layer
box(9.25, 3.1, 'N_r', 'resistant cells (cost c)', PURPLE, w=2.3)
box(9.25, 1.2, 'N_s', 'sensitive cells', PURPLE, w=2.3)
arr((9.25, 1.6), (9.25, 2.7), PURPLE); lab(9.95, 2.15, 'mutation', PURPLE, fs=8.5)
lab(8.3, 2.15, 'compete for\ncarrying\ncapacity K', PURPLE, fs=8)
# couplings
arr((3.85, 3.55), (8.3, 3.52), GREEN, cs='arc3,rad=-0.3')
lab(6.45, 4.28, 'ω: F boosts growth of both clones, r₀(1 + ωF)', GREEN, fs=9)
arr((8.1, 2.9), (4.47, 1.2), GREEN, style='-[, widthB=0.6, lengthB=0.25')
lab(6.6, 1.55, 'φ: resistant fraction ρ\ncaps C_max = 1 − φρ', GREEN, fs=9)
arr((4.8, -0.75), (9.25, 0.8), RED, style='-[, widthB=0.6, lengthB=0.25', cs='angle,angleA=0,angleB=90,rad=8')
lab(7.0, -0.75, 'T_C also kills sensitive cells', RED)

ax.text(5.6, -1.28, 'Extensions: the same rules on an explicit lattice (agent-based model) and several metastatic sites coupled by migration.',
        fontsize=8.5, style='italic', color=GREY, ha='center')
plt.tight_layout(); plt.savefig(FIGDIR / 'Fig1.png', dpi=200, bbox_inches='tight')
print('Saved Fig1.png')
