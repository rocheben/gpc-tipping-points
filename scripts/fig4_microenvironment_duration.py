"""
Thomas framing note, Hypothesis 3: modifying the microenvironment induces
systemic transitions. Perturbation of spatial structure S alone (T_S) compared
with perturbation of coherence C alone (T_C), as a function of exposure duration.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location('m', HERE / 'fig3_sequential_vs_simultaneous.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.T_END = 400.0
ws = np.array([5, 8, 10, 12, 15, 20, 25, 30, 40, 60])
aC = np.array([m.critical_amplitude('C', w, hi=3) for w in ws])
aS = np.array([m.critical_amplitude('S', w, hi=3) for w in ws])
for w, c, s in zip(ws, aC, aS): print(f'w={w:>3}  C-only a*={c:.3f}  S-only a*={s:.3f}')
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(ws, aC, 'o-', color='C0', lw=2, label='Coherence channel only (T_C)')
ax.plot(ws, aS, 's-', color='C1', lw=2, label='Microenvironment channel only (T_S)')
for w, s in zip(ws, aS):
    if not np.isfinite(s): ax.plot(w, 3.0, 'x', color='C1', ms=9)
ax.set_xlabel('Exposure duration w'); ax.set_ylabel('Critical amplitude a* for collapse')
ax.set_title('Microenvironmental perturbation tips the GPC only with sustained exposure')
ax.text(5, 2.85, 'x : no collapse up to a = 3', color='C1', fontsize=9)
ax.set_ylim(0, 3.1); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR / 'Fig4.png', dpi=150, bbox_inches='tight')
print('Saved Fig4.png')
