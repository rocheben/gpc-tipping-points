"""
Conditional irreversibility: once collapse is induced, how long does the system
stay collapsed when intrinsic noise is present and therapy has stopped?
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location('m', HERE / 'fig3_sequential_vs_simultaneous.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P = m.P

def relapse_curves(sigma, n=400, t_after=300.0, dt=0.05, seed=2):
    rng = np.random.default_rng(seed)
    X = np.tile(np.array([0.083, 0.0, 0.0]), (n, 1))   # start in the collapsed attractor
    steps = int(t_after/dt); frac = np.zeros(steps)
    relapsed = np.zeros(n, bool)
    for i in range(steps):
        D, C, S = X.T
        Fv = C*S*4*D*(1-D)
        dX = np.column_stack([P['mu0']*Fv + P['mu1'] - P['delta_D']*D,
                              P['alpha']*Fv*(1-C) - P['delta_C']*C,
                              P['gamma']*C*(1-S) - P['delta_S']*S])
        X = np.clip(X + dX*dt + sigma*np.sqrt(dt)*rng.standard_normal(X.shape), 0, 1)
        relapsed |= (X[:, 1]*X[:, 2]*4*X[:, 0]*(1-X[:, 0])) > 0.3
        frac[i] = 1 - relapsed.mean()
    return np.arange(steps)*dt, frac

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sigma, c in [(0.01, 'C0'), (0.015, 'C1'), (0.02, 'C2'), (0.025, 'C4'), (0.03, 'C3')]:
        t, f = relapse_curves(sigma)
        ax.plot(t, f, color=c, lw=2, label=f'sigma = {sigma}')
        print(f'sigma={sigma}: fraction still collapsed at t=100: {f[int(100/0.05)-1]:.2f}, t=300: {f[-1]:.2f}')
        half = t[np.argmax(f <= 0.5)] if (f <= 0.5).any() else float('nan')
        print(f'   time at which half of the runs have relapsed: {half:.1f}')
    ax.set_xlabel('Time after therapy withdrawal')
    ax.set_ylabel('Fraction of runs still collapsed')
    ax.set_title('Collapse is metastable: noise-driven return to the functional GPC')
    ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIGDIR / 'FigS3.png', dpi=150, bbox_inches='tight')
    print('Saved FigS3.png')
