"""
Quasi-potential V(C, S), diversity marginalised (Supplementary Figure S2).
This is the operational quantification of 'irreversible' destabilization.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

PARAMS = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.8,
              delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15)

def g(D): return 4.0 * D * (1.0 - D)
def F(D, C, S): return C * S * g(D)

def rhs(state, t, T, p=PARAMS):
    D, C, S = state
    D = np.clip(D, 0, 1); C = np.clip(C, 0, 1); S = np.clip(S, 0, 1)
    Fv = F(D, C, S)
    dD = p['mu0']*Fv + p['mu1'] - p['delta_D']*D
    dC = p['alpha']*Fv*(1-C) - p['beta']*T - p['delta_C']*C
    dS = p['gamma']*C*(1-S) - p['kappa']*T - p['delta_S']*S
    return [dD, dC, dS]

# Approximate the quasi-potential V by integrating the deterministic flow magnitude
# along trajectories from a grid of initial conditions to attractor.
# Use the line-integral V(x) = integral of (|f| ds) along reverse-time path - simpler:
# Use V(x) ~ -log of stationary density estimated from many noisy trajectories.

def sample_stationary(T_val, n_traj=1500, t_end=100, dt=0.1, sigma=0.05, seed=0):
    """Vectorized: simulate n_traj trajectories in parallel."""
    np.random.seed(seed)
    n_steps = int(t_end/dt)
    states = np.random.uniform(0.05, 0.95, (n_traj, 3))
    p = PARAMS
    for _ in range(n_steps):
        D = np.clip(states[:,0], 0, 1)
        C = np.clip(states[:,1], 0, 1)
        S = np.clip(states[:,2], 0, 1)
        Fv = C * S * 4*D*(1-D)
        dD = p['mu0']*Fv + p['mu1'] - p['delta_D']*D
        dC = p['alpha']*Fv*(1-C) - p['beta']*T_val - p['delta_C']*C
        dS = p['gamma']*C*(1-S) - p['kappa']*T_val - p['delta_S']*S
        f = np.column_stack([dD, dC, dS])
        states = states + f*dt + sigma*np.sqrt(dt)*np.random.randn(n_traj, 3)
        states = np.clip(states, 0, 1)
    return states

# Estimate V(C, S) at three therapy levels
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# therapy levels chosen on either side of the bifurcation of Figure 2 (functional branch lost
# between T = 0.13 and T = 0.15 in script fig2_bistability), matching the two levels used in Figure 2B
for idx, (T_val, title) in enumerate([(0.0, 'Pre-therapy (T=0)'),
                                       (0.10, 'Subcritical (T=0.10)'),
                                       (0.30, 'Supracritical (T=0.30)')]):
    endpoints = sample_stationary(T_val, n_traj=2000, t_end=80, sigma=0.05, seed=idx)
    print(f'T = {T_val:.2f}: share of noisy trajectories ending with C > 0.5: {(endpoints[:,1] > 0.5).mean():.2f}')
    # 2D histogram on (C, S), marginalize over D
    H, Cedge, Sedge = np.histogram2d(endpoints[:,1], endpoints[:,2],
                                      bins=30, range=[[0,1],[0,1]])
    # smooth
    from scipy.ndimage import gaussian_filter
    H_smooth = gaussian_filter(H, sigma=1.0)
    H_smooth = H_smooth + 1e-3  # avoid log(0)
    V = -np.log(H_smooth / H_smooth.max())

    ax = axes[idx]
    Cc = 0.5*(Cedge[:-1]+Cedge[1:])
    Sc = 0.5*(Sedge[:-1]+Sedge[1:])
    Cm, Sm = np.meshgrid(Cc, Sc, indexing='ij')
    im = ax.contourf(Cm, Sm, V, levels=20, cmap='viridis')
    ax.contour(Cm, Sm, V, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax.set_xlabel('C (functional coherence)')
    ax.set_ylabel('S (spatial structure)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Quasi-potential V')

plt.tight_layout()
plt.savefig(FIGDIR / 'FigS2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved FigS2.png")

