"""
GPC model applied to the Zhang et al. 2022 abiraterone case (eLife 2022).

The bare GPC model has no clone-level resolution and no explicit resistance.
With or without a generic resistance term, it cannot reproduce the precise
~2x TTP ratio reported clinically (that requires clone-level structure).
What it DOES show, qualitatively, is the GPC-level mechanism the authors
invoke: adaptive dosing keeps the system NEAR but not PAST the tipping point,
oscillating around the separatrix. This is a complementary, not competing,
account to clone-based models.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt

PARAMS = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.8,
              delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15)

def g(D): return 4.0 * D * (1.0 - D)
def F(D, C, S): return C * S * g(D)

def step(state, T, dt=0.05, sigma=0.012):
    D, C, S = state
    D = np.clip(D, 0, 1); C = np.clip(C, 0, 1); S = np.clip(S, 0, 1)
    Fv = F(D, C, S)
    p = PARAMS
    dD = p['mu0']*Fv + p['mu1'] - p['delta_D']*D
    dC = p['alpha']*Fv*(1-C) - p['beta']*T - p['delta_C']*C
    dS = p['gamma']*C*(1-S) - p['kappa']*T - p['delta_S']*S
    new = np.array([D + dD*dt, C + dC*dt, S + dS*dt]) + sigma*np.sqrt(dt)*np.random.randn(3)
    return np.clip(new, 0, 1)

def run(sched_fn, t_end=300, dt=0.05, seed=0):
    np.random.seed(seed)
    n = int(t_end/dt)
    traj = np.zeros((n, 3))
    Ts = np.zeros(n)
    traj[0] = [0.7, 0.8, 0.8]
    for i in range(1, n):
        Ts[i] = sched_fn(i*dt, traj[i-1])
        traj[i] = step(traj[i-1], Ts[i], dt=dt)
    return np.arange(n)*dt, traj, Ts

def continuous_high(t, state): return 0.7

def make_adaptive(F_on=0.45, F_off=0.30, T_amp=0.7):
    sv = {'on': True}
    def sched(t, state):
        Fv = F(*state)
        if sv['on'] and Fv < F_off: sv['on'] = False
        elif (not sv['on']) and Fv > F_on: sv['on'] = True
        return T_amp if sv['on'] else 0.0
    return sched

t_c, traj_c, Ts_c = run(continuous_high, t_end=300, seed=0)
t_a, traj_a, Ts_a = run(make_adaptive(), t_end=300, seed=0)
F_c = np.array([F(*s) for s in traj_c])
F_a = np.array([F(*s) for s in traj_a])
cum_c = np.cumsum(Ts_c) * 0.05
cum_a = np.cumsum(Ts_a) * 0.05

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(t_c, F_c, 'C3-', label='Continuous high-dose', lw=1.5)
ax.plot(t_a, F_a, 'C0-', label='Adaptive (GPC)', lw=1.5)
ax.axhline(0.05, color='gray', ls=':', alpha=0.7, label='Collapse threshold')
ax.set_xlabel('Time'); ax.set_ylabel('Tumor functionality F')
ax.set_title('Both schedules suppress F; adaptive oscillates near tipping point')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_c, cum_c, 'C3-', label='Continuous high-dose', lw=2)
ax.plot(t_a, cum_a, 'C0-', label='Adaptive (GPC)', lw=2)
ax.set_xlabel('Time'); ax.set_ylabel('Cumulative drug exposure ∫T dt')
ax.set_title(f'Adaptive uses {(cum_a[-1]/cum_c[-1]*100):.0f}% of cumulative dose')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(traj_c[:,1], traj_c[:,2], 'C3-', label='Continuous high-dose', alpha=0.7, lw=1)
ax.plot(traj_a[:,1], traj_a[:,2], 'C0-', label='Adaptive (GPC)', alpha=0.7, lw=1)
ax.plot(0.8, 0.84, 'k*', markersize=15, label='Functional attractor')
ax.plot(0.02, 0.02, 'kx', markersize=12, label='Collapsed attractor')
ax.set_xlabel('C (coherence)'); ax.set_ylabel('S (spatial structure)')
ax.set_title('Adaptive trajectory hugs the separatrix without crossing')
ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1); ax.set_ylim(-0.05, 1)

ax = axes[1, 1]
ax.plot(t_c, Ts_c, 'C3-', lw=1, alpha=0.7, label='Continuous')
ax.plot(t_a, Ts_a + 0.05, 'C0-', lw=1, alpha=0.9, label='Adaptive (offset +0.05)')
ax.set_xlabel('Time'); ax.set_ylabel('Therapy intensity T(t)')
ax.set_title('Schedule comparison')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 0.85)

plt.tight_layout()
plt.savefig(FIGDIR / 'FigS4.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Total drug exposure:")
print(f"  Continuous: {cum_c[-1]:.1f}")
print(f"  Adaptive:   {cum_a[-1]:.1f}  ({cum_a[-1]/cum_c[-1]*100:.0f}% of continuous)")
print(f"\nMean F (after transient):")
print(f"  Continuous: F_mean = {F_c[2000:].mean():.3f}")
print(f"  Adaptive:   F_mean = {F_a[2000:].mean():.3f}")
print(f"\nReported for the abiraterone adaptive trial (Zhang et al. 2017, Nat Commun; 2022, eLife):")
print(f"  >2x increase in TTP, ~50% cumulative dose vs continuous")
print(f"\nGPC model qualitative match: both suppress tumor function, adaptive")
print(f"with substantially less cumulative drug. Quantitative TTP ratio")
print(f"requires clone-level resistance not captured by the bare GPC model.")
print("Saved FigS4.png")
