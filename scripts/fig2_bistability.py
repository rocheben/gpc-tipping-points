"""
GPC model: bistability between functional and collapsed attractors.

State: D (phenotypic diversity), C (functional coherence), S (spatial structure)
Functional output: F = C * S * g(D), with g(D) = 4*D*(1-D)

Dynamics:
    dD/dt = mu0 * F + mu1 - delta_D * D
    dC/dt = alpha * F * (1 - C) - beta * T  - delta_C * C
    dS/dt = gamma * C * (1 - S) - kappa * T - delta_S * S
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
from scipy.integrate import odeint

# ---------- Parameters ----------
PARAMS = dict(
    mu0=0.8, mu1=0.05,          # diversity generation: F-dependent + baseline
    delta_D=0.6,                # diversity decay
    alpha=1.2,                  # F->C coupling
    beta=0.8,                   # therapy effect on C
    delta_C=0.15,               # C baseline decay
    gamma=1.0,                  # C->S coupling
    kappa=0.6,                  # therapy effect on S
    delta_S=0.15,               # S baseline decay
)

def g(D):
    return 4.0 * D * (1.0 - D)

def F(D, C, S):
    return C * S * g(D)

def rhs(state, t, T, p):
    D, C, S = state
    D = np.clip(D, 0, 1); C = np.clip(C, 0, 1); S = np.clip(S, 0, 1)
    Fv = F(D, C, S)
    dD = p['mu0'] * Fv + p['mu1'] - p['delta_D'] * D
    dC = p['alpha'] * Fv * (1 - C) - p['beta'] * T - p['delta_C'] * C
    dS = p['gamma'] * C * (1 - S) - p['kappa'] * T - p['delta_S'] * S
    return [dD, dC, dS]

# ---------- 1. Find attractors by sweeping initial conditions ----------
def find_attractors(T=0.0, p=PARAMS, n=12, t_end=400):
    t = np.linspace(0, t_end, 2000)
    endpoints = []
    grid = np.linspace(0.05, 0.95, n)
    for d0 in grid:
        for c0 in grid:
            for s0 in grid:
                sol = odeint(rhs, [d0, c0, s0], t, args=(T, p), full_output=False)
                endpoints.append(np.clip(sol[-1], 0, 1))  # clip before recording
    endpoints = np.array(endpoints)
    Fvals = np.array([F(*e) for e in endpoints])
    collapsed = endpoints[Fvals < 0.01]
    functional = endpoints[Fvals >= 0.01]
    return collapsed, functional

collapsed, functional = find_attractors(T=0.0)
print(f"Without therapy (T=0):")
print(f"  Collapsed basin size (sampled): {len(collapsed)} / {len(collapsed)+len(functional)}")
if len(functional) > 0:
    print(f"  Functional attractor: D={functional[:,0].mean():.3f}, "
          f"C={functional[:,1].mean():.3f}, S={functional[:,2].mean():.3f}, "
          f"F={F(*functional.mean(axis=0)):.3f}")
if len(collapsed) > 0:
    print(f"  Collapsed attractor:  D={collapsed[:,0].mean():.3f}, "
          f"C={collapsed[:,1].mean():.3f}, S={collapsed[:,2].mean():.3f}")

# ---------- 2. Bifurcation diagram vs therapy intensity ----------
T_range = np.linspace(0, 1.5, 60)
upper_branch = []
lower_branch = []
for T_val in T_range:
    coll, func = find_attractors(T=T_val, n=6, t_end=300)
    if len(func) > 0:
        upper_branch.append((T_val, F(*func.mean(axis=0))))
    if len(coll) > 0:
        lower_branch.append((T_val, F(*coll.mean(axis=0))))
upper_branch = np.array(upper_branch) if upper_branch else np.empty((0,2))
if len(upper_branch):
    print(f"Functional branch found up to T = {upper_branch[-1,0]:.3f}; next grid value T = {T_range[np.searchsorted(T_range, upper_branch[-1,0])+1]:.3f} has none")
lower_branch = np.array(lower_branch) if lower_branch else np.empty((0,2))

# ---------- 3. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
if len(upper_branch) > 0:
    ax.plot(upper_branch[:,0], upper_branch[:,1], 'o-', color='C0',
            label='Functional GPC', markersize=4)
if len(lower_branch) > 0:
    ax.plot(lower_branch[:,0], lower_branch[:,1], 's-', color='C3',
            label='Collapsed GPC', markersize=4)
ax.set_xlabel('Therapy intensity T')
ax.set_ylabel('Tumor functionality F = C·S·g(D)')
ax.set_title('Bifurcation: bistability and tipping point')
ax.legend()
ax.grid(alpha=0.3)

# Time series: subcritical vs supracritical perturbation
ax = axes[1]
def euler_run(ic, T_amp, t_on=30, t_off=80, t_end=200, dt=0.05, p=PARAMS):
    n = int(t_end/dt)
    out = np.zeros((n, 3))
    state = np.array(ic)
    for i in range(n):
        out[i] = state
        t_cur = i*dt
        T_now = T_amp if t_on <= t_cur <= t_off else 0.0
        D, C, S = np.clip(state, 0, 1)
        Fv = F(D, C, S)
        dD = p['mu0']*Fv + p['mu1'] - p['delta_D']*D
        dC = p['alpha']*Fv*(1-C) - p['beta']*T_now - p['delta_C']*C
        dS = p['gamma']*C*(1-S) - p['kappa']*T_now - p['delta_S']*S
        state = np.clip(state + np.array([dD, dC, dS])*dt, 0, 1)
    return np.arange(n)*dt, out

ic = [0.5, 0.7, 0.7]
for T_amp, color, label in [(0.10, 'C0', 'Subcritical (T=0.10)'),
                              (0.30, 'C3', 'Supracritical (T=0.30)')]:
    t, sol = euler_run(ic, T_amp)
    Ft = np.array([F(*s) for s in sol])
    ax.plot(t, Ft, color=color, label=label, lw=2)

ax.axvspan(30, 80, alpha=0.15, color='gray', label='Therapy window')
ax.set_xlabel('Time')
ax.set_ylabel('F(t)')
ax.set_title('Reversible vs irreversible destabilization')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGDIR / 'Fig2.png', dpi=150, bbox_inches='tight')
print("\nSaved Fig2.png")
plt.close()
