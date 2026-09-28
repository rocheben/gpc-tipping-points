"""
Differential critical slowing down: does autocorrelation rise faster on C
than on D before GPC collapse? This is the key empirical signature that
distinguishes GPC collapse from Allee-driven collapse.
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

def sde_step(state, T, p, dt, sigma):
    D, C, S = state
    D = np.clip(D, 1e-6, 1); C = np.clip(C, 1e-6, 1); S = np.clip(S, 1e-6, 1)
    Fv = F(D, C, S)
    dD = (p['mu0']*Fv + p['mu1'] - p['delta_D']*D) * dt
    dC = (p['alpha']*Fv*(1-C) - p['beta']*T - p['delta_C']*C) * dt
    dS = (p['gamma']*C*(1-S) - p['kappa']*T - p['delta_S']*S) * dt
    noise = sigma * np.sqrt(dt) * np.random.randn(3)
    return np.array([D+dD+noise[0], C+dC+noise[1], S+dS+noise[2]])

def simulate_with_ramp(T_max, t_end=800, dt=0.05, sigma=0.02, seed=0):
    np.random.seed(seed)
    n = int(t_end/dt)
    traj = np.zeros((n, 3))
    traj[0] = [0.7, 0.8, 0.8]
    # slow ramp: stay at low T longer
    T_t = np.linspace(0, T_max, n)
    for i in range(1, n):
        traj[i] = sde_step(traj[i-1], T_t[i], PARAMS, dt, sigma)
    return np.arange(n)*dt, traj, T_t

def rolling_ac1(x, window):
    """Lag-1 autocorrelation in a sliding window."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window, len(x)):
        seg = x[i-window:i]
        seg = seg - seg.mean()
        if seg.std() > 1e-9:
            out[i] = np.corrcoef(seg[:-1], seg[1:])[0,1]
    return out

# Run multiple replicates ramping therapy up until collapse
n_reps = 25
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Aggregate trajectories
all_traj_D = []; all_traj_C = []; all_ac_D = []; all_ac_C = []
collapse_times = []

for rep in range(n_reps):
    t, traj, Tt = simulate_with_ramp(T_max=1.0, t_end=800, sigma=0.02, seed=rep)
    Fvals = np.array([F(*s) for s in traj])
    # define collapse: first time F drops below 0.05 and stays for 50 time units
    below = Fvals < 0.05
    if below.any():
        tc_idx = None
        for i in range(len(below)-1000):
            if below[i:i+1000].all():
                tc_idx = i
                break
        if tc_idx is None or tc_idx < 2000:
            continue
    else:
        continue
    collapse_times.append(t[tc_idx])
    # AC1 on D and C, only valid before collapse
    window = 200  # = 10 time units
    ac_D = rolling_ac1(traj[:tc_idx, 0], window)
    ac_C = rolling_ac1(traj[:tc_idx, 1], window)
    all_traj_D.append(traj[:tc_idx, 0])
    all_traj_C.append(traj[:tc_idx, 1])
    all_ac_D.append(ac_D)
    all_ac_C.append(ac_C)

print(f"Collected {len(all_ac_D)} replicates with detected collapse")

# Plot one example trajectory
ax = axes[0, 0]
t0, traj0, _ = simulate_with_ramp(T_max=1.0, t_end=800, sigma=0.02, seed=0)
F0 = np.array([F(*s) for s in traj0])
ax.plot(t0, traj0[:,0], label='D (diversity)', color='C2', alpha=0.7)
ax.plot(t0, traj0[:,1], label='C (coherence)', color='C0', alpha=0.7)
ax.plot(t0, traj0[:,2], label='S (spatial)', color='C1', alpha=0.7)
ax.plot(t0, F0, label='F (function)', color='k', lw=2)
ax.set_xlabel('Time'); ax.set_ylabel('State variables')
ax.set_title('Example trajectory under therapy ramp')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# Align AC1 traces by collapse time and average
ax = axes[0, 1]
N = 1500
aligned_D = np.full((len(all_ac_D), N), np.nan)
aligned_C = np.full((len(all_ac_C), N), np.nan)
for i, (aD, aC) in enumerate(zip(all_ac_D, all_ac_C)):
    if len(aD) >= N:
        aligned_D[i] = aD[-N:]
        aligned_C[i] = aC[-N:]

mean_D = np.nanmean(aligned_D, axis=0)
mean_C = np.nanmean(aligned_C, axis=0)
se_D = np.nanstd(aligned_D, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_D), axis=0))
se_C = np.nanstd(aligned_C, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_C), axis=0))

time_to_collapse = -np.arange(N)[::-1] * 0.05  # dt=0.05
ax.plot(time_to_collapse, mean_C, color='C0', label='AC1(C), coherence', lw=2)
ax.fill_between(time_to_collapse, mean_C-se_C, mean_C+se_C, color='C0', alpha=0.2)
ax.plot(time_to_collapse, mean_D, color='C2', label='AC1(D), diversity', lw=2)
ax.fill_between(time_to_collapse, mean_D-se_D, mean_D+se_D, color='C2', alpha=0.2)
ax.axvline(0, color='k', ls='--', alpha=0.5, label='Collapse')
ax.set_xlabel('Time before collapse')
ax.set_ylabel('Lag-1 autocorrelation')
ax.set_title(f'Differential critical slowing down (n={len(all_ac_D)} reps)')
ax.legend()
ax.grid(alpha=0.3)

# Variance comparison
def rolling_var(x, window):
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window, len(x)):
        out[i] = np.var(x[i-window:i])
    return out

aligned_var_D = np.full((len(all_traj_D), N), np.nan)
aligned_var_C = np.full((len(all_traj_C), N), np.nan)
for i, (tD, tC) in enumerate(zip(all_traj_D, all_traj_C)):
    if len(tD) >= N:
        aligned_var_D[i] = rolling_var(tD, 200)[-N:]
        aligned_var_C[i] = rolling_var(tC, 200)[-N:]

ax = axes[1, 0]
mvD = np.nanmean(aligned_var_D, axis=0)
mvC = np.nanmean(aligned_var_C, axis=0)
# normalize each by its early-window mean for fair comparison
mvD_norm = mvD / np.nanmean(mvD[:500])
mvC_norm = mvC / np.nanmean(mvC[:500])
ax.plot(time_to_collapse, mvC_norm, color='C0', label='Var(C) / baseline', lw=2)
ax.plot(time_to_collapse, mvD_norm, color='C2', label='Var(D) / baseline', lw=2)
ax.axvline(0, color='k', ls='--', alpha=0.5)
ax.set_xlabel('Time before collapse')
ax.set_ylabel('Variance (normalized)')
ax.set_title('Variance rise before collapse')
ax.legend()
ax.grid(alpha=0.3)

# Quantitative: time at which AC1 first crosses 0.9
ax = axes[1, 1]
def crossing_time(ac, thresh=0.9):
    mask = ac > thresh
    if not mask.any():
        return np.nan
    return np.argmax(mask)

cross_D = []; cross_C = []
for aD, aC in zip(all_ac_D, all_ac_C):
    cD = crossing_time(aD)
    cC = crossing_time(aC)
    if not (np.isnan(cD) or np.isnan(cC)):
        # convert to time-before-collapse
        cross_D.append((len(aD) - cD) * 0.05)
        cross_C.append((len(aC) - cC) * 0.05)

cross_D = np.array(cross_D); cross_C = np.array(cross_C)
ax.boxplot([cross_C, cross_D], tick_labels=['AC1(C)', 'AC1(D)'], widths=0.5)
ax.set_ylabel('Lead time before collapse\n(at AC1 = 0.9)')
ax.set_title(f'Lead time at AC1 = 0.9 (mean C = {cross_C.mean():.1f}, D = {cross_D.mean():.1f})')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGDIR / 'FigS4.png', dpi=150, bbox_inches='tight')
print(f"Lead time (mean) — C: {cross_C.mean():.2f}, D: {cross_D.mean():.2f}")
print(f"Difference C-D: {(cross_C - cross_D).mean():.2f} (paired)")
from scipy.stats import wilcoxon
if len(cross_C) > 5:
    stat, pval = wilcoxon(cross_C, cross_D)
    print(f"Wilcoxon paired test: stat={stat:.2f}, p={pval:.4f}")
print("Saved FigS4.png")
