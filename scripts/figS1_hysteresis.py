"""
Hysteresis loop under a slow up-and-down therapy ramp (Supplementary Figure S1).
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

# Hysteresis loop: ramp T up then down, plot F(T)
fig, ax = plt.subplots(figsize=(8, 5))

# Slow ramp up then down
t_total = 1000
dt = 0.05
n = int(t_total/dt)
t = np.arange(n)*dt
T_ramp_up = np.linspace(0, 1.2, n//2)
T_ramp_dn = np.linspace(1.2, 0, n - n//2)
T_t = np.concatenate([T_ramp_up, T_ramp_dn])

# Deterministic trajectory
state = np.array([0.7, 0.8, 0.8])
traj = np.zeros((n, 3))
for i in range(n):
    f = np.array(rhs(state, 0, T_t[i]))
    state = state + f*dt
    state = np.clip(state, 0, 1)
    traj[i] = state

Ft = np.array([F(*s) for s in traj])
i_up = np.argmax(Ft[:n//2] < 0.05)
print(f'F at start of ramp: {Ft[0]:.3f}; F falls below 0.05 on the way up at T = {T_t[i_up]:.3f}')
print(f'F at the end of the downward ramp (T = {T_t[-1]:.2f}): {Ft[-1]:.3f}')
ax.plot(T_t[:n//2], Ft[:n//2], 'C0-', label='Ramp up (T increases)', lw=2)
ax.plot(T_t[n//2:], Ft[n//2:], 'C3-', label='Ramp down (T decreases)', lw=2)
# arrows
for i in [n//8, n//8*3]:
    ax.annotate('', xy=(T_t[i+10], Ft[i+10]), xytext=(T_t[i], Ft[i]),
                arrowprops=dict(arrowstyle='->', color='C0', lw=1.5))
for i in [n//2 + n//8, n//2 + n//8*3]:
    ax.annotate('', xy=(T_t[i+10], Ft[i+10]), xytext=(T_t[i], Ft[i]),
                arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))

ax.set_xlabel('Therapy intensity T')
ax.set_ylabel('Tumor functionality F')
ax.set_title('Hysteresis: the collapsed state persists after therapy returns to zero')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / 'FigS1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved FigS1.png")
