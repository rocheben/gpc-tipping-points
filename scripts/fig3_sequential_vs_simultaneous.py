"""
Thomas framing note, Hypothesis 2: sequential perturbations are more effective
than simultaneous interventions.

Two therapy channels:
  T_C acts on functional coherence C (e.g. blocking shared cooperative signaling)
  T_S acts on spatial structure S   (e.g. microenvironmental / vascular perturbation)

Each channel is applied at amplitude a for a window of length w, so the
cumulative dose per channel (a*w) is identical across schedules:
  SIMULTANEOUS : T_C and T_S both on during [t0, t0+w]
  SEQ C->S     : T_C on [t0, t0+w], then T_S on [t0+w+d, t0+2w+d]
  SEQ S->C     : same with order reversed
Outcome: does the system end in the collapsed basin after all therapy stops?
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt

P = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.8,
         delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15)
IC = np.array([0.753, 0.801, 0.842])   # functional attractor (from fig1_bistability.py)
DT, T0, T_END = 0.05, 10.0, 250.0

def schedule(kind, a, w, d=0.0):
    t1 = T0 + w
    if kind == 'sim':
        return lambda t: (a if T0 <= t < t1 else 0.0, a if T0 <= t < t1 else 0.0)
    if kind == 'CS':
        return lambda t: (a if T0 <= t < t1 else 0.0, a if t1+d <= t < t1+d+w else 0.0)
    if kind == 'SC':
        return lambda t: (a if t1+d <= t < t1+d+w else 0.0, a if T0 <= t < t1 else 0.0)
    if kind == 'C':
        return lambda t: (a if T0 <= t < t1 else 0.0, 0.0)
    if kind == 'S':
        return lambda t: (0.0, a if T0 <= t < t1 else 0.0)

def simulate(sched, n=1, sigma=0.0, seed=0, record=False):
    rng = np.random.default_rng(seed)
    X = np.tile(IC, (n, 1))
    steps = int(T_END/DT)
    rec = np.zeros((steps, n, 3)) if record else None
    for i in range(steps):
        tc, ts = sched(i*DT)
        D, C, S = X[:, 0], X[:, 1], X[:, 2]
        Fv = C*S*4*D*(1-D)
        dX = np.column_stack([P['mu0']*Fv + P['mu1'] - P['delta_D']*D,
                              P['alpha']*Fv*(1-C) - P['beta']*tc - P['delta_C']*C,
                              P['gamma']*C*(1-S) - P['kappa']*ts - P['delta_S']*S])
        X = np.clip(X + dX*DT + (sigma*np.sqrt(DT)*rng.standard_normal(X.shape) if sigma else 0), 0, 1)
        if record: rec[i] = X
    Ffinal = X[:, 1]*X[:, 2]*4*X[:, 0]*(1-X[:, 0])
    return (Ffinal < 0.05), rec

def critical_amplitude(kind, w, d=0.0, lo=0.0, hi=3.0, tol=1e-3):
    """Smallest amplitude a producing collapse (deterministic bisection)."""
    if not simulate(schedule(kind, hi, w, d))[0][0]:
        return np.nan
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if simulate(schedule(kind, mid, w, d))[0][0]: hi = mid
        else: lo = mid
    return hi


def schedule_burden(kind, B, tau):
    """Equal duration tau, equal cumulative dose B*tau, equal peak total burden B."""
    h = T0 + tau/2; e = T0 + tau
    if kind == 'sim':
        return lambda t: (B/2, B/2) if T0 <= t < e else (0.0, 0.0)
    if kind == 'CS':
        return lambda t: (B, 0.0) if T0 <= t < h else ((0.0, B) if h <= t < e else (0.0, 0.0))
    if kind == 'SC':
        return lambda t: (0.0, B) if T0 <= t < h else ((B, 0.0) if h <= t < e else (0.0, 0.0))
    if kind == 'Cfull':
        return lambda t: (B, 0.0) if T0 <= t < e else (0.0, 0.0)

def crit(make, lo=0.0, hi=4.0, tol=1e-3):
    if not simulate(make(hi))[0][0]:
        return np.nan
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if simulate(make(mid))[0][0]: hi = mid
        else: lo = mid
    return hi

if __name__ == '__main__':
    lab = {'sim': 'Simultaneous', 'CS': 'Sequential C then S', 'SC': 'Sequential S then C',
           'Cfull': 'C only (control)', 'C': 'C only', 'S': 'S only'}
    col = {'sim': 'C3', 'CS': 'C0', 'SC': 'C2', 'Cfull': 'C7'}

    # Regime 1: equal dose per drug (each drug at amplitude a for w = 5)
    r1 = {k: critical_amplitude(k, 5.0) for k in ['C', 'S', 'sim', 'CS', 'SC']}
    print('Regime 1 (equal dose per drug, w=5), critical amplitude per drug:')
    for k, v in r1.items(): print(f'  {lab[k]:<22} {v:.3f}')

    # Regime 2: equal peak total burden, same duration and cumulative dose
    taus = np.array([4, 6, 10, 16, 24, 40])
    r2 = {k: np.array([crit(lambda B, k=k, tau=tau: schedule_burden(k, B, tau)) for tau in taus])
          for k in ['sim', 'CS', 'SC', 'Cfull']}
    print('\nRegime 2 (equal peak burden), critical burden B* vs tau', list(taus))
    for k, v in r2.items(): print(f'  {lab[k]:<22}', ' '.join(f'{x:.3f}' for x in v))

    # Delay effect, regime 1, standard and slow diversity dynamics
    delays = np.array([0, 2, 5, 10, 20, 30])
    delay_std = np.array([critical_amplitude('CS', 5.0, d) for d in delays])
    P.update(mu0=0.8*0.03, mu1=0.05*0.03, delta_D=0.6*0.03)
    globals()['T_END'] = 600.0
    delay_slow = np.array([critical_amplitude('CS', 5.0, d) for d in delays])
    sim_slow = critical_amplitude('sim', 5.0)
    P.update(mu0=0.8, mu1=0.05, delta_D=0.6); globals()['T_END'] = 250.0
    print('\nDelay (C then S), standard D:', ' '.join(f'{x:.3f}' for x in delay_std))
    print('Delay (C then S), slow D     :', ' '.join(f'{x:.3f}' for x in delay_slow), f'(sim {sim_slow:.3f})')

    # Collapse probability under noise, regime 2, tau = 16
    Bs = np.linspace(0.1, 0.6, 26)
    globals()['T_END'] = T0 + 16 + 60.0   # outcome read 60 time units after therapy ends
    prob = {k: [simulate(schedule_burden(k, B, 16), n=200, sigma=0.01, seed=1)[0].mean() for B in Bs]
            for k in ['sim', 'CS', 'SC']}
    globals()['T_END'] = 250.0
    for k in prob:
        p = np.array(prob[k]); print(f'  P(collapse)>=0.5 from B = {Bs[np.argmax(p>=0.5)]:.2f} ({lab[k]})')

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a0 = ax[0, 0]
    ks = ['sim', 'CS', 'SC']
    x = np.arange(len(ks)); wbar = 0.38
    v1 = [r1[k] for k in ks]; v2 = [r2[k][list(taus).index(16)] for k in ks]
    a0.bar(x - wbar/2, v1, wbar, color=[col[k] for k in ks], alpha=0.45, edgecolor='k',
           label='Equal dose per drug (w = 5)')
    a0.bar(x + wbar/2, v2, wbar, color=[col[k] for k in ks], alpha=0.95, edgecolor='k', hatch='//',
           label='Equal peak total burden (tau = 16)')
    a0.set_xticks(x); a0.set_xticklabels([lab[k] for k in ks], rotation=12)
    a0.set_ylabel('Critical intensity for collapse')
    a0.set_title('A. The winner depends on the constraint')
    a0.legend(fontsize=8); a0.grid(alpha=0.3, axis='y')

    a1 = ax[0, 1]
    for k in ['sim', 'CS', 'SC', 'Cfull']:
        a1.plot(taus, r2[k], marker='o', color=col[k], label=lab[k], ms=4,
                ls='--' if k == 'Cfull' else '-')
    a1.set_xlabel('Course duration tau'); a1.set_ylabel('Critical peak burden B*')
    a1.set_title('B. Equal peak burden: sequencing helps, concentration on C helps more')
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    a2 = ax[1, 0]
    a2.plot(delays, delay_std, 'o-', color='C0', label='C then S, standard diversity dynamics')
    a2.plot(delays, delay_slow, 's-', color='C9', label='C then S, diversity 33x slower')
    a2.axhline(r1['sim'], color='C3', ls='--', label='Simultaneous, standard')
    a2.axhline(sim_slow, color='C3', ls=':', label='Simultaneous, slow diversity')
    a2.set_xlabel('Delay between strikes d'); a2.set_ylabel('Critical amplitude a*')
    a2.set_title('C. No vulnerability window: delay never helps')
    a2.legend(fontsize=8); a2.grid(alpha=0.3)

    a3 = ax[1, 1]
    for k in ['sim', 'CS', 'SC']:
        a3.plot(Bs, prob[k], lw=2, color=col[k], label=lab[k])
    a3.set_xlabel('Peak total burden B'); a3.set_ylabel('Probability of collapse')
    a3.set_title('D. Collapse probability, noise sigma = 0.01 (200 runs, tau = 16)')
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGDIR / 'Fig3.png', dpi=150, bbox_inches='tight')
    print('Saved Fig3.png')
