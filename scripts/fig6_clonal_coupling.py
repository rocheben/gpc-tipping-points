"""
Coupling the GPC model to a clonal compartment with explicit resistance.

Clonal layer: sensitive S_n and resistant R_n cells, competing for a shared
carrying capacity. Resistance carries a fitness cost c. Therapy kills sensitive
cells only (channel T_C also acts on coherence, as before).

Coupling, in both directions:
  clones -> GPC : resistant cells cooperate less, so the achievable coherence is
                  capped at Cmax = 1 - phi * rho, with rho the resistant fraction.
  GPC -> clones : collective function F supplies a cooperative growth benefit
                  shared by both clones, so proliferation is r0*(1 + omega*F).

This is the minimal structure needed for continuous therapy to FAIL (resistance
sweeps) while adaptive therapy retains sensitive cells that suppress resistance.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt

P = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.12, delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15,
         r0=0.10, cost=0.30, K=1.0, kill=2.0, mut=2e-5,
         phi=0.9, omega=1.0)
DT = 0.05

def derivs(y, T_C, T_S, p):
    D, C, S, Ns, Nr = y
    N = Ns + Nr
    rho = Nr/N if N > 1e-12 else 0.0
    Cmax = max(1.0 - p['phi']*rho, 0.0)
    F = C*S*4*D*(1-D)
    growth = p['r0']*(1 + p['omega']*F)*(1 - N/p['K'])
    dNs = growth*Ns - p['kill']*T_C*Ns - p['mut']*Ns
    dNr = growth*(1 - p['cost'])*Nr + p['mut']*Ns
    return np.array([
        p['mu0']*F + p['mu1'] - p['delta_D']*D,
        p['alpha']*F*(Cmax - C) - p['beta']*T_C - p['delta_C']*C,
        p['gamma']*C*(1-S) - p['kappa']*T_S - p['delta_S']*S,
        dNs, dNr])

def run(sched, t_end=1200.0, p=P, y0=(0.753, 0.801, 0.842, 0.85, 1e-4)):
    n = int(t_end/DT); y = np.array(y0, float)
    out = np.zeros((n, 5)); Ts = np.zeros(n)
    for i in range(n):
        t = i*DT
        tc, ts = sched(t, y)
        y = y + derivs(y, tc, ts, p)*DT
        y[:3] = np.clip(y[:3], 0, 1); y[3:] = np.clip(y[3:], 0, None)
        out[i] = y; Ts[i] = tc
    return np.arange(n)*DT, out, Ts

def burden(out): return out[:, 3] + out[:, 4]
def Ffunc(out): return out[:, 1]*out[:, 2]*4*out[:, 0]*(1-out[:, 0])

def ttp(t, out, thresh=0.90):
    """Progression: total burden reaches thresh x carrying capacity."""
    hit = np.where(burden(out) > thresh)[0]
    return t[hit[0]] if len(hit) else np.nan

# ---- schedules ----
def continuous(amp=1.0):
    return lambda t, y: (amp if t >= 0 else 0.0, 0.0)

def adaptive(amp=1.0, on_frac=1.0, off_frac=0.85):
    st = {'on': False, 'ref': None}
    def s(t, y):
        b = y[3] + y[4]
        if st['ref'] is None: st['ref'] = b; st['on'] = True
        if st['on'] and b < off_frac*st['ref']: st['on'] = False
        elif not st['on'] and b > on_frac*st['ref']: st['on'] = True
        return (amp if st['on'] else 0.0, 0.0)
    return s

if __name__ == '__main__':
    t, out_c, Tc = run(continuous())
    t, out_a, Ta = run(adaptive())
    tp_c, tp_a = ttp(t, out_c), ttp(t, out_a)
    print(f'burden at diagnosis {burden(out_c)[0]:.3f}')
    print(f'TTP continuous {tp_c:.0f}   adaptive {tp_a:.0f}   ratio {tp_a/tp_c:.2f}')
    dose_c = Tc.sum()*DT; dose_a = Ta.sum()*DT
    print(f'cumulative dose: continuous {dose_c:.0f}, adaptive {dose_a:.0f} '
          f'({100*dose_a/dose_c:.0f}%)')
    for nm, o in [('continuous', out_c), ('adaptive', out_a)]:
        rho = o[:, 4]/np.maximum(o[:, 3]+o[:, 4], 1e-12)
        i5 = np.argmax(rho > 0.5)
        print(f'{nm}: resistant fraction >50% at t = {t[i5]:.0f}' if rho.max() > 0.5
              else f'{nm}: resistant fraction stays below 50% (max {rho.max():.2f})')

    # sensitivity of the TTP ratio to the two coupling strengths
    print('\nTTP ratio (adaptive/continuous) across coupling strengths:')
    print('  phi\\omega ' + ' '.join(f'{w:>6.1f}' for w in [0.0, 0.5, 1.0, 2.0]))
    grid = {}
    for ph in [0.0, 0.45, 0.9]:
        row = []
        for om in [0.0, 0.5, 1.0, 2.0]:
            p = dict(P, phi=ph, omega=om)
            _, oc, _ = run(continuous(), p=p); _, oa, _ = run(adaptive(), p=p)
            a, b = ttp(t, oc), ttp(t, oa)
            v = (b/a) if (np.isfinite(a) and np.isfinite(b) and a > 0) else np.nan
            row.append(v); grid[(ph, om)] = v
        print(f'  {ph:<9.2f}' + ' '.join(f'{x:>6.2f}' if np.isfinite(x) else '   inf' for x in row))

    # sensitivity of the TTP ratio to the burden at diagnosis (reference phi, omega)
    print('\nTTP ratio vs initial sensitive burden (phi = %.2f, omega = %.1f):' % (P['phi'], P['omega']))
    for nb in [0.70, 0.75, 0.80, 0.85]:
        y0 = (0.753, 0.801, 0.842, nb, 1e-4)
        _, oc, _ = run(continuous(), y0=y0); _, oa, _ = run(adaptive(), y0=y0)
        a, b = ttp(t, oc), ttp(t, oa)
        print(f'  N_s(0) = {nb:.2f}: TTP continuous {a:.0f}, adaptive {b:.0f}, ratio {b/a:.2f}')

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a0 = ax[0, 0]
    a0.plot(t, burden(out_c), 'C3', lw=1.8, label='Continuous')
    a0.plot(t, burden(out_a), 'C0', lw=1.8, label='Adaptive')
    a0.axhline(0.90, color='k', ls='--', lw=1, label='Progression threshold')
    for tp, c in [(tp_c, 'C3'), (tp_a, 'C0')]:
        if np.isfinite(tp): a0.plot(tp, 0.90, 'v', color=c, ms=9)
    a0.set_xlabel('Time'); a0.set_ylabel('Total tumour burden')
    a0.set_title('A. Time to progression'); a0.legend(fontsize=8); a0.grid(alpha=0.3)

    a1 = ax[0, 1]
    for o, c, nm in [(out_c, 'C3', 'Continuous'), (out_a, 'C0', 'Adaptive')]:
        a1.plot(t, o[:, 4]/np.maximum(o[:, 3]+o[:, 4], 1e-12), color=c, lw=1.8, label=nm)
    a1.set_xlabel('Time'); a1.set_ylabel('Resistant fraction')
    a1.set_title('B. Competitive suppression of resistance')
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    a2 = ax[1, 0]
    for o, c, nm in [(out_c, 'C3', 'Continuous'), (out_a, 'C0', 'Adaptive')]:
        a2.plot(t, Ffunc(o), color=c, lw=1.5, label=nm)
    a2.set_xlabel('Time'); a2.set_ylabel('Collective function F')
    a2.set_title('C. GPC state under the two schedules')
    a2.legend(fontsize=8); a2.grid(alpha=0.3)

    a3 = ax[1, 1]
    oms = [0.0, 0.5, 1.0, 2.0]
    for ph, c, m in [(0.0, 'C7', 'o'), (0.45, 'C9', 's'), (0.9, 'C0', '^')]:
        a3.plot(oms, [grid[(ph, o)] for o in oms], marker=m, color=c, lw=1.8,
                label=f'phi = {ph}')
    a3.axhline(1.0, color='k', ls=':', lw=1)
    a3.set_xlabel('Cooperative growth benefit omega')
    a3.set_ylabel('TTP ratio, adaptive / continuous')
    a3.set_title('D. Cooperative growth benefit erodes the adaptive advantage')
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(FIGDIR / 'Fig6.png', dpi=150, bbox_inches='tight')
    print('\nSaved Fig6.png')
