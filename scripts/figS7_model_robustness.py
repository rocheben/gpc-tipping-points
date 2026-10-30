"""
Robustness of the collective-model results to two modelling choices.

1. Collapse criterion. Collapse is read as F < 0.05 at the end of the run. Critical
   amplitudes are recomputed with cutoffs 0.01, 0.025, 0.05 and 0.10.

2. Coupling between coherence and spatial structure. In the reference model spatial
   structure is reinforced directly by coherence (gamma C (1 - S)), and coherence by the
   collective function F (alpha F (1 - C)), so S acts on C only through F. Two alternative
   couplings are tested:
     symmetric : dS/dt = gamma' F (1 - S) - ...   (both C and S reinforced by F)
     inverted  : dC/dt = alpha' S (1 - C) - ...,  dS/dt = gamma' F (1 - S) - ...
                 (coherence now sits downstream of spatial structure)
   alpha' and gamma' are rescaled so that the functional equilibrium of the reference model
   is unchanged (alpha' = alpha F*/S*, gamma' = gamma C*/F*); all other parameters are those
   of Table S2. Therapy channels are unchanged (beta T_C on C, kappa T_S on S).

Outputs: critical amplitudes printed to the log and Fig. S7.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
import importlib.util
spec = importlib.util.spec_from_file_location('m', HERE / 'fig3_sequential_vs_simultaneous.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

P = dict(m.P)
DT, T0 = m.DT, m.T0
IC = m.IC.copy()
Dst, Cst, Sst = IC
Fst = Cst*Sst*4*Dst*(1 - Dst)


def final_F(sched, variant='reference', t_end=250.0, ic=None):
    """Explicit Euler, step DT, variables clipped to [0, 1] (as in fig3), in plain floats for speed."""
    D, C, S = (IC if ic is None else ic)
    mu0, mu1, dD_, al, be, dC_, ga, ka, dS_ = (P[k] for k in
        ('mu0', 'mu1', 'delta_D', 'alpha', 'beta', 'delta_C', 'gamma', 'kappa', 'delta_S'))
    alp, gap = al*Fst/Sst, ga*Cst/Fst
    cl = lambda x: 0.0 if x < 0 else (1.0 if x > 1 else x)
    for i in range(int(t_end/DT)):
        tc, ts = sched(i*DT)
        F = C*S*4*D*(1 - D)
        dD = mu0*F + mu1 - dD_*D
        if variant == 'reference':
            gC, gS = al*F*(1 - C), ga*C*(1 - S)
        elif variant == 'symmetric':
            gC, gS = al*F*(1 - C), gap*F*(1 - S)
        else:
            gC, gS = alp*S*(1 - C), gap*F*(1 - S)
        D, C, S = cl(D + dD*DT), cl(C + (gC - be*tc - dC_*C)*DT), cl(S + (gS - ka*ts - dS_*S)*DT)
    return C*S*4*D*(1 - D), np.array([D, C, S])


def bisect(make, variant='reference', cut=0.05, t_end=250.0, lo=0.0, hi=3.0, tol=1e-3):
    if not final_F(make(hi), variant, t_end)[0] < cut:
        return np.nan
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if final_F(make(mid), variant, t_end)[0] < cut: hi = mid
        else: lo = mid
    return hi


SCHED = {  # name: (constructor of a schedule from an amplitude, end time)
    'equal amplitude, T_C only':        (lambda a: m.schedule('C', a, 5.0), 250.0),
    'equal amplitude, simultaneous':    (lambda a: m.schedule('sim', a, 5.0), 250.0),
    'equal amplitude, T_C then T_S':    (lambda a: m.schedule('CS', a, 5.0), 250.0),
    'equal amplitude, T_S then T_C':    (lambda a: m.schedule('SC', a, 5.0), 250.0),
    'equal peak total, simultaneous':   (lambda a: m.schedule_burden('sim', a, 16), 250.0),
    'equal peak total, T_C then T_S':   (lambda a: m.schedule_burden('CS', a, 16), 250.0),
    'equal peak total, T_S then T_C':   (lambda a: m.schedule_burden('SC', a, 16), 250.0),
    'equal peak total, T_C only':       (lambda a: m.schedule_burden('Cfull', a, 16), 250.0),
    'T_C only, w = 5':                  (lambda a: m.schedule('C', a, 5.0), 400.0),
    'T_C only, w = 60':                 (lambda a: m.schedule('C', a, 60.0), 400.0),
    'T_S only, w = 12':                 (lambda a: m.schedule('S', a, 12.0), 400.0),
    'T_S only, w = 40':                 (lambda a: m.schedule('S', a, 40.0), 400.0),
}

if __name__ == '__main__':
    # ---------------------------------------------------------------- 1. collapse cutoff
    CUTS = [0.01, 0.025, 0.05, 0.10]
    print('1. Critical amplitude against the collapse cutoff (F below cutoff at the end of the run)')
    print(f'  {"schedule":<34}' + ''.join(f'{c:>9}' for c in CUTS))
    cut_res = {}
    for name, (mk, te) in SCHED.items():
        cut_res[name] = [bisect(mk, cut=c, t_end=te) for c in CUTS]
        print(f'  {name:<34}' + ''.join(f'{x:>9.3f}' for x in cut_res[name]))
    spread = max(np.nanmax(v) - np.nanmin(v) for v in cut_res.values())
    print(f'  largest difference across cutoffs: {spread:.3f}')
    Fend = []
    for name, (mk, te) in SCHED.items():
        for a in np.linspace(0.05, 2.0, 40):
            Fend.append(final_F(mk(a), t_end=te)[0])
    Fend = np.array(Fend)
    print(f'  end-of-run F over {len(Fend)} runs (amplitudes 0.05 to 2): '
          f'{(Fend < 0.01).sum()} below 0.01, {((Fend >= 0.01) & (Fend < 0.3)).sum()} between 0.01 and 0.3, '
          f'{(Fend >= 0.3).sum()} above 0.3')

    # ---------------------------------------------------------------- 2. coupling variants
    VARS = ['reference', 'symmetric', 'inverted']
    print('\n2. Alternative couplings between coherence and spatial structure')
    print(f'  rescaled constants: alpha\' = {P["alpha"]*Fst/Sst:.3f}, gamma\' = {P["gamma"]*Cst/Fst:.3f}')
    zero = lambda t: (0.0, 0.0)
    for v in VARS:
        Fa, Xa = final_F(zero, v, 300.0)
        Fb, Xb = final_F(zero, v, 300.0, ic=np.array([0.5, 0.2, 0.2]))
        print(f'  {v:<10} from functional state: D, C, S, F = {Xa[0]:.3f} {Xa[1]:.3f} {Xa[2]:.3f} {Fa:.3f};'
              f' from (0.5, 0.2, 0.2): F = {Fb:.3f}')
    WS = np.array([5, 8, 10, 12, 15, 20, 30, 40, 60])
    dur = {}
    print('  critical amplitude against exposure duration w (single channel, end 400):')
    for v in VARS:
        aC = np.array([bisect(lambda a, w=w: m.schedule('C', a, float(w)), v, t_end=400.0) for w in WS])
        aS = np.array([bisect(lambda a, w=w: m.schedule('S', a, float(w)), v, t_end=400.0) for w in WS])
        dur[v] = (aC, aS)
        print(f'  {v:<10} T_C: ' + ' '.join('  -  ' if not np.isfinite(x) else f'{x:.3f}' for x in aC))
        print(f'  {"":<10} T_S: ' + ' '.join('  -  ' if not np.isfinite(x) else f'{x:.3f}' for x in aS))
    print(f'  (w = {" ".join(str(w) for w in WS)}; "-" = no collapse up to amplitude 3)')
    KS = ['equal amplitude, T_C only', 'equal amplitude, simultaneous', 'equal amplitude, T_C then T_S',
          'equal amplitude, T_S then T_C', 'equal peak total, simultaneous', 'equal peak total, T_C then T_S',
          'equal peak total, T_S then T_C', 'equal peak total, T_C only']
    extra = {'equal amplitude, T_S only': (lambda a: m.schedule('S', a, 5.0), 250.0),
             'equal peak total, T_S only': (lambda a: (lambda t: (0.0, a) if T0 <= t < T0 + 16 else (0.0, 0.0)), 250.0)}
    allk = KS + list(extra)
    sch = {**SCHED, **extra}
    var_res = {v: {k: bisect(sch[k][0], v, t_end=sch[k][1]) for k in allk} for v in VARS}
    print('  sequencing comparisons (window 5 for equal amplitude, course 16 for equal peak total):')
    print(f'  {"schedule":<34}' + ''.join(f'{v:>12}' for v in VARS))
    for k in allk:
        print(f'  {k:<34}' + ''.join('           -' if not np.isfinite(var_res[v][k]) else f'{var_res[v][k]:>12.3f}' for v in VARS))

    DEL = [0, 5, 10, 20, 30]
    print('  delay between strikes, equal amplitude (window 5), critical amplitude for d =', DEL)
    for v in VARS:
        for kind, nm in [('CS', 'T_C then T_S'), ('SC', 'T_S then T_C')]:
            vals = [bisect(lambda a, d=d: m.schedule(kind, a, 5.0, d), v) for d in DEL]
            print(f'  {v:<10} {nm}: ' + ' '.join(f'{x:.3f}' for x in vals))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    a0 = ax[0]
    for i, k in enumerate(KS):
        a0.plot(CUTS, cut_res[k], 'o-', color=f'C{i}', label=k.replace('T_C', '$T_C$').replace('T_S', '$T_S$'))
    a0.set_xscale('log'); a0.set_xticks(CUTS); a0.set_xticklabels([str(c) for c in CUTS])
    a0.set_xlabel('Collapse cutoff on F'); a0.set_ylabel('Critical amplitude')
    a0.set_title('A. Thresholds do not depend on the cutoff'); a0.legend(fontsize=9); a0.grid(alpha=0.3)
    a1 = ax[1]
    sty = {'reference': '-', 'symmetric': '--', 'inverted': ':'}
    for v in VARS:
        aC, aS = dur[v]
        a1.plot(WS, aC, 'o' + sty[v], color='C0', lw=2, label=f'$T_C$, {v}')
        a1.plot(WS, aS, 's' + sty[v], color='C1', lw=2, label=f'$T_S$, {v}')
    a1.set_ylim(0, 1.7); a1.set_xlabel('Exposure duration w'); a1.set_ylabel('Critical amplitude a*')
    a1.set_title('B. The channel placed downstream\nneeds sustained exposure'); a1.legend(fontsize=9, ncol=2); a1.grid(alpha=0.3)
    a2 = ax[2]
    short = ['$T_C$', '$T_S$', 'sim', '$T_C$→$T_S$', '$T_S$→$T_C$']
    grp = [['equal amplitude, T_C only', 'equal amplitude, T_S only', 'equal amplitude, simultaneous',
            'equal amplitude, T_C then T_S', 'equal amplitude, T_S then T_C'],
           ['equal peak total, T_C only', 'equal peak total, T_S only', 'equal peak total, simultaneous',
            'equal peak total, T_C then T_S', 'equal peak total, T_S then T_C']]
    x = np.arange(len(VARS)); wb = 0.08
    for gi, g in enumerate(grp):
        for j, k in enumerate(g):
            vals = [var_res[v][k] for v in VARS]
            xx = x + (gi*5 + j - 4.5)*wb
            a2.bar(xx, [0 if not np.isfinite(y) else y for y in vals], wb, color=f'C{j}', alpha=0.45 if gi == 0 else 0.95,
                   hatch='' if gi == 0 else '//', edgecolor='k', label=short[j] if gi == 0 else None)
            for xi, y in zip(xx, vals):
                if not np.isfinite(y): a2.plot(xi, 0.03, 'x', color='k', ms=8, mew=2)
    a2.set_xticks(x); a2.set_xticklabels(VARS); a2.set_ylim(0, 0.9)
    a2.set_ylabel('Critical amplitude')
    a2.set_title('C. Sequencing comparisons by coupling\n(plain: equal amplitude; hatched: equal peak total;\nx: no collapse up to amplitude 3)')
    a2.legend(fontsize=10, ncol=5, loc='upper center'); a2.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(FIGDIR / 'FigS7.png', dpi=150, bbox_inches='tight')
    print('\nSaved FigS7.png')
