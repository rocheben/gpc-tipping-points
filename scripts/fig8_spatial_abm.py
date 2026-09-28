"""
Spatially explicit agent-based version of the GPC model (framing note, approach 5.3).

Lattice L x L (torus). Each site holds one cell with
  - a phenotype b in {A, B}          -> local diversity  g(D_loc) = 4 D_loc (1 - D_loc), D_loc = share of B
  - a cooperative state c in {0, 1}  -> local coherence  C_loc
  - a structured-niche flag s in {0,1} (microenvironment of the site) -> local structure S_loc
Local collective function F_i = C_loc S_loc g(D_loc), computed on the (2r+1)^2 Moore
neighbourhood of cell i (self included). r is the interaction radius; r = None means
every cell sees the whole lattice (well-mixed limit).

Per-capita transition rates are the ODE fluxes of fig2 and fig3 evaluated on the cell's own
neighbourhood, converted to per-capita rates:
  A -> B : (mu0 F_i + mu1) / (1 - D_loc)        B -> A : delta_D
  0 -> 1 (cooperation) : alpha F_i              1 -> 0 : delta_C + beta T_C(x) / C_glob
  0 -> 1 (structure)   : gamma C_loc            1 -> 0 : delta_S + kappa T_S(x) / S_glob
Therapy acts on the whole lattice with a per-capita hazard such that the total flux equals the
ODE term (beta T, kappa T), i.e. the drug is spatially uniform unless a mask T(x) is given.
In the well-mixed limit the expected dynamics are exactly those of the ODE, which is checked
below (panel A). All parameters are those of fig2 and fig3; nothing is re-tuned.

Outcome: collapse = no cooperative cell left at t_end (C = 0 is absorbing).
Threshold a50 = amplitude giving 50 % collapse, from a coarse scan then a refined scan.

Runtime: about 20-30 min on 2 cores (multiprocessing over all available cores).
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt, os, sys, time, importlib.util
from multiprocessing import Pool
from scipy.ndimage import uniform_filter

P = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.8, delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15)
IC = (0.753, 0.801, 0.842)          # functional attractor of the ODE (fig2_bistability.py)
L, DT, T0, T_END = 48, 0.05, 10.0, 300.0

# ------------------------------------------------------------------ ABM core
def local_mean(x, radius):
    if radius is None:
        return np.broadcast_to(x.mean(axis=(1, 2), keepdims=True), x.shape)
    k = 2*radius + 1
    return uniform_filter(x, size=(1, k, k), mode='wrap')

def run(sched, B, radius, seed, t_end=T_END, state=None, tmask=None, snap_times=()):
    rng = np.random.default_rng(seed)
    if state is None:
        b, c, s = [(rng.random((B, L, L)) < v).astype(np.float32) for v in IC]
    else:
        b, c, s = [x.copy() for x in state]
    snaps = {}
    for i in range(int(round(t_end/DT))):
        t = i*DT
        for ts in snap_times:
            if abs(t - ts) < DT/2: snaps[ts] = c[0].copy()
        TC, TS = sched(t)
        TC = np.asarray(TC, np.float32).reshape(-1, 1, 1); TS = np.asarray(TS, np.float32).reshape(-1, 1, 1)
        if tmask is not None: TC = TC*tmask; TS = TS*tmask
        Dl = local_mean(b, radius); Cl = local_mean(c, radius); Sl = local_mean(s, radius)
        Fl = Cl*Sl*4*Dl*(1 - Dl)
        Cg = c.mean(axis=(1, 2), keepdims=True); Sg = s.mean(axis=(1, 2), keepdims=True)
        r_b = np.where(b > 0, P['delta_D'], (P['mu0']*Fl + P['mu1'])/np.maximum(1 - Dl, 1e-6))
        r_c = np.where(c > 0, P['delta_C'] + P['beta']*TC/np.maximum(Cg, 1e-6), P['alpha']*Fl)
        r_s = np.where(s > 0, P['delta_S'] + P['kappa']*TS/np.maximum(Sg, 1e-6), P['gamma']*Cl)
        u = rng.random((3,) + b.shape, dtype=np.float32)
        b = np.where(u[0] < -np.expm1(-r_b*DT), 1 - b, b)
        c = np.where(u[1] < -np.expm1(-r_c*DT), 1 - c, c)
        s = np.where(u[2] < -np.expm1(-r_s*DT), 1 - s, s)
    return (b, c, s), snaps

def make_sched(kind, a, w=5.0, tau=16.0, d=0.0):
    a = np.asarray(a, float); z = np.zeros_like(a)
    t1 = T0 + w; h = T0 + tau/2; e = T0 + tau
    def f(t):
        if kind == 'C':   return (a if T0 <= t < t1 else z, z)
        if kind == 'sim': return (a, a) if T0 <= t < t1 else (z, z)
        if kind == 'CS':  return (a if T0 <= t < t1 else z, a if t1 + d <= t < t1 + d + w else z)
        if kind == 'SC':  return (a if t1 + d <= t < t1 + d + w else z, a if T0 <= t < t1 else z)
        if kind == 'sim2':   return (a/2, a/2) if T0 <= t < e else (z, z)
        if kind == 'CS2':    return (a, z) if T0 <= t < h else ((z, a) if h <= t < e else (z, z))
        if kind == 'SC2':    return (z, a) if T0 <= t < h else ((a, z) if h <= t < e else (z, z))
        if kind == 'Cfull2': return (a, z) if T0 <= t < e else (z, z)
    return f

def pcollapse(kind, amps, reps, radius, seed, **kw):
    A = np.repeat(np.asarray(amps, float), reps)
    (b, c, s), _ = run(make_sched(kind, A, **kw), len(A), radius, seed)
    Cf = c.mean(axis=(1, 2)).reshape(len(amps), reps)
    return (Cf == 0).mean(1), ((Cf > 0) & (Cf < 0.3)).mean(1)

def a50(amps, p):
    i = np.where(p >= 0.5)[0]
    if len(i) == 0: return np.nan
    i = i[0]
    if i == 0: return amps[0]
    return amps[i-1] + (0.5 - p[i-1])*(amps[i] - amps[i-1])/(p[i] - p[i-1])

def threshold_job(args):
    kind, radius, seed, kw = args
    g1 = np.round(np.arange(0.10, 0.701, 0.03), 3)
    p1, _ = pcollapse(kind, g1, 2, radius, seed, **kw)
    c = a50(g1, p1)
    if not np.isfinite(c): return (kind, radius, kw.get('d', 0.0), np.nan, np.nan)
    g2 = np.linspace(c - 0.045, c + 0.045, 10)
    p2, amb = pcollapse(kind, g2, 8, radius, seed + 1, **kw)
    return (kind, radius, kw.get('d', 0.0), a50(g2, p2), amb.max())

def sanctuary_job(args):
    radius, sides, amp, reps, seed = args
    out = []
    for side in sides:
        m = np.ones((L, L), np.float32); m[:side, :side] = 0.0
        (b, c, s), _ = run(make_sched('C', np.full(reps, amp)), reps, radius, seed + side, tmask=m)
        out.append((c.mean(axis=(1, 2)) == 0).mean())
    return radius, out

def attractor_job(radius):
    z = lambda t: (np.zeros(4), np.zeros(4))
    (b, c, s), _ = run(z, 4, radius, 11, t_end=150.0)
    Dl, Cl, Sl = [local_mean(x, radius) for x in (b, c, s)]
    return radius, (b.mean(), c.mean(), s.mean(), (Cl*Sl*4*Dl*(1-Dl)).mean())

def snapshot_job(_):
    m = np.ones((L, L), np.float32); m[:6, :6] = 0.0
    times = (T0, T0 + 5.0, 60.0, 150.0, 290.0)
    _, snaps = run(make_sched('C', np.array([0.6])), 1, 2, 5, tmask=m, snap_times=times)
    return snaps

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    t_start = time.time()
    RADII = [None, 4, 2, 1]
    KINDS = ['C', 'sim', 'CS', 'SC', 'sim2', 'CS2', 'SC2', 'Cfull2']
    jobs = [(k, r, 100*j + 7, {}) for r in RADII for j, k in enumerate(KINDS)]
    DELAYS = [0.0, 5.0, 10.0, 20.0]
    jobs += [('CS', r, 900 + int(d), {'d': d}) for r in [2, 1] for d in DELAYS if d > 0]
    SIDES = [0, 2, 3, 4, 5, 6, 8, 12]
    with Pool(os.cpu_count()) as pool:
        att = pool.map_async(attractor_job, RADII)
        snaps = pool.map_async(snapshot_job, [0])
        sanc = pool.map_async(sanctuary_job, [(r, SIDES, 0.6, 12, 300) for r in [4, 2, 1]])
        thr = pool.map(threshold_job, jobs)
        att = att.get(); snaps = snaps.get()[0]; sanc = sanc.get()

    # mean-field reference, computed with the functions of fig3
    spec = importlib.util.spec_from_file_location('m05', HERE / 'fig3_sequential_vs_simultaneous.py')
    m05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m05)
    ode = {k: m05.critical_amplitude(k, 5.0) for k in ['C', 'sim', 'CS', 'SC']}
    ode.update({k + '2': m05.crit(lambda B, k=k: m05.schedule_burden(k, B, 16)) for k in ['sim', 'CS', 'SC', 'Cfull']})

    rname = lambda r: 'well-mixed' if r is None else f'r = {r}'
    print('Stationary state without therapy (global D, C, S, mean local F):')
    for r, v in att: print(f'  {rname(r):<11}', ' '.join(f'{x:.3f}' for x in v))
    T = {(k, r, d): (a, amb) for k, r, d, a, amb in thr}
    print('\nCritical amplitude a50 (regime 1: amplitude per drug, w = 5; regime 2: peak burden, tau = 16)')
    print(f'  {"schedule":<8}{"ODE":>8}' + ''.join(f'{rname(r):>12}' for r in RADII) + '   max ambiguous share')
    for k in KINDS:
        row = [T[(k, r, 0.0)] for r in RADII]
        print(f'  {k:<8}{ode[k]:>8.3f}' + ''.join(f'{a:>12.3f}' for a, _ in row) + f'   {max(x for _, x in row):.2f}')
    print('\nDelay between strikes (C then S, regime 1), a50:')
    for r in [2, 1]:
        print(f'  {rname(r)}: ' + '  '.join(f'd={d:g}: {T[("CS", r, d)][0]:.3f}' for d in DELAYS)
              + f'   (simultaneous {T[("sim", r, 0.0)][0]:.3f})')
    print('\nSanctuary (C only, a = 0.6, w = 5; unexposed square of given side, lattice 48 x 48): P(collapse)')
    for r, v in sanc: print(f'  {rname(r):<6}', '  '.join(f'side {s}: {p:.2f}' for s, p in zip(SIDES, v)))

    # ---- figure
    fig = plt.figure(figsize=(16, 10))
    xs = {None: 8, 4: 4, 2: 2, 1: 1}
    a0 = fig.add_subplot(2, 3, 1)
    for k in KINDS:
        a0.plot(ode[k], T[(k, None, 0.0)][0], 'o', ms=7, label=k)
    lim = [0.18, 0.36]; a0.plot(lim, lim, 'k:', lw=1)
    a0.set_xlabel('critical amplitude, equations (script fig3)'); a0.set_ylabel('ABM, well-mixed, a50')
    a0.set_title('A. Well-mixed ABM recovers the mean-field thresholds'); a0.legend(fontsize=7, ncol=2); a0.grid(alpha=0.3)
    lab = {'C': 'C only', 'sim': 'Simultaneous', 'CS': 'Sequential C->S', 'SC': 'Sequential S->C',
           'sim2': 'Simultaneous', 'CS2': 'Sequential C->S', 'SC2': 'Sequential S->C', 'Cfull2': 'C only (control)'}
    col = {'C': 'C7', 'sim': 'C3', 'CS': 'C0', 'SC': 'C2', 'sim2': 'C3', 'CS2': 'C0', 'SC2': 'C2', 'Cfull2': 'C7'}
    for pos, ks, ttl in [(2, ['C', 'sim', 'CS', 'SC'], 'B. Equal dose per drug (w = 5)'),
                         (3, ['sim2', 'CS2', 'SC2', 'Cfull2'], 'C. Equal peak total burden (tau = 16)')]:
        ax = fig.add_subplot(2, 3, pos)
        for k in ks:
            ax.plot([xs[r] for r in RADII], [T[(k, r, 0.0)][0] for r in RADII], 'o-', color=col[k], label=lab[k])
            ax.plot(13, ode[k], '_', color=col[k], ms=14, mew=2)
        ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 13]); ax.set_xticklabels(['1', '2', '4', 'well-\nmixed', 'ODE'])
        ax.set_xlabel('Interaction radius r (cells)'); ax.set_ylabel('Critical intensity a50')
        ax.set_title(ttl); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    a3 = fig.add_subplot(2, 3, 4)
    for r, c_ in [(2, 'C0'), (1, 'C9')]:
        a3.plot(DELAYS, [T[('CS', r, d)][0] for d in DELAYS], 'o-', color=c_, label=f'C then S, {rname(r)}')
        a3.axhline(T[('sim', r, 0.0)][0], color=c_, ls='--', lw=1, label=f'simultaneous, {rname(r)}')
    a3.set_xlabel('Delay between strikes d'); a3.set_ylabel('a50')
    a3.set_title('D. Delay does not open a vulnerability window'); a3.legend(fontsize=8); a3.grid(alpha=0.3)
    a4 = fig.add_subplot(2, 3, 5)
    for (r, v), c_ in zip(sanc, ['C1', 'C0', 'C9']):
        a4.plot(SIDES, v, 'o-', color=c_, label=rname(r))
    a4.set_xlabel('Side of the unexposed square (cells; lattice 48 x 48)'); a4.set_ylabel('P(collapse)')
    a4.set_title('E. Dose 0.6 (about 2x threshold): a small sanctuary prevents collapse'); a4.legend(fontsize=8); a4.grid(alpha=0.3)
    a5 = fig.add_subplot(2, 3, 6)
    ts = sorted(snaps); strip = np.concatenate([np.pad(snaps[t], ((0, 0), (0, 3)), constant_values=0.5) for t in ts], axis=1)
    a5.imshow(strip, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    a5.set_xticks([L/2 + i*(L + 3) for i in range(len(ts))]); a5.set_xticklabels([f't = {t:g}' for t in ts], fontsize=8)
    a5.set_yticks([]); a5.set_title('F. Re-invasion from a 6 x 6 sanctuary (r = 2, a = 0.6)\nblack = cooperative cell', fontsize=10)
    plt.tight_layout(); plt.savefig(FIGDIR / 'Fig8.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved Fig8.png   ({(time.time() - t_start)/60:.1f} min)')
