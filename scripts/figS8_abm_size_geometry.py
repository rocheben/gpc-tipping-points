"""
Agent-based model: dependence on tumour geometry and size.

The rules are those of fig8_spatial_abm.py. Two things are generalised:
  - the lattice can be bounded (no wrap-around) instead of periodic;
  - only the sites of an occupancy mask carry cells (a disc for a round tumour). Local shares
    are computed over the occupied sites of the (2r+1) x (2r+1) Moore neighbourhood, and the
    therapy hazard is normalised by the shares over occupied sites, as in fig8.

Experiments
  1. Geometry at constant area (about 2300 cells): periodic 48 x 48 square (reference of
     Fig. 8), bounded 48 x 48 square, bounded disc of radius 27. Interaction radius r = 1 and 2.
     Outputs: state without therapy, a50 for the coherence channel alone (window 5) and for the
     single-channel control at equal peak total amplitude (course 16), and the probability of
     collapse with an untreated square pocket (amplitude 0.6), placed at the centre, and also
     at the edge of the disc.
  2. Size at r = 2: bounded discs of radius 14, 27 and 54 and periodic squares of side 24, 48
     and 96. Outputs: a50 (coherence channel, window 5) and collapse with a central untreated
     pocket (amplitude 0.6).
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, os, time, importlib.util
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
from multiprocessing import Pool
from scipy.ndimage import uniform_filter

spec = importlib.util.spec_from_file_location('m8', HERE / 'fig8_spatial_abm.py')
m8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m8)
P, IC, DT, T0, T_END = m8.P, m8.IC, m8.DT, m8.T0, m8.T_END


def domain(kind, size):
    """Returns (occupancy mask, boundary mode). kind: 'periodic', 'square' (bounded), 'disc' (bounded, radius size)."""
    if kind == 'periodic': return np.ones((size, size), np.float32), 'wrap'
    if kind == 'square':   return np.ones((size, size), np.float32), 'constant'
    n = 2*size + 2; y, x = np.mgrid[:n, :n] - (n - 1)/2
    return (x**2 + y**2 <= size**2).astype(np.float32), 'constant'


def run(sched, B, radius, seed, occ, mode, tmask=None, t_end=T_END):
    rng = np.random.default_rng(seed)
    occ3 = occ[None]; n_occ = occ.sum()
    b, c, s = [((rng.random((B,) + occ.shape) < v)*occ3).astype(np.float32) for v in IC]
    k = 2*radius + 1
    den = np.maximum(uniform_filter(occ, size=k, mode=mode), 1e-9)[None]
    lm = lambda x: uniform_filter(x, size=(1, k, k), mode=mode)/den
    for i in range(int(round(t_end/DT))):
        TC, TS = sched(i*DT)
        TC = np.asarray(TC, np.float32).reshape(-1, 1, 1); TS = np.asarray(TS, np.float32).reshape(-1, 1, 1)
        if tmask is not None: TC = TC*tmask; TS = TS*tmask
        Dl, Cl, Sl = lm(b), lm(c), lm(s)
        Fl = Cl*Sl*4*Dl*(1 - Dl)
        Cg = c.sum(axis=(1, 2), keepdims=True)/n_occ; Sg = s.sum(axis=(1, 2), keepdims=True)/n_occ
        r_b = np.where(b > 0, P['delta_D'], (P['mu0']*Fl + P['mu1'])/np.maximum(1 - Dl, 1e-6))
        r_c = np.where(c > 0, P['delta_C'] + P['beta']*TC/np.maximum(Cg, 1e-6), P['alpha']*Fl)
        r_s = np.where(s > 0, P['delta_S'] + P['kappa']*TS/np.maximum(Sg, 1e-6), P['gamma']*Cl)
        u = rng.random((3,) + b.shape, dtype=np.float32)
        b = np.where(u[0] < -np.expm1(-r_b*DT), 1 - b, b)*occ3
        c = np.where(u[1] < -np.expm1(-r_c*DT), 1 - c, c)*occ3
        s = np.where(u[2] < -np.expm1(-r_s*DT), 1 - s, s)*occ3
    return b, c, s


def pcollapse(kind, amps, reps, radius, seed, occ, mode, **kw):
    A = np.repeat(np.asarray(amps, float), reps)
    b, c, s = run(m8.make_sched(kind, A, **kw), len(A), radius, seed, occ, mode)
    Cf = c.sum(axis=(1, 2)).reshape(len(amps), reps)
    return (Cf == 0).mean(1)


def threshold_job(args):
    tag, dkind, size, radius, kind, seed = args
    occ, mode = domain(dkind, size)
    g1 = np.round(np.arange(0.07, 0.701, 0.03), 3)
    c = m8.a50(g1, pcollapse(kind, g1, 2, radius, seed, occ, mode))
    if not np.isfinite(c): return args, np.nan
    g2 = np.linspace(c - 0.045, c + 0.045, 10)
    return args, m8.a50(g2, pcollapse(kind, g2, 8, radius, seed + 1, occ, mode))


def pocket_mask(occ, side, where):
    t = np.ones_like(occ)
    if side == 0: return t
    n = occ.shape[0]
    if where == 'centre':
        i0 = (n - side)//2; t[i0:i0 + side, i0:i0 + side] = 0
    else:  # edge: the square touching the boundary of the occupied region on the left, mid-height
        rows = np.where(occ.any(1))[0]; mid = (rows[0] + rows[-1])//2
        i0 = mid - side//2
        j0 = np.where(occ[mid] > 0)[0][0]
        t[i0:i0 + side, j0:j0 + side] = 0
    return t


def pocket_job(args):
    tag, dkind, size, radius, where, sides, seed = args
    occ, mode = domain(dkind, size)
    out = []
    for side in sides:
        b, c, s = run(m8.make_sched('C', np.full(12, 0.6)), 12, radius, seed + side, occ, mode,
                      tmask=pocket_mask(occ, side, where))
        out.append((c.sum(axis=(1, 2)) == 0).mean())
    return args, out


def state_job(args):
    tag, dkind, size, radius = args
    occ, mode = domain(dkind, size)
    z = lambda t: (np.zeros(4), np.zeros(4))
    b, c, s = run(z, 4, radius, 11, occ, mode, t_end=150.0)
    n = occ.sum()
    k = 2*radius + 1
    den = np.maximum(uniform_filter(occ, size=k, mode=mode), 1e-9)[None]
    lm = lambda x: uniform_filter(x, size=(1, k, k), mode=mode)/den
    Fl = lm(c)*lm(s)*4*lm(b)*(1 - lm(b))
    return args, (b.sum()/(4*n), c.sum()/(4*n), s.sum()/(4*n), (Fl*occ).sum()/(4*n))


if __name__ == '__main__':
    t_start = time.time()
    GEOM = [('periodic square 48', 'periodic', 48), ('bounded square 48', 'square', 48), ('disc radius 27', 'disc', 27)]
    SIZE = [('disc radius 14', 'disc', 14), ('disc radius 27', 'disc', 27), ('disc radius 54', 'disc', 54),
            ('periodic square 24', 'periodic', 24), ('periodic square 48', 'periodic', 48), ('periodic square 96', 'periodic', 96)]
    SIDES = [0, 2, 3, 4, 5, 6, 8]
    thr_jobs = [(t, d, z, r, k, 500 + 10*i + r) for i, (t, d, z) in enumerate(GEOM) for r in [1, 2] for k in ['C', 'Cfull2']]
    thr_jobs += [(t, d, z, 2, 'C', 700 + 10*i) for i, (t, d, z) in enumerate(SIZE) if z not in (27, 48)]
    pk_jobs = [(t, d, z, r, 'centre', SIDES, 800 + r) for (t, d, z) in GEOM for r in [1, 2]]
    pk_jobs += [('disc radius 27', 'disc', 27, r, 'edge', SIDES, 850 + r) for r in [1, 2]]
    pk_jobs += [(t, d, z, 2, 'centre', SIDES, 870) for (t, d, z) in SIZE if z not in (27, 48)]
    st_jobs = [(t, d, z, r) for (t, d, z) in GEOM for r in [1, 2]] + [(t, d, z, 2) for (t, d, z) in SIZE if z not in (27, 48)]
    big = lambda j: -j[2]*(4 if j[1] == 'disc' else 1)**0  # larger domains first
    with Pool(os.cpu_count()) as pool:
        st = pool.map_async(state_job, st_jobs)
        pk = pool.map_async(pocket_job, sorted(pk_jobs, key=lambda j: -j[2]))
        th = pool.map(threshold_job, sorted(thr_jobs, key=lambda j: -j[2]))
        st = st.get(); pk = pk.get()
    TH = {(a[0], a[3], a[4]): v for a, v in th}
    PK = {(a[0], a[3], a[4]): v for a, v in pk}
    ST = {(a[0], a[3]): v for a, v in st}
    ncell = {t: int(domain(d, z)[0].sum()) for (t, d, z) in GEOM + SIZE}

    print('1. Geometry at constant area (cells: ' + ', '.join(f'{t} {ncell[t]}' for t, _, _ in GEOM) + ')')
    print('  State without therapy after 150 time units (D, C, S, mean local F):')
    for t, _, _ in GEOM:
        for r in [1, 2]: print(f'    {t:<20} r = {r}: ' + ' '.join(f'{x:.3f}' for x in ST[(t, r)]))
    print('  a50, coherence channel alone (window 5) / single-channel control at equal peak total amplitude (course 16):')
    for t, _, _ in GEOM:
        for r in [1, 2]: print(f'    {t:<20} r = {r}: {TH[(t, r, "C")]:.3f} / {TH[(t, r, "Cfull2")]:.3f}')
    print('  P(collapse) with an untreated square pocket (amplitude 0.6), sides', SIDES)
    for t, _, _ in GEOM:
        for r in [1, 2]: print(f'    {t:<20} r = {r}, centre: ' + ' '.join(f'{x:.2f}' for x in PK[(t, r, "centre")]))
    for r in [1, 2]: print(f'    {"disc radius 27":<20} r = {r}, edge:   ' + ' '.join(f'{x:.2f}' for x in PK[("disc radius 27", r, "edge")]))

    print('\n2. Size, r = 2 (cells: ' + ', '.join(f'{t} {ncell[t]}' for t, _, _ in SIZE) + ')')
    for t, _, _ in SIZE:
        print(f'    {t:<20} a50 {TH[(t, 2, "C")]:.3f}   pocket centre: ' + ' '.join(f'{x:.2f}' for x in PK[(t, 2, "centre")]))
    for t, _, z in SIZE:
        if z not in (27, 48): print(f'    {t:<20} state without therapy: ' + ' '.join(f'{x:.3f}' for x in ST[(t, 2)]))

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 11)); ax = ax.ravel()
    a0 = ax[0]; xg = np.arange(len(GEOM)); wb = 0.2
    for j, (r, k, lab) in enumerate([(1, 'C', '$T_C$ only, r = 1'), (2, 'C', '$T_C$ only, r = 2'),
                                     (1, 'Cfull2', 'control, peak total, r = 1'), (2, 'Cfull2', 'control, peak total, r = 2')]):
        a0.bar(xg + (j - 1.5)*wb, [TH[(t, r, k)] for t, _, _ in GEOM], wb, color=['C0', 'C9', 'C3', 'C1'][j], edgecolor='k', label=lab)
    a0.set_xticks(xg); a0.set_xticklabels([t.replace(' 48', '\n48 x 48').replace(' radius', '\nradius') for t, _, _ in GEOM])
    a0.set_ylim(0, 0.45); a0.set_ylabel('Critical intensity a50')
    a0.set_title('A. Thresholds at constant area'); a0.legend(fontsize=10, ncol=2, loc='upper center'); a0.grid(alpha=0.3, axis='y')
    a1 = ax[1]
    for (t, _, _), c_ in zip(GEOM, ['k', 'C2', 'C4']):
        for r, ls in [(1, ':'), (2, '-')]:
            a1.plot(SIDES, PK[(t, r, 'centre')], 'o' + ls, color=c_, label=f'{t}, r = {r}')
    for r, ls in [(1, ':'), (2, '-')]:
        a1.plot(SIDES, PK[('disc radius 27', r, 'edge')], 's' + ls, color='C1', label=f'disc, edge pocket, r = {r}')
    a1.set_xlabel('Side of the untreated square (cells)'); a1.set_ylabel('P(collapse)')
    a1.set_title('B. Untreated pocket at constant area, amplitude 0.6'); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2 = ax[2]
    for kind, c_ in [('disc', 'C4'), ('periodic', 'k')]:
        rows = [(ncell[t], t) for (t, d, _) in SIZE if d == kind]
        a2.plot([n for n, _ in rows], [TH[(t, 2, 'C')] for _, t in rows], 'o-', color=c_, label=f'{kind}')
    a2.set_xscale('log'); a2.set_ylim(0, 0.45); a2.set_xlabel('Number of cells'); a2.set_ylabel('Critical intensity a50 ($T_C$ only, r = 2)')
    a2.set_title('C. Threshold against tumor size'); a2.legend(fontsize=10); a2.grid(alpha=0.3)
    a3 = ax[3]
    for (t, d, z), c_ in zip(SIZE, ['C4', 'C4', 'C4', 'k', 'k', 'k']):
        lw = {14: 1, 27: 2, 54: 3, 24: 1, 48: 2, 96: 3}[z]
        a3.plot(SIDES, PK[(t, 2, 'centre')], 'o-' if d == 'disc' else 's--', color=c_, lw=lw, label=f'{t} ({ncell[t]} cells)')
    a3.set_xlabel('Side of the untreated square (cells)'); a3.set_ylabel('P(collapse)')
    a3.set_title('D. Untreated pocket against tumor size, r = 2'); a3.legend(fontsize=9); a3.grid(alpha=0.3)
    import json
    (HERE.parent / 'logs' / 'figS8_results.json').write_text(json.dumps({'thresholds': {'|'.join(map(str, k)): v for k, v in TH.items()},
        'pockets': {'|'.join(map(str, k)): v for k, v in PK.items()}, 'states': {'|'.join(map(str, k)): [float(x) for x in v] for k, v in ST.items()}}, indent=1, default=float))
    plt.tight_layout(); plt.savefig(FIGDIR / 'FigS8.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved FigS8.png   ({(time.time() - t_start)/60:.1f} min)')
