"""
Where does adaptive therapy stop beating continuous therapy in the (phi, omega) plane?

Clonal-GPC model of fig6_clonal_coupling.py, unchanged (same parameters, same Euler scheme, same
schedules, same progression criterion). The only change is that the model is
vectorised over a grid of (phi, omega) so the plane can be mapped at fine resolution.

Outputs
  - TTP ratio adaptive/continuous on a 51 x 65 grid, phi in [0,1], omega in [0,16]
  - the contour ratio = 1 and the boundary where continuous therapy collapses the GPC
  - cross-sections along omega (at fixed phi) and along phi (at fixed omega)
  - sensitivity of the boundary locations to the parameters fig6_clonal_coupling.py fixes by hand
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt, time
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

BASE = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.12, delta_C=0.15, gamma=1.0, kappa=0.6,
            delta_S=0.15, r0=0.10, cost=0.30, K=1.0, kill=2.0, mut=2e-5)
Y0 = (0.753, 0.801, 0.842, 0.85, 1e-4)
DT = 0.05

def run_vec(phi, omega, mode, p=BASE, y0=Y0, t_end=1200.0, thresh=0.90, on_frac=1.0, off_frac=0.85):
    """Vectorised copy of fig6_clonal_coupling.py run(). Returns TTP, min F before progression, cumulative dose."""
    phi = np.asarray(phi, float); omega = np.asarray(omega, float); shp = phi.shape
    phi = phi.ravel(); omega = omega.ravel(); n = phi.size
    D, C, S, Ns, Nr = [np.full(n, v, float) for v in y0]
    ttp = np.full(n, np.nan); Fmin = np.full(n, 9.0); dose = np.zeros(n)
    on = np.ones(n, bool); ref = Ns + Nr
    for i in range(int(t_end/DT)):
        t = i*DT
        if mode == 'cont':
            Tc = np.ones(n)
        else:
            b = Ns + Nr
            off_now = on & (b < off_frac*ref); on_now = (~on) & (b > on_frac*ref)
            on = (on & ~off_now) | on_now
            Tc = on.astype(float)
        N = Ns + Nr; rho = np.where(N > 1e-12, Nr/np.maximum(N, 1e-300), 0.0)
        Cmax = np.maximum(1 - phi*rho, 0.0)
        F = C*S*4*D*(1-D)
        growth = p['r0']*(1 + omega*F)*(1 - N/p['K'])
        dNs = growth*Ns - p['kill']*Tc*Ns - p['mut']*Ns
        dNr = growth*(1-p['cost'])*Nr + p['mut']*Ns
        dD = p['mu0']*F + p['mu1'] - p['delta_D']*D
        dC = p['alpha']*F*(Cmax - C) - p['beta']*Tc - p['delta_C']*C
        dS = p['gamma']*C*(1-S) - p['delta_S']*S
        D = np.clip(D + dD*DT, 0, 1); C = np.clip(C + dC*DT, 0, 1); S = np.clip(S + dS*DT, 0, 1)
        Ns = np.clip(Ns + dNs*DT, 0, None); Nr = np.clip(Nr + dNr*DT, 0, None)
        dose += Tc*DT
        hit = np.isnan(ttp) & ((Ns + Nr) > thresh); ttp[hit] = t
        Fn = C*S*4*D*(1-D)
        Fmin = np.where(np.isnan(ttp) | hit, np.minimum(Fmin, Fn), Fmin)
    return dict(ttp=ttp.reshape(shp), Fmin=Fmin.reshape(shp), dose=dose.reshape(shp))

def ratio_map(ph, om, **kw):
    PH, OM = np.meshgrid(ph, om, indexing='ij')
    c = run_vec(PH, OM, 'cont', **kw); a = run_vec(PH, OM, 'adapt', **kw)
    return a['ttp']/c['ttp'], c, a

def first_cross(x, r):
    """First x where r drops below 1 (linear interpolation); inf if never."""
    k = np.where(r < 1)[0]
    if len(k) == 0: return np.inf
    k = k[0]
    if k == 0: return x[0]
    return x[k-1] + (r[k-1]-1)*(x[k]-x[k-1])/(r[k-1]-r[k])

if __name__ == '__main__':
    t_start = time.time()
    # 0. reproduction check against fig6_clonal_coupling.py
    chk = ratio_map(np.array([0.0, 0.45, 0.9]), np.array([0.0, 0.5, 1.0, 2.0]))[0]
    print('Reproduction of fig6_clonal_coupling.py grid (phi rows 0, 0.45, 0.9; omega 0, 0.5, 1, 2):')
    print(np.round(chk, 2))

    F_func = Y0[1]*Y0[2]*4*Y0[0]*(1-Y0[0])
    print(f'F at the initial (functional) GPC state = {F_func:.3f}; growth multiplier 1 + omega*F = {1+6*F_func:.2f} at omega = 6')

    # 1. main map
    ph = np.linspace(0, 1, 51); om = np.linspace(0, 16, 65)
    R, c, a = ratio_map(ph, om)
    coll = c['Fmin'] < 0.05          # continuous therapy tips the GPC into the collapsed state
    print(f'\nMap {len(ph)}x{len(om)}: fraction of the plane with ratio < 1: {(R < 1).mean():.2f}')
    print('Ratio at phi = 0 (no clone->GPC feedback), omega = 0..16: min %.2f, max %.2f' % (R[0].min(), R[0].max()))

    # collapse boundary phi*(omega) in the continuous arm
    phistar = np.array([ph[np.argmax(coll[:, j])] if coll[:, j].any() else np.nan for j in range(len(om))])
    print('phi* (continuous arm collapses the GPC) over omega in [0,16]: %.2f to %.2f' % (np.nanmin(phistar), np.nanmax(phistar)))

    # crossing omega*(phi)
    omstar = np.array([first_cross(om, R[i]) for i in range(len(ph))])
    for v in [0.0, 0.1, 0.2, 0.5, 0.9, 1.0]:
        i = np.argmin(abs(ph - v)); print(f'  omega* at phi = {ph[i]:.2f}: {omstar[i]:.2f}')

    # sharpness: jump across phi* vs slope along omega
    j6 = np.argmin(abs(om - 6)); i_s = np.argmax(coll[:, j6])
    print(f'\nAlong phi at omega = {om[j6]:.1f}: ratio {R[i_s-1, j6]:.2f} at phi = {ph[i_s-1]:.2f}  ->  {R[i_s, j6]:.2f} at phi = {ph[i_s]:.2f}  (one grid step, 0.02)')
    i5 = np.argmin(abs(ph - 0.5)); r5 = R[i5]
    w_hi = first_cross(om, r5/1.2); w_lo = first_cross(om, r5/0.8)
    print(f'Along omega at phi = 0.5: ratio goes from 1.2 to 0.8 between omega = {w_hi:.2f} and {w_lo:.2f} (width {w_lo-w_hi:.2f})')
    dRdom = np.abs(np.diff(R, axis=1))/(om[1]-om[0]); dRdph = np.abs(np.diff(R, axis=0))/(ph[1]-ph[0])
    print(f'max |dR/domega| = {dRdom.max():.2f} per unit omega; max |dR/dphi| = {dRdph.max():.1f} per unit phi')

    # beyond omega = 16
    ext_om = np.array([16, 20, 24, 28, 32.])
    Rx = ratio_map(np.array([0.2, 0.5, 0.9]), ext_om)[0]
    print('\nRatio for omega = 16..32 (rows phi = 0.2, 0.5, 0.9):'); print(np.round(Rx, 2))

    # mechanism: TTP of each arm vs omega at phi = 0 and phi = 0.5
    i0 = 0
    print('\nTTP continuous / adaptive at omega = 0, 4, 8, 16:')
    for i in [i0, i5]:
        js = [np.argmin(abs(om - w)) for w in [0, 4, 8, 16]]
        print(f'  phi = {ph[i]:.1f}: cont', np.round(c['ttp'][i, js], 1), ' adapt', np.round(a['ttp'][i, js], 1))

    # 2. sensitivity of the boundary locations
    sph = np.r_[np.arange(0, 0.301, 0.02), 0.5, 0.9]; som = np.arange(0, 20.01, 0.5)
    variants = [('reference (fig6_clonal_coupling.py)', {}, {}), ('beta = 0.08', {'beta': 0.08}, {}), ('beta = 0.16', {'beta': 0.16}, {}),
                ('cost = 0.2', {'cost': 0.2}, {}), ('cost = 0.4', {'cost': 0.4}, {}),
                ('kill = 1', {'kill': 1.0}, {}), ('kill = 3', {'kill': 3.0}, {}), ('r0 = 0.05', {'r0': 0.05}, {}),
                ('burden at diagnosis 0.5', {}, {'y0': (0.753, 0.801, 0.842, 0.5, 1e-4)}),
                ('adaptive pause at 50%', {}, {'off_frac': 0.5})]
    sens = []
    print('\nSensitivity (phi* at omega = 6; omega* at phi = 0.5 and 0.9; ratio at phi = 0.9, omega = 2):')
    for name, pp, kw in variants:
        Rs, cs, _ = ratio_map(sph, som, p=dict(BASE, **pp), **kw)
        cl = cs['Fmin'] < 0.05; j = np.argmin(abs(som - 6))
        ps = sph[np.argmax(cl[:, j])] if cl[:, j].any() else np.nan
        o5 = first_cross(som, Rs[list(sph).index(0.5)]); o9 = first_cross(som, Rs[list(sph).index(0.9)])
        r92 = Rs[list(sph).index(0.9), list(som).index(2.0)]
        # shape of the boundary: jump across phi* (one grid step, 0.02) at omega = 6, and the largest
        # change per unit omega above phi* (phi = 0.5), to check step-in-phi / gradual-in-omega
        ip = int(np.argmax(cl[:, j])) if cl[:, j].any() else None
        jump = (Rs[ip-1, j] - Rs[ip, j]) if (ip is not None and ip > 0) else np.nan
        slope = np.abs(np.diff(Rs[list(sph).index(0.5)]))/np.diff(som)
        sens.append((name, Rs[0, 0], ps, o5, o9, r92))
        print(f'  {name:<26} R(no coupling) {Rs[0,0]:.2f}  phi* {ps:.2f}  omega*(0.5) {o5:.2f}  omega*(0.9) {o9:.2f}  R(0.9,2) {r92:.2f}'
              f'  jump at phi* {jump:.2f}  max|dR/domega| {slope.max():.2f}')

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    a0 = ax[0, 0]
    im = a0.pcolormesh(om, ph, np.log2(R), cmap='RdBu', vmin=-1.6, vmax=1.6, shading='nearest')
    a0.contour(om, ph, R, levels=[1.0], colors='k', linewidths=2)
    a0.plot(om, phistar, 'y--', lw=2, label='continuous therapy collapses the GPC (phi*)')
    a0.plot([om[0]], [ph[0]], 'k-', lw=2, label='ratio = 1')
    a0.set_xlabel('Cooperative growth benefit omega'); a0.set_ylabel('Coherence cap by resistant cells phi')
    a0.set_title('A. log2 TTP ratio (adaptive / continuous)')
    cb = fig.colorbar(im, ax=a0); cb.set_label('log2 ratio  (>0: adaptive better)')
    a0.legend(fontsize=11, loc='upper right')

    a1 = ax[0, 1]
    for v, col in [(0.0, 'C7'), (0.06, 'C8'), (0.14, 'C1'), (0.3, 'C2'), (0.5, 'C0'), (0.9, 'C4')]:
        i = np.argmin(abs(ph - v)); a1.plot(om, R[i], color=col, lw=1.8, label=f'phi = {ph[i]:.2f}')
    a1.axhline(1, color='k', ls=':'); a1.set_xlabel('omega'); a1.set_ylabel('TTP ratio')
    a1.set_title('B. Along omega: gradual decline through ratio = 1'); a1.legend(fontsize=11); a1.grid(alpha=0.3)

    a2 = ax[1, 0]
    for v, col in [(1, 'C7'), (2, 'C1'), (6, 'C0'), (12, 'C4')]:
        j = np.argmin(abs(om - v)); a2.plot(ph, R[:, j], 'o-', ms=2.5, color=col, lw=1.5, label=f'omega = {om[j]:.0f}')
    a2.axhline(1, color='k', ls=':'); a2.set_xlabel('phi'); a2.set_ylabel('TTP ratio')
    a2.set_title('C. Along phi: a step where continuous therapy tips the GPC'); a2.legend(fontsize=11); a2.grid(alpha=0.3)

    a3 = ax[1, 1]
    for i, ls in [(0, '--'), (i5, '-')]:
        a3.plot(om, c['ttp'][i], color='C3', ls=ls, lw=1.8, label=f'continuous, phi = {ph[i]:.1f}')
        a3.plot(om, a['ttp'][i], color='C0', ls=ls, lw=1.8, label=f'adaptive, phi = {ph[i]:.1f}')
    a3.set_xlabel('omega'); a3.set_ylabel('Time to progression')
    a3.set_title('D. Mechanism: with phi > phi*, continuous TTP is insensitive to omega')
    a3.legend(fontsize=11); a3.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIGDIR / 'Fig7.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved Fig7.png   ({time.time()-t_start:.0f} s)')
