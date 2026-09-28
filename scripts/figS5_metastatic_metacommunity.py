"""
Metastatic meta-community: n sites, each carrying its own GPC (D_k, C_k, S_k) and a
tumour burden N_k, coupled by cell migration.

Within a site, the GPC equations are those of fig1 and fig3 (same parameters), with
the coherence channel T_C applied at a site-specific exposure a_k (heterogeneous drug
penetration, a_k = abar * LogNormal(0, sd)).

Burden: dN_k = N_k [ r0 (1 + omega F_k)(1 - N_k/K) - d ] - m N_k + m/(n-1) sum_{j!=k} N_j
  ASSUMPTION (not in fig1 to fig6): d is chosen between r0 and r0(1 + omega F_func), so a
  site whose GPC has collapsed (F = 0) declines and a functional site persists. Without this
  assumption GPC collapse slows growth but never eliminates a site, and "persistence" is
  undefined. d is a free parameter.

Migration also carries composition: immigrants bring the phenotypic diversity D and
cooperative engagement C of their source sites (mass-weighted), so C_k and D_k relax
toward the immigrant mean at rate inflow/N_k. S_k (local microenvironment) is not
transported. Variant 'burden only' switches composition transport off.

A site whose burden falls below 1e-4 is extinct (set to 0; it can be recolonised by immigrants).
Persistence = at least one site not extinct at t_end = 600. A collapsed site with no inflow
reaches the floor well before t_end; in the meantime GPC noise (sigma = 0.01, as in scripts
figS2/fig3) can return it to the functional state, which is part of the model, not an artefact.
Runtime: about 20 min on one core.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, matplotlib.pyplot as plt, time

P = dict(mu0=0.8, mu1=0.05, delta_D=0.6, alpha=1.2, beta=0.8, delta_C=0.15, gamma=1.0, kappa=0.6, delta_S=0.15,
         r0=0.10, omega=1.0, d=0.12, K=1.0)
IC = (0.753, 0.801, 0.842)
DT, T0, W, T_END, NFLOOR = 0.05, 10.0, 10.0, 600.0, 1e-4

def n_star(p=P):
    F = IC[1]*IC[2]*4*IC[0]*(1-IC[0])
    return p['K']*(1 - p['d']/(p['r0']*(1 + p['omega']*F)))

def simulate(n, abar, sd, m, reps, seed=0, sigma=0.01, transport=True, p=P, record=False, a_sites=None):
    """reps independent metastatic systems of n sites each. Returns persistence (bool, reps) and extras."""
    rng = np.random.default_rng(seed)
    a = abar*np.exp(sd*rng.standard_normal((reps, n)) - 0.5*sd**2) if a_sites is None else np.broadcast_to(a_sites, (reps, n)).astype(float)
    D = np.full((reps, n), IC[0]); C = np.full((reps, n), IC[1]); S = np.full((reps, n), IC[2])
    N = np.full((reps, n), n_star(p))
    rec = []
    for i in range(int(T_END/DT)):
        t = i*DT
        Tc = a if T0 <= t < T0 + W else 0.0
        F = C*S*4*D*(1-D)
        dD = p['mu0']*F + p['mu1'] - p['delta_D']*D
        dC = p['alpha']*F*(1-C) - p['beta']*Tc - p['delta_C']*C
        dS = p['gamma']*C*(1-S) - p['delta_S']*S
        dN = N*(p['r0']*(1 + p['omega']*F)*(1 - N/p['K']) - p['d'])
        noise = sigma*np.sqrt(DT)*rng.standard_normal((3, reps, n)) if sigma else 0.0
        D = D + dD*DT + (noise[0] if sigma else 0); C = C + dC*DT + (noise[1] if sigma else 0)
        S = S + dS*DT + (noise[2] if sigma else 0)
        N = N + dN*DT
        if n > 1 and m > 0:
            tot = N.sum(1, keepdims=True)
            inflow = m*(tot - N)/(n - 1)                       # immigrants per unit time
            if transport:
                Cin = ((N*C).sum(1, keepdims=True) - N*C)/np.maximum(tot - N, 1e-12)
                Din = ((N*D).sum(1, keepdims=True) - N*D)/np.maximum(tot - N, 1e-12)
                f = -np.expm1(-inflow*DT/np.maximum(N, 1e-12))  # share of residents replaced this step
                C = C + f*(Cin - C); D = D + f*(Din - D)
            N = N + (inflow - m*N)*DT
        D = np.clip(D, 0, 1); C = np.clip(C, 0, 1); S = np.clip(S, 0, 1)
        N = np.where(N < NFLOOR, 0.0, N)
        if record and i % 20 == 0: rec.append((t, N.copy(), (C*S*4*D*(1-D)).copy()))
    persist = (N > 0).any(1)
    Ffinal = C*S*4*D*(1-D)
    return dict(persist=persist, a=a, Ffinal=Ffinal, N=N, rec=rec)

def isolated_threshold(p=P, lo=0.05, hi=1.0, tol=1e-3):
    """Exposure above which a single isolated site collapses and goes extinct (deterministic)."""
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if simulate(1, 0, 0, 0, 1, sigma=0, a_sites=np.array([[mid]]), p=p)['persist'][0]: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)

def draw_a(abars, reps, n, sd, rng):
    base = np.repeat(np.asarray(abars, float), reps)[:, None]
    return base*np.exp(sd*rng.standard_normal((len(base), n)) - 0.5*sd**2)

def a50_curve(x, p):
    """exposure at which persistence falls to 0.5 (linear interpolation on a decreasing curve)."""
    k = np.where(p <= 0.5)[0]
    if len(k) == 0: return np.inf
    k = k[0]
    if k == 0: return x[0]
    return x[k-1] + (p[k-1] - 0.5)*(x[k] - x[k-1])/(p[k-1] - p[k])

def auc(score, y):
    """P(score of a persisting system > score of an extinct one), ties counted 1/2."""
    y = np.asarray(y, bool); s1, s0 = score[y], score[~y]
    if len(s1) == 0 or len(s0) == 0: return np.nan
    from scipy.stats import rankdata
    rk = rankdata(np.r_[s1, s0])
    return (rk[:len(s1)].sum() - len(s1)*(len(s1) + 1)/2)/(len(s1)*len(s0))

if __name__ == '__main__':
    t_start = time.time()
    a_star = isolated_threshold()
    print(f'Functional burden per site N* = {n_star():.3f}; isolated-site collapse threshold a* = {a_star:.3f} (w = {W:g})')
    NS = [1, 2, 4, 8, 16, 32]; MS = [0.0, 0.003, 0.01, 0.03, 0.1]
    REL = np.array([0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]); REPS = 80; SD = 0.3
    abars = REL*a_star
    rng = np.random.default_rng(12)

    # 1. persistence vs number of sites and mean exposure, for several migration rates
    Pm = {}; pooled = {}
    for m in MS:
        Pm[m] = np.zeros((len(NS), len(REL)))
        for i, n in enumerate(NS):
            a = draw_a(abars, REPS, n, SD, rng)
            r = simulate(n, 0, 0, m, len(a), seed=1000*i + int(1e4*m), a_sites=a)
            Pm[m][i] = r['persist'].reshape(len(REL), REPS).mean(1)
            pooled.setdefault(m, []).append((n, a, r['persist'], r['Ffinal']))
    print(f'\nExposure giving 50% persistence, in units of a* (heterogeneity sd = {SD}):')
    print('  m \\ n     ' + ''.join(f'{n:>7}' for n in NS))
    A50 = {}
    for m in MS:
        A50[m] = np.array([a50_curve(REL, Pm[m][i]) for i in range(len(NS))])
        print(f'  {m:<9}' + ''.join(f'{x:>7.2f}' for x in A50[m]))

    # 2. what predicts persistence: n, the least-exposed site, or the mean exposure?
    print('\nPredictive power (AUC) for persistence, all n and exposures pooled:')
    print(f'  {"m":<7}{"-log n":>9}{"least exposure":>16}{"mean exposure":>15}{"share of sites below a*":>26}')
    AUC = {}
    for m in MS:
        n_all = np.concatenate([np.full(len(a), n) for n, a, _, _ in pooled[m]])
        amin = np.concatenate([a.min(1) for _, a, _, _ in pooled[m]])
        amean = np.concatenate([a.mean(1) for _, a, _, _ in pooled[m]])
        fbelow = np.concatenate([(a < a_star).mean(1) for _, a, _, _ in pooled[m]])
        y = np.concatenate([p for _, _, p, _ in pooled[m]])
        AUC[m] = (auc(np.log(n_all), y), auc(-amin, y), auc(-amean, y), auc(fbelow, y))
        print(f'  {m:<7}' + ''.join(f'{v:>{w}.3f}' for v, w in zip(AUC[m], [9, 16, 15, 26])))

    # rescue and drag-down, per migration rate
    print('\nFate of sites at t_end relative to their fate in isolation (all runs pooled):')
    for m in MS:
        res = drag = nres = ndrag = 0
        for n, a, p, Ff in pooled[m]:
            if n == 1: continue
            func = Ff > 0.05; would_collapse = a > a_star
            res += (func & would_collapse).sum(); nres += would_collapse.sum()
            drag += (~func & ~would_collapse).sum(); ndrag += (~would_collapse).sum()
        print(f'  m = {m:<6} rescued: {res/max(nres,1):.3f} of sites exposed above a*;  dragged down: {drag/max(ndrag,1):.3f} of sites exposed below a*')

    # 3. heterogeneity among sites: is the n effect intrinsic?
    SDS = [0.0, 0.15, 0.3, 0.5]
    print('\nExposure for 50% persistence (units of a*) vs heterogeneity sd, m = 0 and m = 0.01:')
    A50sd = {}
    for m in [0.0, 0.01]:
        for sd in SDS:
            row = []
            for i, n in enumerate(NS):
                a = draw_a(abars, REPS, n, sd, rng)
                r = simulate(n, 0, 0, m, len(a), seed=77 + i, a_sites=a)
                row.append(a50_curve(REL, r['persist'].reshape(len(REL), REPS).mean(1)))
            A50sd[(m, sd)] = np.array(row)
            print(f'  m = {m:<5} sd = {sd:<5}' + ''.join(f'{x:>7.2f}' for x in row))

    # 4. composition transport off: demographic drain alone
    print('\nComposition transport switched off (migration moves burden only), exposure for 50% persistence:')
    A50nt = {}
    for m in [0.01, 0.03, 0.1]:
        row = []
        for i, n in enumerate(NS):
            a = draw_a(abars, REPS, n, SD, rng)
            r = simulate(n, 0, 0, m, len(a), seed=55 + i, a_sites=a, transport=False)
            row.append(a50_curve(REL, r['persist'].reshape(len(REL), REPS).mean(1)))
        A50nt[m] = np.array(row)
        print(f'  m = {m:<6}' + ''.join(f'{x:>7.2f}' for x in row))

    # 5. without GPC noise (outcome fixed by the exposures alone)
    row_nonoise = []
    for i, n in enumerate(NS):
        a = draw_a(abars, REPS, n, SD, rng)
        r = simulate(n, 0, 0, 0.01, len(a), seed=33 + i, a_sites=a, sigma=0.0)
        row_nonoise.append(a50_curve(REL, r['persist'].reshape(len(REL), REPS).mean(1)))
    print('\nNo GPC noise (sigma = 0), m = 0.01, sd = 0.3:' + ''.join(f'{x:>7.2f}' for x in row_nonoise))

    # 6. identical exposure at every site (no heterogeneity): relapse opportunities add up
    rel_hi = 3.0
    r1 = simulate(1, 0, 0, 0.0, 2000, seed=501, a_sites=np.full((2000, 1), rel_hi*a_star))
    p1 = r1['persist'].mean()
    print(f'\nIdentical exposure {rel_hi:g} a* at every site. Isolated site: P(persist) = {p1:.3f} (noise-driven return before extinction)')
    Phom = {}
    for m in [0.0, 0.01]:
        Phom[m] = []
        for i, n in enumerate(NS):
            r = simulate(n, 0, 0, m, 200, seed=600 + i, a_sites=np.full((200, n), rel_hi*a_star))
            Phom[m].append(r['persist'].mean())
        print(f'  m = {m:<5} P(persist) vs n: ' + '  '.join(f'{n}: {p:.2f}' for n, p in zip(NS, Phom[m])))
    print('  independent-sites prediction 1-(1-p1)^n: ' + '  '.join(f'{n}: {1-(1-p1)**n:.2f}' for n in NS))

    # ---- figure
    cap = lambda v: np.where(np.isfinite(v), v, np.nan)
    cnt = {}
    def mark_inf(ax, v, c):
        for n, x in zip(NS, v):
            if not np.isfinite(x):
                k = cnt.get((id(ax), n), 0); cnt[(id(ax), n)] = k + 1
                ax.plot(n, 3.1 - 0.07*k, 'x', color=c, ms=8, mew=2)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    a0 = ax[0, 0]
    for m, c in zip(MS, ['k', 'C0', 'C2', 'C1', 'C3']):
        a0.plot(NS, cap(A50[m]), 'o-', color=c, label=f'm = {m:g}'); mark_inf(a0, A50[m], c)
    a0.set_xscale('log', base=2); a0.set_xlabel('Number of metastatic sites n'); a0.set_ylabel('Exposure for 50% eradication (x a*)')
    a0.set_ylim(0.9, 3.2); a0.set_title(f'A. Dose needed vs number of sites (sd = {SD})\nx : not reached at 3 a*', fontsize=10)
    a0.legend(fontsize=8); a0.grid(alpha=0.3)
    a1 = ax[0, 1]
    xw = np.arange(4); wb = 0.16
    for j, (m, c) in enumerate(zip(MS, ['k', 'C0', 'C2', 'C1', 'C3'])):
        a1.bar(xw + (j - 2)*wb, AUC[m], wb, color=c, label=f'm = {m:g}')
    a1.axhline(0.5, color='k', ls=':'); a1.set_xticks(xw)
    a1.set_xticklabels(['number of sites', 'least-exposed site', 'mean exposure', 'share of sites < a*'], fontsize=8)
    a1.set_ylabel('AUC for persistence'); a1.set_ylim(0.4, 1.0)
    a1.set_title('B. What predicts persistence (all runs pooled)', fontsize=10); a1.legend(fontsize=8); a1.grid(alpha=0.3, axis='y')
    a2 = ax[0, 2]
    for sd, c in zip(SDS, ['k', 'C9', 'C0', 'C4']):
        a2.plot(NS, cap(A50sd[(0.0, sd)]), 'o-', color=c, label=f'sd = {sd}, m = 0'); mark_inf(a2, A50sd[(0.0, sd)], c)
        a2.plot(NS, cap(A50sd[(0.01, sd)]), 's--', color=c, alpha=0.6, label=f'sd = {sd}, m = 0.01'); mark_inf(a2, A50sd[(0.01, sd)], c)
    a2.set_xscale('log', base=2); a2.set_xlabel('n'); a2.set_ylabel('Exposure for 50% eradication (x a*)'); a2.set_ylim(0.9, 3.2)
    a2.set_title('C. Heterogeneity among sites amplifies the n effect\nx : not reached at 3 a*', fontsize=10)
    a2.legend(fontsize=7, ncol=2); a2.grid(alpha=0.3)
    a3 = ax[1, 0]
    for m, c in zip([0.01, 0.03, 0.1], ['C2', 'C1', 'C3']):
        a3.plot(NS, cap(A50[m]), 'o-', color=c, label=f'm = {m:g}, burden + composition')
        a3.plot(NS, cap(A50nt[m]), 's--', color=c, label=f'm = {m:g}, burden only'); mark_inf(a3, A50nt[m], c)
    a3.plot(NS, cap(A50[0.0]), 'k-', lw=1, label='m = 0')
    a3.set_xscale('log', base=2); a3.set_xlabel('n'); a3.set_ylabel('Exposure for 50% eradication (x a*)'); a3.set_ylim(0.9, 3.2)
    a3.set_title('D. At m <= 0.03 the flattening needs composition transport\n(burden-only migration keeps the n effect)', fontsize=10); a3.legend(fontsize=7); a3.grid(alpha=0.3)
    a4 = ax[1, 1]
    for m, c in zip([0.0, 0.01], ['k', 'C2']):
        a4.plot(NS, Phom[m], 'o-', color=c, label=f'simulated, m = {m:g}')
    a4.plot(NS, [1 - (1 - p1)**n for n in NS], 'k:', label='1 - (1 - p1)^n, independent sites')
    a4.set_xscale('log', base=2); a4.set_xlabel('n'); a4.set_ylabel('P(persistence)'); a4.set_ylim(0, 1)
    a4.set_title(f'E. Identical exposure {rel_hi:g} a* at every site:\nrelapse chances add up with n', fontsize=10)
    a4.legend(fontsize=8); a4.grid(alpha=0.3)
    a5 = ax[1, 2]
    a5.plot(NS, cap(A50sd[(0.01, 0.3)]), 'o-', color='C2', label='GPC noise sigma = 0.01')
    a5.plot(NS, cap(np.array(row_nonoise)), 's--', color='C2', label='no GPC noise')
    a5.set_xscale('log', base=2); a5.set_xlabel('n'); a5.set_ylabel('Exposure for 50% eradication (x a*)')
    a5.set_title('F. m = 0.01, sd = 0.3: with and without GPC noise', fontsize=10); a5.legend(fontsize=8); a5.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIGDIR / 'FigS5.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved FigS5.png   ({(time.time() - t_start)/60:.1f} min)')
