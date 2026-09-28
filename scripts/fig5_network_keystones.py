"""
Thomas framing note, Hypothesis 4: critical subpopulations act as tipping-point
nodes. Synthetic phenotype-interaction networks (no patient data).

Sustained removal of a node (phenotype) lowers the maximal coherence the
collective can reach: dC/dt = alpha*F*(Cmax - C) - ..., with Cmax = E_f / E_0,
E = global efficiency of the interaction network after the node's links are cut.
We find the critical Cmax* below which the functional GPC attractor disappears,
then ask how many nodes must be removed (targeted by degree vs random) to cross it.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
import numpy as np, networkx as nx, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import importlib.util
spec = importlib.util.spec_from_file_location('m', HERE / 'fig3_sequential_vs_simultaneous.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P, IC = m.P, m.IC

def collapses_with_cmax(cmax, t_end=300.0, dt=0.05):
    X = IC.copy()
    for _ in range(int(t_end/dt)):
        D, C, S = X
        Fv = C*S*4*D*(1-D)
        dX = np.array([P['mu0']*Fv + P['mu1'] - P['delta_D']*D,
                       P['alpha']*Fv*(cmax - C) - P['delta_C']*C,
                       P['gamma']*C*(1-S) - P['delta_S']*S])
        X = np.clip(X + dX*dt, 0, 1)
    return X[1]*X[2]*4*X[0]*(1-X[0]) < 0.05

lo, hi = 0.0, 1.0
while hi - lo > 1e-3:
    mid = 0.5*(lo+hi)
    if collapses_with_cmax(mid): lo = mid
    else: hi = mid
CMAX_STAR = hi
print(f'Critical maximal coherence Cmax* = {CMAX_STAR:.3f}')

def efficiency_curve(G, strategy, rng):
    H = G.copy(); E0 = nx.global_efficiency(G); n = G.number_of_nodes()
    alive = set(G.nodes()); out = [1.0]
    order = list(rng.permutation(list(G.nodes()))) if strategy == 'random' else None
    for k in range(n - 1):
        if strategy == 'targeted':
            v = max(alive, key=lambda x: H.degree(x))   # adaptive: highest current degree
        else:
            v = order[k]
        H.remove_edges_from(list(H.edges(v))); alive.discard(v)
        out.append(nx.global_efficiency(H)/E0)
    return np.array(out)

def critical_count(curve):
    below = np.where(curve < CMAX_STAR)[0]
    return below[0] if len(below) else np.nan

N, REPS = 40, 100
rng = np.random.default_rng(0)
res = {('ER', s): [] for s in ['targeted', 'random']}
res.update({('BA', s): [] for s in ['targeted', 'random']})
crit, phi = {k: [] for k in res}, {'ER': [], 'BA': []}
for r in range(REPS):
    nets = {'ER': nx.gnp_random_graph(N, 4/(N-1), seed=int(rng.integers(1e9))),
            'BA': nx.barabasi_albert_graph(N, 2, seed=int(rng.integers(1e9)))}
    for name, G in nets.items():
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)
        lam = max(np.linalg.eigvalsh(nx.to_numpy_array(G)))
        phi[name].append(lam / np.mean([d for _, d in G.degree()]))
        for s in ['targeted', 'random']:
            curve = efficiency_curve(G, s, rng)
            res[(name, s)].append(curve[:N])
            crit[(name, s)].append(critical_count(curve) / G.number_of_nodes())

for k in res:
    c = np.array(crit[k])
    print(f'{k[0]} {k[1]:<9} critical fraction removed: median {np.nanmedian(c):.3f} '
          f'(IQR {np.nanpercentile(c,25):.3f}-{np.nanpercentile(c,75):.3f})')
for name in phi:
    print(f'{name} fragility index Phi: median {np.median(phi[name]):.2f}')
allphi = np.array(phi['ER'] + phi['BA'])
allcrit = np.array(crit[('ER', 'targeted')] + crit[('BA', 'targeted')])
ok = np.isfinite(allcrit)
rho, pv = spearmanr(allphi[ok], allcrit[ok])
print(f'Spearman(Phi, critical targeted fraction) = {rho:.2f}, p = {pv:.1e}')
rows = []
for name in ['ER', 'BA']:
    a, b = np.array(phi[name]), np.array(crit[(name, 'targeted')]); o = np.isfinite(b)
    rw, pw = spearmanr(a[o], b[o]); rows.append((name, rw, pw))
    print(f'  within {name}: rho = {rw:.2f}, p = {pw:.2g}')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
colors = {('ER', 'targeted'): 'C0', ('ER', 'random'): 'C9', ('BA', 'targeted'): 'C3', ('BA', 'random'): 'C1'}
x = np.arange(N)/N
for k, curves in res.items():
    L = min(len(c) for c in curves); arr = np.array([c[:L] for c in curves])
    ax[0].plot(x[:L], arr.mean(0), color=colors[k], lw=2,
               label=f"{'Hub-dominated (BA)' if k[0]=='BA' else 'Homogeneous (ER)'}, {k[1]}")
    ax[0].fill_between(x[:L], arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0), color=colors[k], alpha=0.15)
ax[0].axhline(CMAX_STAR, color='k', ls='--', label=f'Tipping threshold Cmax* = {CMAX_STAR:.2f}')
ax[0].set_xlabel('Fraction of phenotypes removed'); ax[0].set_ylabel('Maximal coherence Cmax = E/E0')
ax[0].set_title('A. Coherence loss under node removal'); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)

keys = list(res.keys())
ax[1].boxplot([np.array(crit[k])[np.isfinite(crit[k])] for k in keys],
              tick_labels=[f'{k[0]}\n{k[1]}' for k in keys], patch_artist=True)
ax[1].set_ylabel('Fraction removed to cross the tipping point')
ax[1].set_title('B. Keystone removal vs random loss'); ax[1].grid(alpha=0.3, axis='y')

ax[2].scatter(phi['ER'], crit[('ER', 'targeted')], s=14, color='C0', alpha=0.6, label='ER')
ax[2].scatter(phi['BA'], crit[('BA', 'targeted')], s=14, color='C3', alpha=0.6, label='BA')
ax[2].set_xlabel('Fragility index Phi = lambda_max / <k>')
ax[2].set_ylabel('Critical targeted fraction')
ax[2].set_title(f'C. Phi predicts vulnerability (pooled Spearman rho = {rho:.2f})')
ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR / 'Fig5.png', dpi=150, bbox_inches='tight')
print('Saved Fig5.png')
