# GPC tipping points

Code for the dynamical model of tumour Group Phenotypic Composition (GPC)
destabilisation: bistability and hysteresis under therapy, sequential vs simultaneous
perturbations, microenvironmental exposure, keystone phenotypes in interaction networks,
coupling with a clonal compartment (adaptive vs continuous therapy), a spatial
agent-based version, and a metastatic meta-community.

Authors: Benjamin Roche, Frédéric Thomas.

## Contents

```
scripts/   one script per figure: figN_*.py writes figures/FigN.png,
           figSN_*.py writes figures/FigSN.png; run_all.py runs them all
figures/   Fig1 to Fig8 (main text), FigS1 to FigS5 (Supplementary Materials)
logs/      printed output of each script (every number reported in the article
           comes from these files) and _run.txt (run times)
```

All paths are relative to the location of the scripts, so the repository can be run
from anywhere.

## Reproduction

Python 3.9 or later.

```bash
pip install -r scripts/requirements.txt
python scripts/run_all.py              # all scripts, about 45 min on 2 cores
python scripts/run_all.py fig3 fig6    # only scripts whose name starts with these prefixes
```

All random draws are seeded, so the logs are reproduced exactly (only the reported run
times change). Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11 and
networkx 3.7.

| Figure | Script | Run time (2 cores) |
|---|---|---|
| Fig. 1 Bistability | `fig1_bistability.py` | 46 s |
| Fig. 2 Hysteresis | `fig2_hysteresis.py` | 1 s |
| Fig. 3 Sequential vs simultaneous | `fig3_sequential_vs_simultaneous.py` | 64 s |
| Fig. 4 Microenvironment, exposure duration | `fig4_microenvironment_duration.py` | 30 s |
| Fig. 5 Networks, keystone phenotypes | `fig5_network_keystones.py` | 5 s |
| Fig. 6 Clonal coupling | `fig6_clonal_coupling.py` | 10 s |
| Fig. 7 (φ, ω) map | `fig7_omega_phi_map.py` | 69 s |
| Fig. 8 Spatial agent-based model | `fig8_spatial_abm.py` | 24 min |
| Fig. S1 Quasi-potential | `figS1_quasipotential.py` | 2 s |
| Fig. S2 Stochastic relapse | `figS2_stochastic_relapse.py` | 2 s |
| Fig. S3 Adaptive therapy, collective model only | `figS3_adaptive_collective.py` | 1 s |
| Fig. S4 Critical slowing down | `figS4_critical_slowing_down.py` | 14 s |
| Fig. S5 Metastatic meta-community | `figS5_metastatic_metacommunity.py` | 16 min |

`fig8_spatial_abm.py` uses every available core; the other scripts run on one core.

## License

MIT, see [LICENSE](LICENSE).
