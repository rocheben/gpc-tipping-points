# GPC tipping points

Code for the dynamical model of tumour Group Phenotypic Composition (GPC)
destabilisation: bistability and hysteresis under therapy, sequential vs simultaneous
perturbations, microenvironmental exposure, keystone phenotypes in interaction networks,
coupling with a clonal compartment (adaptive vs continuous therapy), a spatial
agent-based version, and a metastatic meta-community.

Authors: Benjamin Roche, Margaux Bieuville, Alexandre Fontanella, Antoine M. Dujon,
Frédéric Thomas.

## Contents

```
scripts/   one script per figure: figN_*.py writes figures/FigN.png,
           figSN_*.py writes figures/FigSN.png; run_all.py runs them all
figures/   Fig1 to Fig8 (main text), FigS1 to FigS8 (Supplementary Materials)
logs/      printed output of each script (every number reported in the article
           comes from these files) and _run.txt (run times)
```

All paths are relative to the location of the scripts, so the repository can be run
from anywhere.

## Reproduction

Python 3.9 or later.

```bash
pip install -r scripts/requirements.txt
python scripts/run_all.py              # all scripts, about 90 min on 2 cores
python scripts/run_all.py fig3 fig6    # only scripts whose name starts with these prefixes
```

All random draws are seeded, so the logs are reproduced exactly (only the reported run
times change). Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11 and
networkx 3.7.

| Figure | Script | Run time (2 cores) |
|---|---|---|
| Fig. 1 Model schematic | `fig1_model_diagram.py` | 1 s |
| Fig. 2 Bistability | `fig2_bistability.py` | 46 s |
| Fig. 3 Sequential vs simultaneous | `fig3_sequential_vs_simultaneous.py` | 61 s |
| Fig. 4 Microenvironment, exposure duration | `fig4_microenvironment_duration.py` | 30 s |
| Fig. 5 Networks, keystone phenotypes | `fig5_network_keystones.py` | 5 s |
| Fig. 6 Clonal coupling | `fig6_clonal_coupling.py` | 10 s |
| Fig. 7 (φ, ω) map | `fig7_omega_phi_map.py` | 67 s |
| Fig. 8 Spatial agent-based model | `fig8_spatial_abm.py` | 24 min |
| Fig. S1 Hysteresis | `figS1_hysteresis.py` | 1 s |
| Fig. S2 Quasi-potential | `figS2_quasipotential.py` | 2 s |
| Fig. S3 Stochastic relapse | `figS3_stochastic_relapse.py` | 2 s |
| Fig. S4 Adaptive therapy, collective model only | `figS4_adaptive_collective.py` | 1 s |
| Fig. S5 Critical slowing down | `figS5_critical_slowing_down.py` | 13 s |
| Fig. S6 Metastatic meta-community | `figS6_metastatic_metacommunity.py` | 16 min |
| Fig. S7 Collapse threshold, C–S couplings | `figS7_model_robustness.py` | 40 s |
| Fig. S8 Agent-based model: boundaries, shape, size | `figS8_abm_size_geometry.py` | 42 min |

`fig8_spatial_abm.py` and `figS8_abm_size_geometry.py` use every available core; the other scripts run on one core.

## License

MIT, see [LICENSE](LICENSE).
