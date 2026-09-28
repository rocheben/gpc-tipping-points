"""Run every figure script and keep its printed output in ../logs/<script>.log.
Usage:  python run_all.py            (all scripts)
        python run_all.py fig3 fig6   (only scripts whose name starts with these prefixes)
Scripts fig8 (about 27 min on 2 cores) and figS5 (about 17 min) are the slow ones."""
import os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / 'logs'; LOGS.mkdir(exist_ok=True)
# UTF-8 logs on every OS, and no display needed (headless servers, clusters)
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
scripts = sorted(p for p in HERE.glob('fig*.py'))
if len(sys.argv) > 1:
    scripts = [p for p in scripts if any(p.name.startswith(a) for a in sys.argv[1:])]
for p in scripts:
    t0 = time.time()
    with open(LOGS / (p.stem + '.log'), 'w', encoding='utf-8') as f:
        r = subprocess.run([sys.executable, p.name], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    print(f'{p.name:<40} exit {r.returncode}   {time.time()-t0:6.0f} s', flush=True)
