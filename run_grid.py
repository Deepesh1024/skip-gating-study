import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

EPOCHS = 80                                    # fixed budget; do not change after any result is seen
seeds = [int(s) for s in sys.argv[1].split(",")]
workers = int(sys.argv[2])
order = [("concat","clean"),("static","clean"),("arf","clean"),("concat","skip"),
         ("static","skip"),("arf","skip"),("arf_nodiff","clean"),("arf_nodiff","skip")]
jobs = [(v, r, s) for s in seeds for (v, r) in order]

def done(v, r, s):
    p = f"runs/{v}_{r}_s{s}/log.json"
    return os.path.exists(p) and len(json.load(open(p))["log"]) == EPOCHS

def run(job):
    v, r, s = job
    if done(*job): return f"SKIP (already complete) {job}"
    d = f"runs/{v}_{r}_s{s}"; os.makedirs(d, exist_ok=True); t = time.time()
    with open(f"{d}/stdout.txt", "w") as f:
        rc = subprocess.call([sys.executable, "train.py", "--variant", v, "--regime", r, "--seed", str(s),
                              "--epochs", str(EPOCHS)], stdout=f, stderr=subprocess.STDOUT)
    return f"{job} rc={rc} {time.time()-t:.0f}s"

with ThreadPoolExecutor(workers) as ex:
    for fu in as_completed([ex.submit(run, j) for j in jobs]): print(fu.result(), flush=True)

