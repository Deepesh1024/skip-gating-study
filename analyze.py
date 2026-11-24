import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
B = 10000; RNG = np.random.default_rng(20261106)
VAR = ["concat", "static", "arf", "arf_nodiff"]; REG = ["clean", "skip"]
CORR = [("noise", [.1, .2, .4]), ("blur", [1, 2, 3]), ("bias", [.1, .2, .3])]
GFLOPS = {"concat": 20.85, "static": 19.77, "arf": 23.53, "arf_nodiff": 22.32}      # from phase0.log

R = {}
for v in VAR:
    for r in REG:
        for s in SEEDS:
            try: R[(v, r, s)] = json.load(open(f"results/{v}_{r}_s{s}.json"))
            except FileNotFoundError: sys.exit(f"missing results/{v}_{r}_s{s}.json")

_c = {}
def mat(v, r, cond):
    k = (v, r, cond)
    if k not in _c:
        ids = sorted(R[(v, r, SEEDS[0])]["dice"][cond])
        for s in SEEDS: assert sorted(R[(v, r, s)]["dice"][cond]) == ids
        _c[k] = (ids, np.array([[R[(v, r, s)]["dice"][cond][p] for p in ids] for s in SEEDS]))
    return _c[k]
def P(v, r, cond): return mat(v, r, cond)[1].mean(0)            # per-patient Dice, averaged over seeds

def vec(terms, cond):
    ids0 = mat(*terms[0][1], cond)[0]; out = 0
    for sg, (v, r) in terms:
        assert mat(v, r, cond)[0] == ids0; out = out + sg * P(v, r, cond)
    return out
def DIFF(a, b, cond): return vec([(1, a), (-1, b)], cond)
def INTER(a1, b1, a0, b0, cond): return vec([(1, a1), (-1, b1), (-1, a0), (1, b0)], cond)

def boot(d):                                                      # patient-clustered paired bootstrap
    rng = np.random.default_rng(20261106)
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = max(2 * min((bm <= 0).mean(), (bm >= 0).mean()), 1 / B)
    return dict(est=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(p, 1)), n=int(len(d)))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
    return adj
fmt = lambda c: f"{c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"

ARF_S, STA_S, ARF_C, STA_C = ("arf", "skip"), ("static", "skip"), ("arf", "clean"), ("static", "clean")
sd = float(np.std([R[("concat", "clean", s)]["best_val"] for s in SEEDS], ddof=1)); delta = max(0.005, sd)
out = dict(delta=delta, seed_sd_val_concat_clean=sd)
print(f"delta = max(0.005, SD_seeds(concat-clean val)={sd:.4f}) = {delta:.4f}\n")

# ---- primary / control ----
h1 = boot(INTER(ARF_S, STA_S, ARF_C, STA_C, "test_ood/clean")); out["H1_ood"] = h1
print("H1  interaction [(ARF-static)|skip - (ARF-static)|clean], OOD(FG), per-patient Dice:", fmt(h1), f"p={h1['p']:.4f}")
print("    ->", "SUPPORTED (CI excludes 0 and estimate > delta)" if h1["lo"] > 0 and h1["est"] > delta
      else "NOT SUPPORTED: no evidence of an interaction at this design's resolution")
h2 = boot(DIFF(ARF_C, STA_C, "test_id/clean")); out["H2_id"] = h2
print("H2  ARF-static, clean training, ID test:", fmt(h2),
      "->", "within +-delta" if h2["lo"] > -delta and h2["hi"] < delta else "NOT shown to be within +-delta")

# ---- descriptive contrasts ----
out["contrasts"] = {}
def show(name, d, cond):
    c = boot(d); out["contrasts"][f"{name}|{cond}"] = c; print(f"  {name:45s} {cond:14s} {fmt(c)}")
print("\nDescriptive contrasts (no multiplicity control; exploratory):")
for cond in ["test_id/clean", "test_ood/clean"]:
    show("ARF-static | clean", DIFF(ARF_C, STA_C, cond), cond)
    show("ARF-static | skip", DIFF(ARF_S, STA_S, cond), cond)
    show("interaction ARF-static", INTER(ARF_S, STA_S, ARF_C, STA_C, cond), cond)
    show("concat-static | clean", DIFF(("concat", "clean"), STA_C, cond), cond)
    show("concat-static | skip", DIFF(("concat", "skip"), STA_S, cond), cond)
    for v in VAR: show(f"skip-regime effect, {v}", DIFF((v, "skip"), (v, "clean"), cond), cond)
    show("H3a interaction (ARF - ARFnodiff)", INTER(ARF_S, ("arf_nodiff", "skip"), ARF_C, ("arf_nodiff", "clean"), cond), cond)

# ---- corruption family (Holm over 9 interaction tests) ----
fam = {}
for kind, lv in CORR:
    for l in lv:
        c = f"test_id/{kind}_{l}"; fam[c] = boot(INTER(ARF_S, STA_S, ARF_C, STA_C, c))
for (c, d), a in zip(fam.items(), holm([d["p"] for d in fam.values()])): d["p_holm"] = float(a)
out["corruption_interactions"] = fam
print("\nCorruption interactions (ARF-static, skip minus clean), Holm over 9:")
for c, d in fam.items(): print(f"  {c:22s} {fmt(d)}  p={d['p']:.4f}  p_holm={d['p_holm']:.4f}")

# ---- SRI and gate statistics ----
def sri(v, r, s, L):
    d = R[(v, r, s)]["dice"]; c = np.mean(list(d["test_id/clean"].values()))
    return (c - np.mean(list(d[f"test_id/roll_L{L}"].values()))) / c
mpm = lambda x: f"{np.mean(x):.3f}+-{np.std(x, ddof=1):.3f}"
print("\nSRI (test_id; mean+-SD over seeds), level 0 = shallowest:")
out["SRI"] = {}
for v in VAR:
    for r in REG:
        row = [[sri(v, r, s, L) for s in SEEDS] for L in range(4)]
        out["SRI"][f"{v}/{r}"] = [float(np.mean(x)) for x in row]
        print(f"  {v:10s} {r:5s} " + "  ".join(f"L{L} {mpm(row[L])}" for L in range(4)))
print("\nGate alpha on test_id (mean over seeds of: mean alpha | between-image SD | within-image SD):")
for v in ["static", "arf", "arf_nodiff"]:
    for r in REG:
        for L in range(4):
            g = [R[(v, r, s)]["alpha_test_id"][str(L)] for s in SEEDS]
            f = lambda k: "n/a" if g[0][k] is None else f"{np.mean([x[k] for x in g]):.4f}"
            print(f"  {v:10s} {r:5s} L{L}  mean {f('mean')} | between {f('between_img_sd')} | within {f('within_img_sd')}")

# ---- main table ----
from train import build
print("\nMAIN TABLE (mean+-SD over seeds):")
print("variant/regime | params | GFLOPs | val | ID Dice | OOD Dice | ID HD95(px) | ID FP px/empty slice")
for v in VAR:
    n = sum(p.numel() for p in build(v).parameters())
    for r in REG:
        g = lambda f: [f(R[(v, r, s)]) for s in SEEDS]
        pm = lambda c: np.mean(list(c.values()))
        hd = [x for x in g(lambda q: q["slices"]["test_id"]["hd95_mean"]) if x is not None]
        print(f"{v}/{r} | {n:,} | {GFLOPS[v]} | {mpm(g(lambda q: q['best_val']))} | "
              f"{mpm(g(lambda q: pm(q['dice']['test_id/clean'])))} | {mpm(g(lambda q: pm(q['dice']['test_ood/clean'])))} | "
              f"{mpm(hd) if hd else 'n/a'} | {mpm(g(lambda q: q['slices']['test_id']['fp_pixels_per_empty_slice']))}")

# ---- figures ----
col = {"concat": "tab:gray", "static": "tab:blue", "arf": "tab:red", "arf_nodiff": "tab:orange"}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for a, (kind, lv) in zip(ax, CORR):
    for v in VAR:
        for r in REG:
            ys = [P(v, r, "test_id/clean").mean()] + [P(v, r, f"test_id/{kind}_{l}").mean() for l in lv]
            a.plot(range(4), ys, "-" if r == "skip" else "--", color=col[v], marker="o", ms=3, label=f"{v}/{r}")
    a.set_xticks(range(4)); a.set_xticklabels(["clean"] + [str(l) for l in lv]); a.set_title(kind)
ax[0].set_ylabel("per-patient Dice (ID test)"); ax[2].legend(fontsize=6); fig.tight_layout()
fig.savefig("fig1_corruption.pdf"); fig.savefig("fig1_corruption.png", dpi=200)
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for a, r in zip(ax, REG):
    for j, v in enumerate(VAR):
        rows = [[sri(v, r, s, L) for s in SEEDS] for L in range(4)]
        a.bar(np.arange(4) + .2 * j - .3, [np.mean(x) for x in rows], .2, yerr=[np.std(x, ddof=1) for x in rows], color=col[v], label=v)
    a.set_title(f"{r} training"); a.set_xticks(range(4)); a.set_xlabel("skip level (0 = shallowest)")
ax[0].set_ylabel("SRI"); ax[1].legend(fontsize=6); fig.tight_layout()
fig.savefig("fig2_sri.pdf"); fig.savefig("fig2_sri.png", dpi=200)
json.dump(out, open("analysis.json", "w"), indent=1)
print("\nwrote analysis.json, fig1_corruption.*, fig2_sri.*")

