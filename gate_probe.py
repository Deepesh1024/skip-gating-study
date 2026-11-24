"""Post-hoc gate probe (EXPLORATORY, not pre-specified). Inference only, no training.

Question: when one skip is corrupted at test time, does an input-dependent gate (ARF) close on it?
For every trained run and every skip level L, the raw skip E_L is corrupted with perturbations that
were SEEN in skip training (roll, noise) and UNSEEN (shift, blur, chdrop, zero). Recorded:
  dice    per-patient Dice with E_L corrupted (all operators) -> SRI on unseen perturbations
  dalpha  per-sample change of mean gate weight alpha_L, corrupted minus clean (gated operators)
  oracle  Dice with level-L fusion forced to the decoder path (alpha_L := 0, E_L ignored).
          oracle > dice means closing the gate would have helped ("gate regret").
Defaults to the validation split so the test sets stay untouched.

  python gate_probe.py                       # all 24 runs on val, writes gate_probe_val.json
  python gate_probe.py --smoke 1 --device cpu  # random weights + synthetic data, checks the plumbing
"""
import argparse, json, os, types
import numpy as np, torch, torch.nn.functional as F
from train import build

KINDS = ["roll", "noise", "shift", "blur", "chdrop", "zero"]       # first two are seen in skip training
GATED = {"static", "arf", "arf_nodiff"}
LEVELS = range(4)

def corrupt(kind, E, g):
    if kind == "roll":   return E.roll(1, 0)                                         # batch order is shuffled
    if kind == "noise":  return E + torch.randn(E.shape, device=E.device, generator=g) * E.std((1, 2, 3), keepdim=True)
    if kind == "shift":  H, W = E.shape[-2:]; return E.roll((H // 8, W // 8), (2, 3))  # circular, 1/8 of size
    if kind == "blur":   return F.avg_pool2d(E, 5, 1, 2, count_include_pad=False)
    if kind == "chdrop": return E * (torch.rand(E.shape[0], E.shape[1], 1, 1, device=E.device, generator=g) < 0.5).to(E.dtype)
    if kind == "zero":   return torch.zeros_like(E)
    raise ValueError(kind)

@torch.no_grad()
def run_pass(model, data, batches, level=None, kind=None, oracle=False, seed=1234):
    """Returns per-slice tp/fp/fn and per-sample mean alpha per level (None for oracle passes)."""
    model.eval(); g = torch.Generator(device=data.X.device).manual_seed(seed)   # same noise for every model
    skip_fn = (lambda E, l: corrupt(kind, E, g) if l == level else E) if kind else None
    fu = model.fuse[3 - level] if oracle else None                                # fuse[0] is level 3
    if oracle: fu.forward = lambda E, D, fu=fu: fu.aD(D)
    try:
        T, P, N, A = [], [], [], {l: [] for l in LEVELS}
        for b in batches:
            x, y = data.X[b], data.Y[b]; p = model(x, skip_fn=skip_fn) > 0
            T.append((p & y).flatten(1).sum(1)); P.append((p & ~y).flatten(1).sum(1)); N.append((~p & y).flatten(1).sum(1))
            for i, f in enumerate(model.fuse):
                if f.alpha is not None and not oracle:
                    A[3 - i].append(f.alpha.float().mean((1, 2, 3)).expand(len(b)).cpu())
    finally:
        if oracle: del fu.forward
    tp, fp, fn = [torch.cat(v).cpu().numpy().astype(np.int64) for v in (T, P, N)]
    alpha = {l: torch.cat(v).numpy() for l, v in A.items() if v} or None
    return tp, fp, fn, alpha

def pdice(tp, fp, fn, pid):
    out = {}
    for q in np.unique(pid):
        m = pid == q; den = 2 * tp[m].sum() + fp[m].sum() + fn[m].sum()
        out[str(q)] = 1.0 if den == 0 else float(2 * tp[m].sum() / den)
    return out

def per_patient(v, pid): return {str(q): float(v[pid == q].mean()) for q in np.unique(pid)}

def probe(model, data, batches, pid, gated):
    tp, fp, fn, a0 = run_pass(model, data, batches)
    res = dict(clean=pdice(tp, fp, fn, pid), oracle={}, corr={})
    if gated: res["alpha_clean"] = {l: float(a0[l].mean()) for l in LEVELS}
    for L in LEVELS:
        if gated: res["oracle"][L] = pdice(*run_pass(model, data, batches, level=L, oracle=True)[:3], pid)
        for k in KINDS:
            tp, fp, fn, a = run_pass(model, data, batches, level=L, kind=k)
            r = dict(dice=pdice(tp, fp, fn, pid))
            if gated:
                d = a[L] - a0[L]
                r.update(dalpha=float(d.mean()), dalpha_sd=float(d.std()), frac_down=float((d < 0).mean()),
                         dalpha_patient=per_patient(d, pid))
            res["corr"].setdefault(k, {})[L] = r
    return res

def smoke_data(device):
    torch.manual_seed(0); n = 40
    X = torch.randn(n, 1, 256, 256, device=device); Y = torch.rand(n, 1, 256, 256, device=device) > 0.9
    pid = np.array(["A"] * 20 + ["B"] * 20)
    return types.SimpleNamespace(X=X, Y=Y, pid=pid, indices=lambda s: torch.arange(n, device=device))

def mean_dice(d): return float(np.mean(list(d.values())))

def report(allres):
    cfgs = sorted({(r["variant"], r["regime"]) for r in allres.values()})
    pick = lambda v, rg: [r for r in allres.values() if r["variant"] == v and r["regime"] == rg]
    ms = lambda xs: f"{np.mean(xs):6.3f}±{np.std(xs):.3f}"
    print("\n== Relative Dice drop (SRI) when skip L alone is corrupted; mean±sd over seeds. roll/noise seen in skip training ==")
    print(f"{'operator':11s}{'regime':7s}{'pert':7s}" + "".join(f"{'L'+str(L):>14s}" for L in LEVELS))
    for v, rg in cfgs:
        rs = pick(v, rg)
        for k in KINDS:
            cells = []
            for L in LEVELS:
                c = [mean_dice(r["clean"]) for r in rs]; x = [mean_dice(r["corr"][k][L]["dice"]) for r in rs]
                cells.append(ms([(ci - xi) / ci for ci, xi in zip(c, x)]))
            print(f"{v:11s}{rg:7s}{k:7s}" + "".join(f"{s:>14s}" for s in cells))
    print("\n== Gate response at the corrupted level: mean d(alpha) [frac of samples where alpha fell]; gated operators ==")
    print(f"{'operator':11s}{'regime':7s}{'pert':7s}" + "".join(f"{'L'+str(L):>20s}" for L in LEVELS))
    for v, rg in cfgs:
        if v not in GATED: continue
        rs = pick(v, rg)
        for k in KINDS:
            cells = [f"{np.mean([r['corr'][k][L]['dalpha'] for r in rs]):+.4f} [{np.mean([r['corr'][k][L]['frac_down'] for r in rs]):.2f}]"
                     for L in LEVELS]
            print(f"{v:11s}{rg:7s}{k:7s}" + "".join(f"{s:>20s}" for s in cells))
    print("\n== Gate regret: Dice(alpha_L:=0) - Dice(E_L corrupted); >0 means closing the gate would have helped ==")
    print(f"{'operator':11s}{'regime':7s}{'pert':7s}" + "".join(f"{'L'+str(L):>14s}" for L in LEVELS))
    for v, rg in cfgs:
        if v not in GATED: continue
        rs = pick(v, rg)
        for k in KINDS:
            cells = [ms([mean_dice(r["oracle"][L]) - mean_dice(r["corr"][k][L]["dice"]) for r in rs]) for L in LEVELS]
            print(f"{v:11s}{rg:7s}{k:7s}" + "".join(f"{s:>14s}" for s in cells))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="concat,static,arf,arf_nodiff"); ap.add_argument("--regimes", default="clean,skip")
    ap.add_argument("--runs", default="runs"); ap.add_argument("--root", default="data/lgg/kaggle_3m")
    ap.add_argument("--split_json", default="split.json"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.split.startswith("test"):
        print(f"WARNING: probing {a.split}. Test sets were reserved for the pre-specified analysis; label any use as post hoc.")
    out = a.out or (f"gate_probe_smoke.json" if a.smoke else f"gate_probe_{a.split}.json")
    if a.smoke:
        data = smoke_data(a.device)
    else:
        from data import LGG
        data = LGG(a.root, a.split_json, device=a.device)
    idx = data.indices(a.split)
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(777)).to(idx.device)]   # as in eval.py
    batches = torch.tensor_split(idx, int(np.ceil(len(idx) / 32))); pid = data.pid[idx.cpu().numpy()]
    allres = json.load(open(out)) if os.path.exists(out) else {}
    for v in a.variants.split(","):
        for rg in a.regimes.split(","):
            for s in (int(x) for x in a.seeds.split(",")):
                key = f"{v}_{rg}_s{s}"
                if key in allres: print("skip", key); continue
                m = build(v).to(a.device)
                if not a.smoke: m.load_state_dict(torch.load(f"{a.runs}/{key}/best.pt", map_location=a.device))
                r = probe(m, data, batches, pid, v in GATED); r.update(variant=v, regime=rg, seed=s, split=a.split)
                allres[key] = json.loads(json.dumps(r))                                    # int level keys -> str
                json.dump(allres, open(out, "w"))
                print(f"{key:22s} clean {mean_dice(r['clean']):.3f}", flush=True)
    for r in allres.values():                                                               # restore int level keys
        r["corr"] = {k: {int(L): x for L, x in d.items()} for k, d in r["corr"].items()}
        r["oracle"] = {int(L): x for L, x in r["oracle"].items()}
    report(allres)
    print(f"\nwrote {out}")

if __name__ == "__main__": main()
