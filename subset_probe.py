"""Post-hoc probe 2 (EXPLORATORY, not pre-specified). Inference only, validation split by default.

(A) Level-subset removal: for every non-empty subset S of skip levels {0,1,2,3} (15 subsets), Dice when
    zero  : raw skips E_l, l in S, are set to 0 (all operators)
    shut  : fusion at l in S returns the decoder path D' only, i.e. alpha := 0 (gated operators)
    Redundancy test: if levels compensate for each other, loss(S) > sum of single-level losses.
(B) Pixel-level gate change (ARF variants): mean |alpha_corrupted - alpha_clean| over all gate elements at
    the corrupted level, vs the natural within-image SD of alpha. Checks that per-sample means in
    gate_probe.py did not hide a spatial redistribution of the gate.

  python subset_probe.py 2>&1 | tee subset_probe.log           # writes subset_probe_val.json
  python subset_probe.py --smoke 1 --device cpu                # random weights, tiny synthetic data
"""
import argparse, itertools, json, os, types
import numpy as np, torch
from train import build
from gate_probe import corrupt, pdice, KINDS, GATED, LEVELS

SUBSETS = [s for k in range(1, 5) for s in itertools.combinations(LEVELS, k)]
NAME = lambda s: "".join(map(str, s))
ARF = {"arf", "arf_nodiff"}

@torch.no_grad()
def dice_pass(model, data, batches, pid, zero=(), shut=()):
    model.eval()
    skip_fn = (lambda E, l: torch.zeros_like(E) if l in zero else E) if zero else None
    mods = [model.fuse[3 - l] for l in shut]                                   # fuse[0] is level 3
    for fu in mods: fu.forward = lambda E, D, fu=fu: fu.aD(D)
    try:
        T, P, N = [], [], []
        for b in batches:
            x, y = data.X[b], data.Y[b]; p = model(x, skip_fn=skip_fn) > 0
            T.append((p & y).flatten(1).sum(1)); P.append((p & ~y).flatten(1).sum(1)); N.append((~p & y).flatten(1).sum(1))
    finally:
        for fu in mods: del fu.forward
    return pdice(*[torch.cat(v).cpu().numpy().astype(np.int64) for v in (T, P, N)], pid)

@torch.no_grad()
def pixel_alpha(model, data, batches, L, kind, seed=1234):
    model.eval(); fu = model.fuse[3 - L]; g = torch.Generator(device=data.X.device).manual_seed(seed)
    s_abs = s_big = s_sd = 0.0; n = nb = 0
    for b in batches:
        x = data.X[b]
        model(x); a0 = fu.alpha.float()
        model(x, skip_fn=lambda E, l: corrupt(kind, E, g) if l == L else E); a1 = fu.alpha.float()
        d = (a1 - a0).abs(); s_abs += d.sum().item(); s_big += (d > 0.1).sum().item(); n += d.numel()
        s_sd += a0.flatten(1).std(1).sum().item(); nb += len(b)
    return dict(mean_abs=s_abs / n, frac_gt_0p1=s_big / n, within_img_sd_clean=s_sd / nb)

def probe(model, data, batches, pid, v):
    r = dict(base=dice_pass(model, data, batches, pid), zero={}, shut={}, pixel={})
    for s in SUBSETS:
        r["zero"][NAME(s)] = dice_pass(model, data, batches, pid, zero=s)
        if v in GATED: r["shut"][NAME(s)] = dice_pass(model, data, batches, pid, shut=s)
    if v in ARF:
        for k in KINDS:
            r["pixel"][k] = {str(L): pixel_alpha(model, data, batches, L, k) for L in LEVELS}
    return r

def md(d): return float(np.mean(list(d.values())))

def report(allres):
    cfgs = sorted({(r["variant"], r["regime"]) for r in allres.values()})
    pick = lambda v, rg: [r for r in allres.values() if r["variant"] == v and r["regime"] == rg]
    ms = lambda xs: f"{np.mean(xs):.3f}±{np.std(xs):.3f}"
    for mode in ["zero", "shut"]:
        print(f"\n== ({mode}) Redundancy: absolute Dice loss; sum of the 4 single-level losses vs all 4 removed together; mean±sd over seeds ==")
        print(f"{'operator':11s}{'regime':7s}{'base':>13s}{'sum singles':>14s}{'all four':>14s}{'ratio':>8s}"
              + "".join(f"{'k='+str(k)+' mean':>13s}" for k in range(1, 5)))
        for v, rg in cfgs:
            rs = [r for r in pick(v, rg) if r[mode]]
            if not rs: continue
            loss = lambda r, s: md(r["base"]) - md(r[mode][s])
            singles = [sum(loss(r, str(L)) for L in LEVELS) for r in rs]; full = [loss(r, "0123") for r in rs]
            ratio = f"{np.mean(full) / np.mean(singles):8.1f}" if np.mean(singles) > 0.005 else f"{'n/a':>8s}"
            byk = [ms([np.mean([loss(r, NAME(s)) for s in SUBSETS if len(s) == k]) for r in rs]) for k in range(1, 5)]
            print(f"{v:11s}{rg:7s}{ms([md(r['base']) for r in rs]):>13s}{ms(singles):>14s}{ms(full):>14s}{ratio}"
                  + "".join(f"{c:>13s}" for c in byk))
        print(f"\n== ({mode}) Dice loss per removed subset (mean over seeds) ==")
        print(f"{'operator':11s}{'regime':7s}" + "".join(f"{NAME(s):>6s}" for s in SUBSETS))
        for v, rg in cfgs:
            rs = [r for r in pick(v, rg) if r[mode]]
            if not rs: continue
            print(f"{v:11s}{rg:7s}" + "".join(f"{np.mean([md(r['base']) - md(r[mode][NAME(s)]) for r in rs]):6.3f}" for s in SUBSETS))
    print("\n== Pixel-level gate change at the corrupted level: mean |d alpha| [frac elements |d alpha|>0.1]; "
          "ref = within-image SD of clean alpha ==")
    print(f"{'operator':11s}{'regime':7s}{'pert':7s}" + "".join(f"{'L'+str(L):>17s}" for L in LEVELS))
    for v, rg in cfgs:
        if v not in ARF: continue
        rs = pick(v, rg)
        print(f"{v:11s}{rg:7s}{'ref':7s}" + "".join(f"{np.mean([r['pixel'][KINDS[0]][str(L)]['within_img_sd_clean'] for r in rs]):17.3f}" for L in LEVELS))
        for k in KINDS:
            print(f"{v:11s}{rg:7s}{k:7s}" + "".join(
                f"{np.mean([r['pixel'][k][str(L)]['mean_abs'] for r in rs]):9.4f} [{np.mean([r['pixel'][k][str(L)]['frac_gt_0p1'] for r in rs]):.2f}]"
                for L in LEVELS))

def smoke_data(device, n=8, hw=64):
    torch.manual_seed(0)
    X = torch.randn(n, 1, hw, hw, device=device); Y = torch.rand(n, 1, hw, hw, device=device) > 0.9
    return types.SimpleNamespace(X=X, Y=Y, pid=np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
                                 indices=lambda s: torch.arange(n, device=device))

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
    out = a.out or ("subset_probe_smoke.json" if a.smoke else f"subset_probe_{a.split}.json")
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
                r = probe(m, data, batches, pid, v); r.update(variant=v, regime=rg, seed=s, split=a.split)
                allres[key] = r; json.dump(allres, open(out, "w"))
                print(f"{key:22s} base {md(r['base']):.3f}  all-zero {md(r['zero']['0123']):.3f}", flush=True)
    report(allres)
    print(f"\nwrote {out}")

if __name__ == "__main__": main()
