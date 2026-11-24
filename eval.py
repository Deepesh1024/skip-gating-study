import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from data import LGG
from train import build

EPOCHS = 80
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--dry", type=int, default=0)
ap.add_argument("--variants", default="concat,static,arf,arf_nodiff")
a = ap.parse_args(); SEEDS = [int(s) for s in a.seeds.split(",")]
GRID = [(v, r) for v in a.variants.split(",") for r in ["clean", "skip"]]
RUNS = [(v, r, s) for (v, r) in GRID for s in SEEDS]
OUT = "results_dry" if a.dry else "results"; os.makedirs(OUT, exist_ok=True)
SPLIT = {"test_id": "val", "test_ood": "val"} if a.dry else {"test_id": "test_id", "test_ood": "test_ood"}

def complete(v, r, s):
    d = f"runs/{v}_{r}_s{s}"
    return os.path.exists(f"{d}/best.pt") and os.path.exists(f"{d}/log.json") and len(json.load(open(f"{d}/log.json"))["log"]) == EPOCHS
if a.dry:
    RUNS = [j for j in RUNS if os.path.exists(f"runs/{j[0]}_{j[1]}_s{j[2]}/best.pt")][:2]
    assert RUNS, "no best.pt yet"
else:
    bad = [j for j in RUNS if not complete(*j)]
    if bad: sys.exit(f"REFUSING: {len(bad)} runs incomplete, e.g. {bad[:3]}. Test sets stay untouched.")

def hd95(p, g):                                       # bool 2D arrays; NaN if undefined
    if not p.any() or not g.any(): return np.nan
    sp, sg = p & ~binary_erosion(p), g & ~binary_erosion(g)
    d = np.concatenate([distance_transform_edt(~sg)[sp], distance_transform_edt(~sp)[sg]])
    return float(np.percentile(d, 95))

def gblur(x, s):
    r = int(np.ceil(3 * s)); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))

class Corrupt:                                        # new object per evaluation => identical noise for every model
    def __init__(s, kind, level, seed=1234):
        s.kind, s.level = kind, level; s.g = torch.Generator(device="cuda").manual_seed(seed)
    def __call__(s, x):
        if s.kind == "noise": return x + s.level * torch.randn(x.shape, device=x.device, generator=s.g)
        if s.kind == "blur": return gblur(x, s.level)
        f = torch.randn(x.shape[0], 1, 4, 4, device=x.device, generator=s.g)           # bias field
        f = F.interpolate(f, size=x.shape[-2:], mode="bicubic", align_corners=False)
        return x * (1 + s.level * f / f.abs().amax((1, 2, 3), keepdim=True))

class AlphaStats:
    def __init__(s): s.acc = {}
    def __call__(s, model):
        for i, fu in enumerate(model.fuse):
            if fu.alpha is None: continue
            al = fu.alpha.float(); d = s.acc.setdefault(3 - i, dict(mean=[], bm=[], ws=[]))   # fuse[0] = level 3
            d["mean"].append(al.mean().item())
            if al.shape[0] > 1 and al.shape[2] > 1:
                d["bm"] += al.mean((1, 2, 3)).tolist(); d["ws"] += al.flatten(1).std(1).tolist()
    def summary(s):
        return {l: dict(mean=float(np.mean(d["mean"])),
                        between_img_sd=float(np.std(d["bm"])) if d["bm"] else None,
                        within_img_sd=float(np.mean(d["ws"])) if d["ws"] else None) for l, d in s.acc.items()}

@torch.no_grad()
def predict(model, data, split, skip_fn=None, transform=None, perm_seed=None, hd=False, cb=None):
    model.eval(); idx = data.indices(split)
    if perm_seed is not None:
        idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(perm_seed)).to(idx.device)]
    TP, FP, FN, H = [], [], [], []
    for b in torch.tensor_split(idx, int(np.ceil(len(idx) / 32))):      # near-equal batches, never size 1
        x, y = data.X[b], data.Y[b]
        if transform is not None: x = transform(x)
        p = model(x, skip_fn=skip_fn) > 0
        if cb is not None: cb(model)
        TP.append((p & y).flatten(1).sum(1)); FP.append((p & ~y).flatten(1).sum(1)); FN.append((~p & y).flatten(1).sum(1))
        if hd: H += [hd95(pi[0], yi[0]) for pi, yi in zip(p.cpu().numpy(), y.cpu().numpy())]
    tp, fp, fn = [torch.cat(v).cpu().numpy().astype(np.int64) for v in (TP, FP, FN)]
    return dict(tp=tp, fp=fp, fn=fn, pid=data.pid[idx.cpu().numpy()], hd=np.array(H))

def pdice(r):
    out = {}
    for q in np.unique(r["pid"]):
        m = r["pid"] == q; den = 2 * r["tp"][m].sum() + r["fp"][m].sum() + r["fn"][m].sum()
        out[str(q)] = 1.0 if den == 0 else float(2 * r["tp"][m].sum() / den)
    return out

def slice_summary(r):
    ne = (r["tp"] + r["fn"]) > 0; d = 2 * r["tp"] / np.maximum(2 * r["tp"] + r["fp"] + r["fn"], 1)
    h = r["hd"]; hn = h[ne] if len(h) else np.array([])
    hp = {str(q): (float(np.nanmean(h[(r["pid"] == q) & ne])) if np.any(~np.isnan(h[(r["pid"] == q) & ne])) else None)
          for q in np.unique(r["pid"])} if len(h) else {}
    return dict(dice_nonempty_slices=float(d[ne].mean()), fp_pixels_per_empty_slice=float(r["fp"][~ne].mean()),
                n_empty=int((~ne).sum()), n_nonempty=int(ne.sum()),
                hd95_mean=float(np.nanmean(hn)) if len(hn) and np.any(~np.isnan(hn)) else None,
                hd95_undefined_frac=float(np.isnan(hn).mean()) if len(hn) else None, hd95_patient=hp)

data = LGG("data/lgg/kaggle_3m", "split.json", device="cuda")
CORR = [("noise", [.1, .2, .4]), ("blur", [1, 2, 3]), ("bias", [.1, .2, .3])]
summ = {}
for (v, r, s) in RUNS:
    path = f"{OUT}/{v}_{r}_s{s}.json"
    if os.path.exists(path): print("skip", path); continue
    run = f"runs/{v}_{r}_s{s}"; m = build(v).cuda(); m.load_state_dict(torch.load(f"{run}/best.pt", map_location="cuda")); m.eval()
    lg = json.load(open(f"{run}/log.json"))["log"]; bi = max(range(len(lg)), key=lambda i: lg[i]["val_dice"])
    res = dict(variant=v, regime=r, seed=s, best_epoch=bi, best_val=lg[bi]["val_dice"], epochs_logged=len(lg), dice={}, slices={})
    for name in ["test_id", "test_ood"]:
        cb = AlphaStats() if name == "test_id" else None
        o = predict(m, data, SPLIT[name], hd=True, cb=cb)
        res["dice"][f"{name}/clean"] = pdice(o); res["slices"][name] = slice_summary(o)
        if cb: res["alpha_test_id"] = cb.summary()
    for kind, levels in CORR:
        for lv in levels:
            res["dice"][f"test_id/{kind}_{lv}"] = pdice(predict(m, data, SPLIT["test_id"], transform=Corrupt(kind, lv)))
    for L in range(4):                                # skip intervention: roll E_L across a fixed shuffled order
        res["dice"][f"test_id/roll_L{L}"] = pdice(predict(m, data, SPLIT["test_id"], perm_seed=777,
                                                          skip_fn=lambda E, l, L=L: E.roll(1, 0) if l == L else E))
    json.dump(res, open(path, "w"))
    c = lambda k: np.mean(list(res["dice"][k].values()))
    print(f"{v:10s} {r:5s} s{s} best_ep {bi:2d} val {res['best_val']:.4f} | id {c('test_id/clean'):.4f} ood {c('test_ood/clean'):.4f} "
          f"| roll L0..3: " + " ".join(f"{c(f'test_id/roll_L{L}'):.3f}" for L in range(4)), flush=True)

