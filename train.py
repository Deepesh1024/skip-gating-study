import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
from model import SegNet
from data import LGG
from metrics import eval_split

VARIANTS = {"concat": dict(kind="concat"), "static": dict(kind="static"), "add": dict(kind="add"),
            "arf": dict(kind="arf"), "arf_nodiff": dict(kind="arf", use_diff=False),
            "noskip": dict(kind="noskip")}

def build(v): return SegNet(**VARIANTS[v])

def loss_fn(logits, y):
    y = y.float(); p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)     # pooled over batch (65% of slices are empty)
    return 0.5 * F.binary_cross_entropy_with_logits(logits, y) + 0.5 * dice

def make_skip_fn(p=0.3):
    """Per sample, per level: with prob p replace E by batch-rolled E (50%) or E + N(0, std(E)^2) (50%)."""
    def fn(E, l):
        B = E.shape[0]; v = lambda t: t.view(B, 1, 1, 1)
        sel = v(torch.rand(B, device=E.device) < p); use_roll = v(torch.rand(B, device=E.device) < 0.5)
        noisy = E + torch.randn_like(E) * E.std(dim=(1, 2, 3), keepdim=True)
        return torch.where(sel, torch.where(use_roll, E.roll(1, 0), noisy), E)
    return fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS); ap.add_argument("--regime", default="clean", choices=["clean", "skip"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--root", default="data/lgg/kaggle_3m"); ap.add_argument("--split_json", default="split.json")
    ap.add_argument("--out", default="runs"); ap.add_argument("--time_only", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"; data = LGG(a.root, a.split_json, device=dev)
    model = build(a.variant).to(dev); nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    skip = make_skip_fn(a.p) if a.regime == "skip" else None
    tr = data.indices("train"); g = torch.Generator().manual_seed(a.seed)
    run = f"{a.out}/{a.variant}_{a.regime}_s{a.seed}"; os.makedirs(run, exist_ok=True)
    log, best, step, t0 = [], -1.0, 0, None; T0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = tr[torch.randperm(len(tr), generator=g).to(dev)]; tl = []
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            if len(b) < 2: continue                                   # roll needs >=2 samples
            x, y = data.X[b], data.Y[b]
            f = (torch.rand(len(b), device=dev) < 0.5).view(-1, 1, 1, 1)   # only augmentation: hflip, both regimes
            x, y = torch.where(f, x.flip(-1), x), torch.where(f, y.flip(-1), y)
            loss = loss_fn(model(x, skip_fn=skip), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl.append(loss.item()); step += 1
            if a.time_only:
                if step == 20: torch.cuda.synchronize(); t0 = time.time()
                if step == 20 + a.time_only:
                    torch.cuda.synchronize(); dt = (time.time() - t0) / a.time_only
                    print(f"{a.variant}/{a.regime}: {dt:.4f} s/step, {len(tr)//a.bs} steps/epoch, "
                          f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, params {nparams:,}"); return
        v = eval_split(model, data, "val")["mean"]                    # selection always on CLEAN val, both regimes
        log.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_dice=v, sec=time.time()-T0))
        if v > best: best = v; torch.save(model.state_dict(), f"{run}/best.pt"); log[-1]["best"] = True
        json.dump(dict(args=vars(a), params=nparams, log=log), open(f"{run}/log.json", "w"), indent=1)
        print(f"ep {ep:3d} loss {np.mean(tl):.4f} val_patient_dice {v:.4f} best {best:.4f}" + f" t {time.time()-T0:.0f}s", flush=True)

if __name__ == "__main__": main()