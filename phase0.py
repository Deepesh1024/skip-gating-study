import hashlib, json, torch, numpy as np
from torch.utils.flop_counter import FlopCounterMode
from data import LGG, load_raw, FLAIR_CH
from train import build, loss_fn, VARIANTS
from model import Fusion

dev = "cuda"; ok = lambda s: print("PASS", s)
data = LGG("data/lgg/kaggle_3m", "split.json", device=dev)
print("split.json sha256:", hashlib.sha256(open("split.json", "rb").read()).hexdigest())

# 1. loader
x, y = data.X[:4], data.Y[:4]
assert x.shape == (4, 1, 256, 256) and y.dtype == torch.bool and y.shape == x.shape
assert set(y.unique().tolist()) <= {False, True}
r, _ = load_raw("data/lgg/kaggle_3m", data.pid[0]); assert r.shape[1:] == (256, 256)
nz = data.X[0, 0][data.X[0, 0] != 0]; assert abs(nz.mean()) < 1e-3 and abs(nz.std() - 1) < 1e-2
ok(f"loader (FLAIR_CH={FLAIR_CH}; also eyeball audit/channels_montage.png)")

# 2. leakage / split structure
sp = {s: set(data.pid[data.split == s]) for s in ["train", "val", "test_id", "test_ood"]}
names = list(sp)
for i in range(4):
    for j in range(i + 1, 4): assert not (sp[names[i]] & sp[names[j]]), (names[i], names[j])
assert sum(len(v) for v in sp.values()) == 110
site = lambda p: p.split("_")[1]
assert {site(p) for p in sp["test_ood"]} == {"FG"} and not any(site(p) == "FG" for s in names[:3] for p in sp[s])
assert any(site(p) == "EZ" for p in sp["train"])
print({s: (len(v), int((data.split == s).sum())) for s, v in sp.items()})
nb = (data.X.flatten(1).abs().amax(1) > 0).cpu().numpy()          # non-blank FLAIR slices
print("all-zero FLAIR slices:", int((~nb).sum()), "| with non-empty mask:", int((data.Y.flatten(1).any(1).cpu().numpy() & ~nb).sum()))
print("  per split:", {s: int(((data.split == s) & ~nb).sum()) for s in names})
h = {s: {hash(data.X[i].cpu().numpy().tobytes()) for i in data.indices(s).tolist() if nb[i]} for s in names}
for i in range(4):
    for j in range(i + 1, 4): assert not (h[names[i]] & h[names[j]])
ok("leakage: patient sets disjoint, OOD = FG only, no shared image across splits")

# 3. alpha=0.5 at init and ARF output = 0.5*(Ea+Da)
xb = data.X[data.indices("train")[:8]]
for v in ["arf", "arf_nodiff"]:
    torch.manual_seed(0); m = build(v).to(dev).eval(); cap = []
    for fu in m.fuse: fu.register_forward_hook(lambda mod, i, o: cap.append((mod, i, o)))
    with torch.no_grad(): m(xb)
    for mod, i, o in cap:
        assert torch.all(mod.alpha == 0.5)
        assert torch.allclose(o, 0.5 * (mod.aE(i[0]) + mod.aD(i[1])), atol=1e-6)
ok("alpha=0.5 and output = 0.5*(Ea+Da) at init")

# 4. gate gradients (first-layer grad is exactly 0 at step 1 BY CONSTRUCTION: last layer is zero-init)
torch.manual_seed(0); m = build("arf").to(dev).train(); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
tb = data.indices("train")[:8]; xb, yb = data.X[tb], data.Y[tb]
for step in (1, 2):
    opt.zero_grad(); loss_fn(m(xb), yb).backward()
    g1 = sum(fu.g1.weight.grad.abs().sum().item() for fu in m.fuse); g2 = sum(fu.g2.weight.grad.abs().sum().item() for fu in m.fuse)
    print(f"  step {step}: |grad g1|={g1:.3e} |grad g2|={g2:.3e}")
    if step == 1: assert g2 > 0 and g1 == 0
    else: assert g1 > 0 and g2 > 0
    opt.step()
ms = build("static").to(dev); loss_fn(ms(xb), yb).backward(); assert all(fu.logit.grad.abs().sum() > 0 for fu in ms.fuse)
ok("gate gradients nonzero by step 2 (ARF) / step 1 (static)")

# 5. overfit one batch
pos = data.indices("train")[data.Y[data.indices("train")].flatten(1).any(1)][:8]; xb, yb = data.X[pos], data.Y[pos]
for v in ["concat", "static", "arf"]:
    torch.manual_seed(0); m = build(v).to(dev).train(); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad(); loss_fn(m(xb), yb).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad(): p = m(xb) > 0
    d = (2 * (p & yb).sum() / (p.sum() + yb.sum())).item(); print(f"  {v}: overfit Dice {d:.3f}"); assert d > 0.85, v
ok("overfit-one-batch")

# 6. params / FLOPs
print(f"{'variant':12s} {'params':>12s} {'GFLOPs(fwd,1x1x256x256)':>26s}")
for v in VARIANTS:
    m = build(v).to(dev).eval(); n = sum(p.numel() for p in m.parameters())
    with FlopCounterMode(display=False) as fc, torch.no_grad(): m(data.X[:1])
    print(f"{v:12s} {n:12,d} {fc.get_total_flops()/1e9:26.2f}")
print("  (FLOP counter covers conv/matmul only; norms/activations not counted)")

# 7. skip intervention changes output, per level
torch.manual_seed(0); m = build("arf").to(dev).eval(); xb = data.X[data.indices("train")[::50][:8]]
with torch.no_grad():
    o0 = m(xb)
    for L in range(4):
        d = (o0 - m(xb, skip_fn=lambda E, l: E.roll(1, 0) if l == L else E)).abs().max().item()
        print(f"  level {L}: max|dlogit|={d:.4f}"); assert d > 1e-4
ok("skip_fn roll changes output at every level")
print("\nNOT tested here: per-image alpha variance on clean data (needs trained model), step time (train.py --time_only).")