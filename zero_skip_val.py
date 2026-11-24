import json, numpy as np, torch
from data import LGG
from train import build
from metrics import eval_split
data = LGG("data/lgg/kaggle_3m", "split.json", device="cuda"); out = {}
Z = lambda L: (lambda E, l: torch.zeros_like(E) if (L is None or l == L) else E)
for v in ["concat", "static", "arf", "arf_nodiff"]:
    for r in ["clean", "skip"]:
        for s in [0, 1, 2]:
            m = build(v).cuda(); m.load_state_dict(torch.load(f"runs/{v}_{r}_s{s}/best.pt", map_location="cuda")); m.eval()
            d = dict(base=eval_split(m, data, "val")["mean"], zero_all=eval_split(m, data, "val", skip_fn=Z(None))["mean"])
            for L in range(4): d[f"zero_L{L}"] = eval_split(m, data, "val", skip_fn=Z(L))["mean"]
            out[f"{v}_{r}_s{s}"] = d
            print(v, r, s, " ".join(f"{k} {x:.3f}" for k, x in d.items()), flush=True)
json.dump(out, open("zero_skip_val.json", "w"), indent=1)

