import glob, hashlib, json, os, re
import numpy as np, torch
from PIL import Image

FLAIR_CH = 1   # 0=pre-contrast, 1=FLAIR, 2=post-contrast (verify with audit/channels_montage.png)

def _num(p): return int(re.search(r"_(\d+)\.tif$", p).group(1))

def load_raw(root, pid):
    imgs = sorted([p for p in glob.glob(f"{root}/{pid}_*/*.tif") if not p.endswith("_mask.tif")], key=_num)
    assert imgs, f"no images for {pid} under {root}"
    xs, ys = [], []
    for p in imgs:
        a = np.array(Image.open(p)); m = np.array(Image.open(p[:-4] + "_mask.tif"))
        assert a.shape == (256, 256, 3) and a.dtype == np.uint8, (p, a.shape, a.dtype)
        assert m.shape == (256, 256) and set(np.unique(m)) <= {0, 255}, (p, m.shape, np.unique(m))
        xs.append(a[..., FLAIR_CH]); ys.append(m > 0)          # mask > 0: never /255, never resize
    return np.stack(xs), np.stack(ys)

def zscore_nonzero(x):                                          # x: uint8 (N,H,W)
    x = x.astype(np.float32); out = np.zeros_like(x)
    for i in range(len(x)):
        nz = x[i] > 0
        if nz.sum() > 1:
            mu, sd = x[i][nz].mean(), x[i][nz].std()
            out[i][nz] = (x[i][nz] - mu) / (sd + 1e-6)          # background stays exactly 0
    return out

class LGG:
    """Whole dataset resident on the GPU (~1 GB). No DataLoader."""
    def __init__(self, root, split_json, device="cuda", cache_dir="cache"):
        split = json.load(open(split_json))
        key = hashlib.sha256(open(split_json, "rb").read()).hexdigest()[:12]
        os.makedirs(cache_dir, exist_ok=True); path = f"{cache_dir}/lgg_{key}.pt"
        if os.path.exists(path):
            d = torch.load(path)
        else:
            X, Y, pid, sp = [], [], [], []
            for q in sorted(split):
                x, y = load_raw(root, q)
                X.append(zscore_nonzero(x)); Y.append(y); pid += [q] * len(x); sp += [split[q]] * len(x)
            d = dict(X=torch.from_numpy(np.concatenate(X))[:, None],
                     Y=torch.from_numpy(np.concatenate(Y))[:, None], pid=pid, split=sp)
            torch.save(d, path)
        assert len(d["pid"]) == 3929, len(d["pid"])
        assert set(d["split"]) == {"train", "val", "test_id", "test_ood"}, set(d["split"])
        self.X, self.Y = d["X"].to(device), d["Y"].to(device)   # X float32 (N,1,256,256), Y bool
        self.pid, self.split = np.array(d["pid"]), np.array(d["split"])

    def indices(self, s):
        return torch.as_tensor(np.where(self.split == s)[0], device=self.X.device)