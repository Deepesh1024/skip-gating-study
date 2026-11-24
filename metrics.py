import numpy as np, torch

@torch.no_grad()
def eval_split(model, data, split, bs=32, skip_fn=None, transform=None):
    """Per-patient Dice from TP/FP/FN summed over a patient's slices (threshold: logit > 0)."""
    model.eval(); idx = data.indices(split); T, P, N = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; x = data.X[b]; y = data.Y[b]
        if transform is not None: x = transform(x)
        p = model(x, skip_fn=skip_fn) > 0
        T.append((p & y).flatten(1).sum(1)); P.append((p & ~y).flatten(1).sum(1)); N.append((~p & y).flatten(1).sum(1))
    tp, fp, fn = [torch.cat(v).cpu().numpy().astype(np.int64) for v in (T, P, N)]
    ii = idx.cpu().numpy(); pid = data.pid[ii]; per = {}
    for q in np.unique(pid):
        m = pid == q; den = 2 * tp[m].sum() + fp[m].sum() + fn[m].sum()
        per[q] = 1.0 if den == 0 else float(2 * tp[m].sum() / den)   # both empty -> 1 (never happens: every patient has tumor)
    return dict(patient_dice=per, mean=float(np.mean(list(per.values()))), tp=tp, fp=fp, fn=fn, pid=pid, idx=ii)