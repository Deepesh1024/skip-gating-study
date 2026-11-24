import torch, torch.nn as nn, torch.nn.functional as F

def gn(c): return nn.GroupNorm(8, c)

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), gn(cout), nn.ReLU(inplace=True))

class Align(nn.Module):                      # identical for E and D in EVERY variant
    def __init__(s, cin, cout):
        super().__init__(); s.c = nn.Conv2d(cin, cout, 1, bias=False); s.n = gn(cout)
    def forward(s, x): return s.n(s.c(x))

class Fusion(nn.Module):
    def __init__(s, w, cd, kind, use_diff=True, mode="channel_spatial"):
        super().__init__()
        s.kind, s.use_diff, s.mode, s.alpha = kind, use_diff, mode, None
        s.aE, s.aD = Align(w, w), Align(cd, w)
        if kind == "concat":
            s.proj = nn.Conv2d(2 * w, w, 1)
        elif kind == "static":
            s.logit = nn.Parameter(torch.zeros(1, w, 1, 1))
        elif kind == "arf":
            cin, hid = w * (3 if use_diff else 2), max(w // 4, 8)
            k = 1 if mode == "channel_only" else 3
            s.g1 = nn.Conv2d(cin, hid, k, padding=k // 2)
            s.g2 = nn.Conv2d(hid, 1 if mode == "spatial_only" else w, 1)
            nn.init.zeros_(s.g2.weight); nn.init.zeros_(s.g2.bias)   # alpha = 0.5 at init
        elif kind == "noskip": pass
        elif kind != "add":
            raise ValueError(kind)

    def forward(s, E, D):
        E, D = s.aE(E), s.aD(D)
        if s.kind == "noskip": return D
        if s.kind == "concat": return s.proj(torch.cat([E, D], 1))
        if s.kind == "add":    return E + D
        if s.kind == "static":
            a = torch.sigmoid(s.logit)
        else:
            z = torch.cat([E, D, (E - D).abs()] if s.use_diff else [E, D], 1)
            if s.mode == "channel_only": z = z.mean((2, 3), keepdim=True)
            a = torch.sigmoid(s.g2(F.relu(s.g1(z))))
        s.alpha = a.detach()                 # for gate analysis
        return a * E + (1 - a) * D

class SegNet(nn.Module):
    def __init__(s, kind="arf", widths=(32, 64, 128, 256), bott=512, in_ch=1, **kw):
        super().__init__()
        s.enc, c = nn.ModuleList(), in_ch
        for w in widths: s.enc.append(conv_block(c, w)); c = w
        s.bott = conv_block(c, bott)
        s.fuse, s.dec, cd = nn.ModuleList(), nn.ModuleList(), bott
        for w in reversed(widths):
            s.fuse.append(Fusion(w, cd, kind, **kw)); s.dec.append(conv_block(w, w)); cd = w
        s.head = nn.Conv2d(widths[0], 1, 1)

    def forward(s, x, skip_fn=None):         # skip_fn(E, level) -> E': corruption / shuffle hook
        skips = []
        for blk in s.enc:
            x = blk(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = s.bott(x)
        for i, (fu, dec) in enumerate(zip(s.fuse, s.dec)):
            lvl = len(skips) - 1 - i
            E = skips[lvl] if skip_fn is None else skip_fn(skips[lvl], lvl)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = dec(fu(E, x))
        return s.head(x)                     # logits; use BCEWithLogits + Dice on sigmoid

