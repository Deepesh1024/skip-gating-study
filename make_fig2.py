"""Fig. 2 (SRI by level, ID test) redrawn in the same validated palette and style as Fig. 3.
Numbers identical to analyze.py's fig2_sri: SRI = (Dice_clean - Dice_roll(L)) / Dice_clean, mean ± SD (ddof=1) over seeds.
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

V = ["concat", "static", "arf", "arf_nodiff"]; SEEDS = [0, 1, 2]
LABEL = {"concat": "concat", "static": "static", "arf": "ARF", "arf_nodiff": "ARF-nodiff"}
COL = {"concat": "#2a78d6", "static": "#eb6834", "arf": "#1baf7a", "arf_nodiff": "#6250d6"}
R = {(v, r, s): json.load(open(f"results/{v}_{r}_s{s}.json")) for v in V for r in ["clean", "skip"] for s in SEEDS}

def sri(v, r, s, L):
    d = R[(v, r, s)]["dice"]; c = np.mean(list(d["test_id/clean"].values()))
    return (c - np.mean(list(d[f"test_id/roll_L{L}"].values()))) / c

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for a, r, title in zip(ax, ["clean", "skip"], ["(a) clean-trained", "(b) skip-corruption-trained"]):
    for j, v in enumerate(V):
        rows = [[sri(v, r, s, L) for s in SEEDS] for L in range(4)]
        a.bar(np.arange(4) + 0.2 * j - 0.3, [np.mean(x) for x in rows], 0.18, yerr=[np.std(x, ddof=1) for x in rows],
              color=COL[v], label=LABEL[v], edgecolor="white", linewidth=0.8, error_kw=dict(lw=0.8, ecolor="#52514e"))
    a.set_title(title, fontsize=8, loc="left"); a.set_xticks(range(4)); a.set_xlabel("skip level (0 = shallowest)")
    a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True); a.axhline(0, color="#8a8984", lw=0.6)
ax[0].set_ylabel("SRI (ID test)"); ax[1].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("fig2_sri.pdf"); fig.savefig("fig2_sri.png", dpi=200)
print("wrote fig2_sri.pdf/.png")
