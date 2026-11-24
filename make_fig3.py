"""Fig. 3: Dice loss vs number of skip levels removed (post-hoc, validation set).
Gated operators: gate shut (alpha := 0); concat: skips zeroed (it has no gate).
Solid = observed (mean over all subsets of size k, then mean ± SD over 3 seeds).
Dotted = additive prediction, k x mean single-level loss.
"""
import itertools, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

S = json.load(open("subset_probe_val.json"))
V = ["concat", "static", "arf", "arf_nodiff"]
LABEL = {"concat": "concat", "static": "static", "arf": "ARF", "arf_nodiff": "ARF-nodiff"}
COL = {"concat": "#2a78d6", "static": "#eb6834", "arf": "#1baf7a", "arf_nodiff": "#6250d6"}   # validated (CVD ΔE ≥ 9.2)
MK = {"concat": "o", "static": "s", "arf": "^", "arf_nodiff": "D"}                              # secondary encoding
SUB = [s for k in range(1, 5) for s in itertools.combinations(range(4), k)]
md = lambda d: float(np.mean(list(d.values())))

def curves(v, rg):
    mode = "zero" if v == "concat" else "shut"; obs, add = [], []
    for s in range(3):
        r = S[f"{v}_{rg}_s{s}"]; b = md(r["base"]); loss = lambda key: b - md(r[mode][key])
        per_k = [0.0] + [np.mean([loss("".join(map(str, t))) for t in SUB if len(t) == k]) for k in range(1, 5)]
        obs.append(per_k); add.append([k * per_k[1] for k in range(5)])
    return np.array(obs), np.array(add)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
k = np.arange(5)
for a, rg, title in zip(ax, ["clean", "skip"], ["(a) clean-trained", "(b) skip-corruption-trained"]):
    for j, v in enumerate(V):
        obs, add = curves(v, rg); dx = (j - 1.5) * 0.06
        a.plot(k + dx, add.mean(0), ":", color=COL[v], lw=1.2)
        a.errorbar(k + dx, obs.mean(0), yerr=obs.std(0, ddof=1), color=COL[v], lw=2, marker=MK[v], ms=5,
                   mec="white", mew=0.8, capsize=0, elinewidth=1, label=LABEL[v])
    a.set_title(title, fontsize=8, loc="left"); a.set_xticks(k); a.set_xlabel("skip levels removed ($k$)")
    a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True); a.set_ylim(-0.02, 0.88)
ax[0].set_ylabel("Dice loss (validation)")
h, l = ax[1].get_legend_handles_labels()
h.append(plt.Line2D([], [], ls=":", color="#52514e", lw=1.2)); l.append("additive prediction")
ax[1].legend(h, l, fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("fig3_redundancy.pdf"); fig.savefig("fig3_redundancy.png", dpi=200)
print("wrote fig3_redundancy.pdf/.png")
