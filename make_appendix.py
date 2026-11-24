"""Generate appendix.tex from the result files, so appendix numbers cannot drift from the data.
Inputs: split.json, audit/manifest.csv, gate_probe_val.json, subset_probe_val.json, runs logs (ns*.txt).
SD is the sample SD over the 3 seeds (ddof=1), as in the main text.
"""
import csv, itertools, json, re
import numpy as np

G = json.load(open("gate_probe_val.json")); S = json.load(open("subset_probe_val.json"))
V = ["concat", "static", "arf", "arf_nodiff"]; GATED = ["static", "arf", "arf_nodiff"]; ARF = ["arf", "arf_nodiff"]
NAME = {"concat": "concat", "static": "static", "arf": "ARF", "arf_nodiff": "ARF-nodiff"}
K = ["roll", "noise", "shift", "blur", "chdrop", "zero"]
KN = {"roll": "roll$^\\dagger$", "noise": "noise$^\\dagger$", "shift": "shift", "blur": "blur", "chdrop": "ch.\\ drop", "zero": "zero"}
SEEDS = [0, 1, 2]; L4 = range(4)
SUB = [s for k in range(1, 5) for s in itertools.combinations(L4, k)]; SN = lambda s: "".join(map(str, s))
md = lambda d: float(np.mean(list(d.values())))
sd = lambda x: float(np.std(x, ddof=1))
f3 = lambda x: (f"{abs(x):.3f}".replace("0.", ".", 1) if round(x, 3) >= 0 or round(x, 3) == 0 else "$-$" + f"{-x:.3f}".replace("0.", ".", 1))
ms = lambda x: f"{f3(np.mean(x))}$\\pm${f3(sd(x)).replace('$-$', '')}"
key = lambda v, r, s: f"{v}_{r}_s{s}"
out = []; W = out.append

# ---------------- A. data and protocol ----------------
split = json.load(open("split.json")); rows = list(csv.DictReader(open("audit/manifest.csv")))
pid2split = {}
for r in rows:
    p = r["patient_id"]; m = [k for k in split if k.startswith(p) or p.startswith(k)]
    assert len(m) == 1, (p, m); pid2split[p] = split[m[0]]
agg = {}
for r in rows:
    sp = pid2split[r["patient_id"]]; a = agg.setdefault(sp, dict(p=set(), sites=set(), n=0, e=0))
    a["p"].add(r["patient_id"]); a["sites"].add(r["site"]); a["n"] += 1; a["e"] += r["is_empty"] == "True"
ns = [float(re.findall(r"best ([0-9.]+)", open(f"ns{s}.txt").read())[-1]) for s in SEEDS]

W(r"""\section{Data, protocol and compute}\label{app:protocol}
\begin{table}[h]
\centering\small
\caption{Patient-level split. Sites are patient-ID prefixes; site FG is held out entirely as the OOD test set.}\label{tab:app_split}
\begin{tabular}{@{}lrrrl@{}}
\toprule
Split & Patients & Slices & Empty-mask slices & Sites\\
\midrule""")
for sp, lab in [("train", "train"), ("val", "validation"), ("test_id", "ID test"), ("test_ood", "OOD test")]:
    a = agg[sp]
    W(f"{lab} & {len(a['p'])} & {a['n']} & {a['e']} ({100*a['e']/a['n']:.0f}\\%) & {', '.join(sorted(a['sites']))}\\\\")
tot_n = sum(a["n"] for a in agg.values()); tot_e = sum(a["e"] for a in agg.values())
W(f"\\midrule\nall & {sum(len(a['p']) for a in agg.values())} & {tot_n} & {tot_e} ({100*tot_e/tot_n:.0f}\\%) & \\\\")
W(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering\small
\caption{Training protocol (identical for all operators and regimes) and settings of the post-hoc probes.}\label{tab:app_protocol}
\begin{tabular}{@{}ll@{}}
\toprule
Item & Setting\\
\midrule
Backbone & 4-level U-Net, widths 32/64/128/256, bottleneck 512, GroupNorm (8 groups)\\
Input & FLAIR channel, per-image $z$-score over nonzero pixels, $256{\times}256$\\
Optimizer & AdamW, lr $3{\times}10^{-4}$, weight decay $10^{-4}$, no scheduler, grad.\ clip 1.0\\
Batch, epochs & 8, 80; checkpoint with best clean validation Dice (per-patient)\\
Augmentation & horizontal flip only (both regimes)\\
Loss & $0.5\,$BCE $+\,0.5\,$Dice (Dice pooled over the batch)\\
Skip regime & per sample and level: keep 0.70, roll 0.15, noise $\mathcal N(0,\mathrm{std}(E)^2)$ 0.15\\
Threshold & logit $>0$; Dice per patient from summed TP/FP/FN over its slices\\
\midrule
Probe batches & validation slices, fixed shuffle (seed 777), $\approx$32 slices per batch\\
Shift / blur & circular shift by $(H/8, W/8)$ / $5{\times}5$ average pooling\\
Channel drop / noise & random half of channels zeroed (no rescaling) / as in training, fixed generator seed\\
Gate closed & fusion output replaced by $D'$ at the chosen levels ($\alpha:=0$)\\
\midrule
Hardware & 1$\times$NVIDIA L40S; 3 runs concurrently; 0.6--1.0\,h per run\\
\bottomrule
\end{tabular}
\end{table}
""")
W(f"\\noindent\\textbf{{No-skip baseline.}} Best validation Dice per seed: {ns[0]:.4f}, {ns[1]:.4f}, {ns[2]:.4f} "
  f"(mean {np.mean(ns):.3f}, SD {sd(ns):.3f}); same protocol, clean regime, 80 epochs.\n")

# ---------------- B. unseen perturbations ----------------
def rel(k, kind, L):
    b = md(G[k]["clean"]); return (b - md(G[k]["corr"][kind][str(L)]["dice"])) / b
W(r"""\clearpage
\section{Complete post-hoc probe results}\label{app:probes}
All values are on the validation set (9 patients), mean $\pm$ SD over 3 seeds, and exploratory. $^\dagger$\,Perturbation used in skip-corruption training.

\begin{table}[h]
\centering\footnotesize
\caption{Relative Dice loss when the skip at level $\ell$ alone is perturbed (the skip reliance index on the validation set, extended to unseen perturbations).}\label{tab:app_unseen}
\begin{tabular}{@{}lllcccc@{}}
\toprule
Operator & Regime & Perturbation & $\ell=0$ & $\ell=1$ & $\ell=2$ & $\ell=3$\\
\midrule""")
for v in V:
    for r in ["clean", "skip"]:
        for kind in K:
            cells = [ms([rel(key(v, r, s), kind, L) for s in SEEDS]) for L in L4]
            W(f"{NAME[v] if kind == K[0] else ''} & {r if kind == K[0] else ''} & {KN[kind]} & " + " & ".join(cells) + "\\\\")
        if not (v == V[-1] and r == "skip"): W("\\addlinespace[2pt]")
W("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- C. gate response ----------------
W(r"""\begin{table}[h]
\centering\footnotesize
\caption{Response of the ARF gate at the perturbed level. Each cell: change in the per-sample mean of $\alpha_\ell$ (perturbed minus clean), with the fraction of samples in which it fell in brackets; below it, the element-wise mean $|\Delta\alpha_\ell|$. Reference row: within-image SD of the clean $\alpha_\ell$.}\label{tab:app_gate}
\begin{tabular}{@{}lllcccc@{}}
\toprule
Operator & Regime & Perturbation & $\ell=0$ & $\ell=1$ & $\ell=2$ & $\ell=3$\\
\midrule""")
for v in ARF:
    for r in ["clean", "skip"]:
        ref = [np.mean([S[key(v, r, s)]["pixel"]["roll"][str(L)]["within_img_sd_clean"] for s in SEEDS]) for L in L4]
        W(f"{NAME[v]} & {r} & \\emph{{ref.\\ SD}} & " + " & ".join(f"{x:.3f}" for x in ref) + "\\\\")
        for kind in K:
            c1 = []; c2 = []
            for L in L4:
                da = [G[key(v, r, s)]["corr"][kind][str(L)]["dalpha"] for s in SEEDS]
                fd = np.mean([G[key(v, r, s)]["corr"][kind][str(L)]["frac_down"] for s in SEEDS])
                px = np.mean([S[key(v, r, s)]["pixel"][kind][str(L)]["mean_abs"] for s in SEEDS])
                sgn = "+" if np.mean(da) >= 0 else "$-$"
                c1.append(f"{sgn}{abs(np.mean(da)):.4f} [{fd:.2f}]"); c2.append(f"{px:.3f}")
            W(f" & & {KN[kind]} & " + " & ".join(c1) + "\\\\")
            W(" & & & " + " & ".join(f"\\textcolor{{gray}}{{{c}}}" for c in c2) + "\\\\")
        if not (v == ARF[-1] and r == "skip"): W("\\addlinespace[3pt]")
W("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- D. gate regret ----------------
W(r"""\begin{table}[h]
\centering\footnotesize
\caption{Gate regret: Dice with the gate at level $\ell$ closed minus Dice with the skip at $\ell$ perturbed. Positive values mean closing the gate would have helped. The static gate cannot respond to its input.}\label{tab:app_regret}
\begin{tabular}{@{}lllcccc@{}}
\toprule
Operator & Regime & Perturbation & $\ell=0$ & $\ell=1$ & $\ell=2$ & $\ell=3$\\
\midrule""")
for v in GATED:
    for r in ["clean", "skip"]:
        for kind in K:
            cells = [ms([md(G[key(v, r, s)]["oracle"][str(L)]) - md(G[key(v, r, s)]["corr"][kind][str(L)]["dice"]) for s in SEEDS]) for L in L4]
            W(f"{NAME[v] if kind == K[0] else ''} & {r if kind == K[0] else ''} & {KN[kind]} & " + " & ".join(cells) + "\\\\")
        if not (v == GATED[-1] and r == "skip"): W("\\addlinespace[2pt]")
W("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- E. subset removal ----------------
for mode, cap in [("zero", "skips at the listed levels zeroed (all operators)"),
                  ("shut", "gates at the listed levels closed, $\\alpha:=0$ (gated operators)")]:
    vs = V if mode == "zero" else GATED
    W(f"""\\begin{{table}}[h]
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}
\\caption{{Absolute Dice loss (mean over 3 seeds) for every removed subset of levels: {cap}.}}\\label{{tab:app_subset_{mode}}}
\\begin{{tabular}}{{@{{}}ll{'c' * len(SUB)}@{{}}}}
\\toprule
Operator & Regime & """ + " & ".join(f"\\{{{','.join(map(str, s))}\\}}" for s in SUB) + "\\\\\n\\midrule")
    for v in vs:
        for r in ["clean", "skip"]:
            cells = [f3(np.mean([md(S[key(v, r, s)]["base"]) - md(S[key(v, r, s)][mode][SN(t)]) for s in SEEDS])) for t in SUB]
            W(f"{NAME[v]} & {r} & " + " & ".join(cells) + "\\\\")
    W("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- F. per-seed removal ----------------
W(r"""\begin{table}[h]
\centering\footnotesize
\caption{Per-seed validation Dice: unperturbed, with all skips zeroed or all gates closed, and the worst single-level removal (level in brackets). Shows which configurations rest on single-seed failures.}\label{tab:app_seed}
\begin{tabular}{@{}llccccc@{}}
\toprule
Operator & Regime & Seed & base & all zeroed & all closed & worst single (zero / closed)\\
\midrule""")
for v in V:
    for r in ["clean", "skip"]:
        for s in SEEDS:
            R = S[key(v, r, s)]; b = md(R["base"])
            wz = min(L4, key=lambda L: md(R["zero"][str(L)]))
            cz = f"{f3(md(R['zero'][str(wz)]))} [{wz}]"
            if v in GATED:
                wsh = min(L4, key=lambda L: md(R["shut"][str(L)]))
                sh_all = f3(md(R["shut"]["0123"])); csh = f"{f3(md(R['shut'][str(wsh)]))} [{wsh}]"
            else:
                sh_all = "--"; csh = "--"
            W(f"{NAME[v] if s == 0 else ''} & {r if s == 0 else ''} & {s} & {f3(b)} & {f3(md(R['zero']['0123']))} & {sh_all} & {cz} / {csh}\\\\")
        if not (v == V[-1] and r == "skip"): W("\\addlinespace[2pt]")
W("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

open("appendix.tex", "w").write("% AUTO-GENERATED by make_appendix.py -- do not edit by hand; rerun the script.\n" + "\n".join(out) + "\n")

# Table 4 of the main text, recomputed with ddof=1 for consistency
print("Table 4 (ddof=1):")
for v in GATED:
    print(NAME[v], " & ".join(ms([md(G[key(v, 'clean', s)]["oracle"][str(L)]) - md(G[key(v, 'clean', s)]["corr"][k][str(L)]["dice"]) for s in SEEDS])
                                  for k in ["roll", "shift"] for L in [0, 1]))
print("split:", {k: (len(a['p']), a['n'], a['e']) for k, a in agg.items()})
