import hashlib, json, os, sys
import numpy as np, pandas as pd

MANIFEST_SHA = "c9d3f5e2cc74f1cb8eb855effece996ead9034517894ff845189f0830a730aef"
OOD_SITE, SEED = "FG", 20261105            # fixed BEFORE any training; do not change

if os.path.exists("split.json"):
    sys.exit("split.json already exists. Refusing to overwrite. Move it away deliberately if you mean to replace it.")

raw = open("audit/manifest.csv", "rb").read()
assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "manifest.csv differs from the audited version. STOP and report."
m = pd.read_csv("audit/manifest.csv"); print(m.columns.tolist())
assert len(m) == 3929 and m["patient_id"].nunique() == 110

pat = m.groupby("patient_id").agg(site=("site", "first")).reset_index()
assert (m.groupby("patient_id")["site"].nunique() == 1).all()      # one site per patient
rng = np.random.default_rng(SEED)

split = {p: "test_ood" for p in pat[pat.site == OOD_SITE].patient_id}
for site, g in pat[pat.site != OOD_SITE].groupby("site"):          # stratified by site; EZ (1 patient) -> train
    ids = rng.permutation(g.patient_id.values); n = len(ids)
    n_te, n_va = int(round(.2 * n)), int(round(.1 * n))
    for p in ids[:n_te]: split[p] = "test_id"
    for p in ids[n_te:n_te + n_va]: split[p] = "val"
    for p in ids[n_te + n_va:]: split[p] = "train"

assert set(split) == set(pat.patient_id)                           # every patient exactly once
m["split"] = m["patient_id"].map(split)
assert m["split"].notna().all()

print(m.groupby("split").agg(patients=("patient_id", "nunique"), slices=("split", "size"),
                             empty_frac=("is_empty", "mean")))
print(pd.crosstab(m["split"], m["site"]))

# structural checks against the design (fail loudly)
cnt = m.groupby("split")["patient_id"].nunique().to_dict()
assert cnt == {"test_ood": 14, "test_id": 19, "train": 68, "val": 9}, cnt
assert int((m.split == "test_ood").sum()) == 640
assert set(m[m.split == "test_ood"].site) == {OOD_SITE}
assert not (m[m.split != "test_ood"].site == OOD_SITE).any()

json.dump(split, open("split.json", "w"), indent=1, sort_keys=True)
h = hashlib.sha256(open("split.json", "rb").read()).hexdigest()
open("split.sha256", "w").write(h + "  split.json\n")
print("split.json sha256:", h)

