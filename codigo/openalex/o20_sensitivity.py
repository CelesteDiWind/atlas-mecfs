"""Paso o20: ¿cambian las conclusiones si solo se usan los trabajos que tratan de verdad de la EM/SFC?
Subconjunto = relevancia central o secundaria (o18-o19), sin retractados. Se repiten, con los mismos métodos:
  - diálogo entre temas: O/E de acoplamiento solo entre temas (cuasi-independencia, o14) en la red restringida
  - mezcla entre temas por ventana (vínculos validados de o04 entre papers del subconjunto, frente a permutar temas)
  - cohesión de la literatura citada por año (o06)
Salida: out/o20_sensitivity.json"""
import json
import os

import numpy as np
from scipy.sparse import load_npz

from o04_network import OUT, WINDOWS
from o14_dialogue import quasi_indep, sym_counts

SEED = 20261022
K = 8


def main():
    rng = np.random.default_rng(SEED)
    m = json.load(open(os.path.join(os.path.dirname(os.path.dirname(OUT)), "..", "viewer", "oa", "oa_map.json"), encoding="utf-8"))
    P = m["papers"]
    lab = np.array([p["c"] for p in P]); yr = np.array([p["y"] for p in P])
    keep = np.array([(p.get("r", 0) <= 1) and not p.get("rt") for p in P])
    f = m["edges"]["cp"]; a, b = np.array(f[0::4], int), np.array(f[1::4], int)
    res = {"n_all": int(len(P)), "n_subset": int(keep.sum())}
    for name, sel in (("todos", np.ones(len(P), bool)), ("solo_em_sfc", keep)):
        e = sel[a] & sel[b]
        oe, _, z = quasi_indep(sym_counts(a[e], b[e], lab), False)
        pairs = sorted(((float(oe[i, j]), i, j) for i in range(K) for j in range(i + 1, K)), reverse=True)
        res[name] = dict(edges=int(e.sum()), cp_ext=np.round(oe, 3).tolist(),
                         top=[[i, j, round(v, 2)] for v, i, j in pairs[:5]], bottom=[[i, j, round(v, 2)] for v, i, j in pairs[-5:]])
        # mezcla entre temas por ventana
        mix = []
        for y0, y1 in WINDOWS:
            V = np.load(os.path.join(OUT, f"val_win_{y0}.npz"))
            wa, wb = V["a"], V["b"]
            ee = sel[wa] & sel[wb]
            if ee.sum() < 50:
                mix.append(None); continue
            wa, wb = wa[ee], wb[ee]
            nodes = np.unique(np.r_[wa, wb])
            obs = (lab[wa] != lab[wb]).mean()
            perm = []
            for _ in range(200):
                l2 = lab.copy(); l2[nodes] = rng.permutation(lab[nodes])
                perm.append((l2[wa] != l2[wb]).mean())
            mix.append(round(float(obs / np.mean(perm)), 3))
        res[name]["mixing"] = dict(zip([f"{y0}-{y1}" for y0, y1 in WINDOWS], mix))
    # cohesión por año en el subconjunto (misma medida que o06: coseno entre perfiles de referencias de dos mitades de 24)
    R = load_npz(os.path.join(OUT, "bip_refs.npz")).astype(np.float64).tocsr()
    k = np.asarray(R.sum(1)).ravel()
    coh = {}
    for y in range(2010, 2027):
        for name, sel in (("todos", np.ones(len(P), bool)), ("solo_em_sfc", keep)):
            idx = np.where((yr == y) & (k > 0) & sel)[0]
            if len(idx) < 48:
                continue
            v = []
            for _ in range(300):
                p_ = rng.permutation(idx)
                x, z_ = np.asarray(R[p_[:24]].sum(0)).ravel(), np.asarray(R[p_[24:48]].sum(0)).ravel()
                v.append(x @ z_ / np.linalg.norm(x) / np.linalg.norm(z_))
            coh.setdefault(str(y), {})[name] = round(float(np.mean(v)), 3)
    res["cohesion"] = coh
    json.dump(res, open(os.path.join(OUT, "o20_sensitivity.json"), "w"), indent=1)
    names = [c["name"][:22] for c in m["clusters"]]
    for name in ("todos", "solo_em_sfc"):
        print(f"\n{name}: {res[name]['edges']} vínculos")
        print("  más diálogo:", [(names[i], names[j], v) for i, j, v in res[name]["top"][:3]])
        print("  menos diálogo:", [(names[i], names[j], v) for i, j, v in res[name]["bottom"][-3:]])
        print("  mezcla por ventana:", res[name]["mixing"])
    print("\ncohesión:", coh)


if __name__ == "__main__":
    main()
