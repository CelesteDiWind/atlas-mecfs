"""Paso 5: H1 y H2 con filtración completa, tres nulos y estabilidad por submuestreo (resultados previos 3 y 4).
    - Real: Rips sobre cada espacio (cuerda completa y geodésicas k6/k15/k30), H1 sin umbral.
      H2 (exploratorio) con 200 puntos de referencia por permutación voraz, igual en real y nulo.
    - Nulos (N_NULL réplicas cada uno, estadístico máximo -> control del falso positivo por diagrama):
      gauss, colperm y gmm (ver common.NullModel). El criterio principal es gmm: conserva los grupos.
    - Sensibilidad (solo SPECTER2): subconjuntos 'clean' y 'with_abstract', espacios full y k15.
    - Estabilidad (SPECTER2, full y k15): 50 submuestras del 80 %; una barra significativa "reaparece" si el
      emparejamiento bottleneck la asocia a una barra con al menos la mitad de su persistencia.
Salidas: out/dgm_*.pkl, out/null_*.npy, out/h1_summary.csv, out/h1_bars_specter2_all_{espacio}.csv"""
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (KS, MAIN, MODELS, N_NULL, N_SUB, SEED, SUB_FRAC, NullModel, col, finite, load_emb, null_maxima,
                    out, rips, significance, space_distance, stable_hash)

SPACES = ("full",) + tuple(f"k{k}" for k in KS)
KINDS = ("gauss", "colperm", "gmm")


def configs():
    for m in MODELS:
        if not os.path.exists(out(f"E_{m}.npy")):
            continue
        for sub in (("all", "clean", "with_abstract") if m == MAIN else ("all",)):
            for sp in (SPACES if sub == "all" else ("full", "k15")):
                dims = (1, 2) if (m == MAIN and sub == "all" and sp in ("full", "k15")) else (1,)
                yield m, sub, sp, dims


def subset_E(m, sub):
    with open(out("subsets.json")) as fh:
        idx = json.load(fh)[sub]
    return load_emb(m)[idx], np.array(idx)


def get_null_model(m, sub):
    path = out(f"nullmodel_{m}_{sub}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    nm = NullModel(subset_E(m, sub)[0])
    with open(path, "wb") as fh:
        pickle.dump(nm, fh)
    return nm


def null_job(args):
    m, sub, sp, kind, dims = args
    with open(out(f"nullmodel_{m}_{sub}.pkl"), "rb") as fh:
        nm = pickle.load(fh)
    E, _ = subset_E(m, sub)
    null_maxima(E, sp, kind, nm, reps=N_NULL, dims=dims, tag=f"{m}_{sub}_")
    return args


def real_dgms(m, sub, sp, dims):
    path = out(f"dgm_{m}_{sub}_{sp}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    E, _ = subset_E(m, sub)
    D = space_distance(E, sp)
    R = rips(D, 1)
    dg = {0: R["dgms"][0], 1: R["dgms"][1]}
    if 2 in dims:
        dg[2] = rips(D, 2, n_perm=200)["dgms"][2]
    with open(path, "wb") as fh:
        pickle.dump(dg, fh)
    return dg


def match_rate(real_sig, E, sp, rng):
    """Fracción de submuestras en las que reaparece cada barra significativa (emparejamiento bottleneck)."""
    from persim import bottleneck
    n = len(E)
    hits = np.zeros(len(real_sig))
    for r in range(N_SUB):
        idx = np.sort(rng.choice(n, int(SUB_FRAC * n), replace=False))
        sub = finite(rips(space_distance(E[idx], sp), 1)["dgms"][1])
        if not len(sub):
            continue
        _, M = bottleneck(real_sig, sub, matching=True)
        M = np.atleast_2d(np.asarray(M))
        for i, j, *_ in M:
            i, j = int(i), int(j)
            if 0 <= i < len(real_sig) and 0 <= j < len(sub):
                if sub[j, 1] - sub[j, 0] >= 0.5 * (real_sig[i, 1] - real_sig[i, 0]):
                    hits[i] += 1
    return hits / N_SUB


def main():
    cfgs = list(configs())
    for m, sub in sorted({(c[0], c[1]) for c in cfgs}):
        get_null_model(m, sub)
        print("modelo nulo", m, sub, "listo", flush=True)
    jobs = [(m, sub, sp, kind, dims) for m, sub, sp, dims in cfgs for kind in KINDS]
    with Pool(4) as pool:
        for k, a in enumerate(pool.imap_unordered(null_job, jobs), 1):
            print(f"  nulo {k}/{len(jobs)}: {a[:4]}", flush=True)

    rows = []
    for m, sub, sp, dims in cfgs:
        dg = real_dgms(m, sub, sp, dims)
        for dim in dims:
            bars = finite(dg[dim])
            bars = bars[bars[:, 0] > 0]
            pers = bars[:, 1] - bars[:, 0]
            rat = bars[:, 1] / bars[:, 0]
            row = dict(model=m, subset=sub, space=sp, dim=dim, n_bars=len(bars),
                       top5_ratio=[round(float(x), 4) for x in np.sort(rat)[::-1][:5]],
                       top5_pers=[round(float(x), 4) for x in np.sort(pers)[::-1][:5]], total_pers=round(float(pers.sum()), 4))
            pvals = {}
            for kind in KINDS:
                nul = np.load(out(f"null2_{m}_{sub}_{sp}_{kind}.npy"))
                tau, p = significance(rat, nul[:, col(dims, dim, "ratio")])        # criterio principal
                taua, _ = significance(pers, nul[:, col(dims, dim, "pers")])       # secundario (aditivo)
                nb, tot = nul[:, col(dims, dim, "nbars")], nul[:, col(dims, dim, "total")]
                row.update({f"tau_{kind}": round(tau, 4), f"nsig_{kind}": int((rat > tau).sum()),
                            f"tau_add_{kind}": round(taua, 4), f"nsig_add_{kind}": int((pers > taua).sum()),
                            # ¿menos ciclos que el azar? cuantil del real en la distribución nula
                            f"q_nbars_{kind}": round(float((nb < len(bars)).mean()), 3),
                            f"q_total_{kind}": round(float((tot < pers.sum()).mean()), 3)})
                pvals[kind] = p
            rows.append(row)
            if m == MAIN and sub == "all" and dim == 1:
                B = pd.DataFrame({"birth": bars[:, 0], "death": bars[:, 1], "pers": pers, "ratio": rat,
                                  **{f"p_{k}": v for k, v in pvals.items()}}).sort_values("ratio", ascending=False)
                sig = B[B.p_gmm < 0.05]
                if sp in ("full", "k15") and len(sig):
                    E, _ = subset_E(m, sub)
                    rr = np.random.default_rng([SEED, stable_hash("sub", sp)])
                    B.loc[sig.index, "reappear"] = match_rate(sig[["birth", "death"]].values, E, sp, rr)
                B.to_csv(out(f"h1_bars_{m}_{sub}_{sp}.csv"), index=False)
        print(f"{m:8s} {sub:13s} {sp:4s} " + " | ".join(
            f"H{r['dim']}: {r['n_bars']} barras, sig (muerte/nac.) gauss/colperm/gmm = {r['nsig_gauss']}/{r['nsig_colperm']}/{r['nsig_gmm']}"
            f" | cuantil n.º barras vs gmm {r['q_nbars_gmm']}"
            for r in rows if (r["model"], r["subset"], r["space"]) == (m, sub, sp)), flush=True)
    pd.DataFrame(rows).to_csv(out("h1_summary.csv"), index=False)


if __name__ == "__main__":
    main()
