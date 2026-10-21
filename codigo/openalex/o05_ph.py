"""Paso o05: homología persistente de la red de acoplamiento del corpus sistemático.
- Filtración por fuerza (rango de significación) en cada ventana de 3 años y en la red completa, con nulo Curveball
  (recableado de la bipartita con los dos grados fijos, reproyección, los m pares más significativos; 100 réplicas).
- Filtración por fecha en la red completa, con nulo de fechas permutadas (200 réplicas).
Criterio como en h15: persistencia frente al máximo de cada réplica (percentil 95).
Reanudable. Salidas: out/o05_null_<clave>.npy, out/o05_summary.json, out/o05_bars_<clave>.csv"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import citnet_big as cb
from o04_network import OUT, WINDOWS, build

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import citnet as cn  # noqa: E402

N_NULL = 100
N_NULL_DATES = 200
SEED = 20261019


def bars(M):
    _, (h1,) = cn.h1_bars(M)
    return h1


def maxpers(h):
    return float((h[:, 1] - h[:, 0]).max()) if len(h) else 0.0


def run_null(key, fn, reps):
    path = os.path.join(OUT, f"o05_null_{key}.npy")
    res = list(np.load(path)) if os.path.exists(path) else []
    t0 = time.time()
    while len(res) < reps:
        res.append(fn())
        if len(res) % 10 == 0 or len(res) == reps:
            np.save(path, np.array(res))
            print(f"    {key}: {len(res)}/{reps} ({time.time() - t0:.0f} s)", flush=True)
    return np.array(res)


def summarize(h, null, label):
    pers = h[:, 1] - h[:, 0] if len(h) else np.array([])
    thr = float(np.quantile(null[:, 0], 0.95))
    pval = np.array([(1 + (null[:, 0] >= p).sum()) / (1 + len(null)) for p in pers])
    return dict(clave=label, n_bars=int(len(h)), max_pers=round(float(pers.max()), 4) if len(pers) else 0.0,
                thr95=round(thr, 4), n_sig=int((pval < 0.05).sum()), min_p=round(float(pval.min()), 4) if len(pval) else 1.0,
                null_nbars_median=float(np.median(null[:, 1])), q_nbars=round(float((null[:, 1] < len(h)).mean()), 3)), pval


def strength(R, sel, key, rng, reps):
    Bs = R[sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    k = np.asarray(Bs.sum(1)).ravel()
    p, hc, _ = cn.fit_bicm(Bs)
    a, b, s, pv = cb.validated(Bs, p, hc, k > 0)
    m = len(a)
    h = bars(cn.rank_filtration(a, b, pv, len(sel)))
    rows0 = cn.rows_of(Bs)

    def null():
        Bn = cn.from_rows(cn.curveball(rows0, rng), Bs.shape)
        na, nb, _, npv = cb.top_m(Bn, p, hc, k > 0, m)
        hn = bars(cn.rank_filtration(na, nb, npv, len(sel)))
        return [maxpers(hn), len(hn)]
    nul = run_null(key, null, reps)
    res, pval = summarize(h, nul, key)
    res.update(papers=int(len(sel)), active=int((k > 0).sum()), validated=int(m))
    pd.DataFrame({"birth": h[:, 0], "death": h[:, 1], "pers": h[:, 1] - h[:, 0], "p": pval}).sort_values(
        "pers", ascending=False).to_csv(os.path.join(OUT, f"o05_bars_{key}.csv"), index=False)
    return res


def main():
    Wk, R, t = build()
    yr = np.floor(t)
    rng = np.random.default_rng(SEED)
    path = os.path.join(OUT, "o05_summary.json")
    summ = json.load(open(path)) if os.path.exists(path) else {}
    for y0, y1 in WINDOWS:
        key = f"win{y0}"
        if key in summ:
            continue
        sel = np.where((yr >= y0) & (yr <= y1))[0]
        summ[key] = strength(R, sel, key, rng, N_NULL)
        summ[key]["ventana"] = f"{y0}-{y1}"
        print(summ[key], flush=True)
        json.dump(summ, open(path, "w"), indent=1, default=float)

    # filtración por fecha en la red completa (nulo de fechas permutadas)
    V = np.load(os.path.join(OUT, "val_full.npz"))
    a, b, te = V["a"], V["b"], V["tedge"]
    nodes = np.where(~np.isnan(t))[0]
    h = bars(cn.time_filtration(a, b, te, t, nodes))

    def null_dates():
        t2 = t.copy()
        t2[nodes] = rng.permutation(t[nodes])
        hn = bars(cn.time_filtration(a, b, np.maximum(t2[a], t2[b]), t2, nodes))
        return [maxpers(hn), len(hn)]
    nul = run_null("full_fecha", null_dates, N_NULL_DATES)
    res, pval = summarize(h, nul, "full_fecha")
    pd.DataFrame({"birth": h[:, 0] + 1900, "death": h[:, 1] + 1900, "years": h[:, 1] - h[:, 0], "p": pval}).sort_values(
        "years", ascending=False).to_csv(os.path.join(OUT, "o05_bars_full_fecha.csv"), index=False)
    summ["full_fecha"] = res
    print(res, flush=True)
    json.dump(summ, open(path, "w"), indent=1, default=float)

    # filtración por fuerza en la red completa (nulo Curveball)
    summ["full_fuerza"] = strength(R, np.where(~np.isnan(t))[0], "full_fuerza", rng, N_NULL)
    print(summ["full_fuerza"], flush=True)
    json.dump(summ, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
