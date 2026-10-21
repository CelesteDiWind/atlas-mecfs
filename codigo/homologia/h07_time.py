"""Paso 7: filtración por año (resultado previo 7), sin heredar distancias.
Para cada año t (acumulado: papers publicados hasta t, mínimo 40):
    - distancias recalculadas solo con ese subconjunto (cuerda completa y geodésica k15 rehecha en el subgrafo)
    - nulo gmm ajustado a ese subconjunto (n igual), 100 réplicas -> umbral de significación del año
    - control de tamaño: 40 submuestras aleatorias del corpus completo con el mismo n, evaluadas contra el
      mismo umbral -> cuántos ciclos "significativos" se esperan solo por el tamaño de muestra
    - curvas de Betti beta1(eps) y persistencia total
Salidas: out/year_filtration.csv, out/betti_curves.npz"""
import os
import pickle

import numpy as np
import pandas as pd

from common import MAIN, SEED, NullModel, finite, load_emb, load_papers, null_maxima, out, ratio, rips, space_distance

N_YEAR_NULL = 100
N_CONTROL = 40
EPS = np.linspace(0, 1.2, 241)


def betti_curve(dg, eps):
    d = finite(dg)
    return np.array([((d[:, 0] <= e) & (d[:, 1] > e)).sum() for e in eps])


def main():
    P = load_papers()["papers"]
    yr = np.array([p["y"] or 0 for p in P])
    E_all = load_emb(MAIN)
    rng = np.random.default_rng(SEED)
    rows, curves = [], {}
    path = out("year_filtration.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    for t in range(2008, int(yr.max()) + 1):
        idx = np.where((yr > 0) & (yr <= t))[0]
        if len(idx) < 40:
            continue
        if len(done) and ((done.year == t)).any():
            rows += done[done.year == t].to_dict("records")
            continue
        E = E_all[idx]
        nmp = out(f"nullmodel_year{t}.pkl")
        if os.path.exists(nmp):
            nm = pickle.load(open(nmp, "rb"))
        else:
            nm = NullModel(E, max_comp=10)
            pickle.dump(nm, open(nmp, "wb"))
        for sp in ("full", "k15"):
            D = space_distance(E, sp)
            dg = rips(D, 1)["dgms"][1]
            pers = finite(dg)[:, 1] - finite(dg)[:, 0]
            rat = ratio(dg)
            nul = null_maxima(E, sp, "gmm", nm, reps=N_YEAR_NULL, tag=f"year{t}_")[:, 1]  # columna 1 = cociente máximo
            tau = np.percentile(nul, 95)
            ctrl = []
            for _ in range(N_CONTROL):  # mismo n, papers de cualquier año
                j = np.sort(rng.choice(len(E_all), len(idx), replace=False))
                ctrl.append(int((ratio(rips(space_distance(E_all[j], sp), 1)["dgms"][1]) > tau).sum()))
            curves[f"{t}_{sp}"] = betti_curve(dg, EPS)
            rows.append(dict(year=t, space=sp, n=len(idx), n_bars=len(pers), h1_total=round(pers.sum(), 4),
                             h1_total_per_paper=round(pers.sum() / len(idx), 5), h1_max=round(pers.max(), 4),
                             tau_gmm=round(tau, 4), n_sig=int((rat > tau).sum()),
                             control_mean=round(np.mean(ctrl), 2), control_p95=float(np.percentile(ctrl, 95)),
                             n_gmm_components=nm.n_comp))
            print(f"{t} {sp:4s} n={len(idx):3d} ciclos sig={rows[-1]['n_sig']} (control mismo n: media "
                  f"{rows[-1]['control_mean']}, p95 {rows[-1]['control_p95']}) | pers. total/paper {rows[-1]['h1_total_per_paper']}",
                  flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    np.savez(out("betti_curves.npz"), eps=EPS, **curves)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
