"""Paso o07: dos episodios que aparecen en el corpus sistemático y no en el manual.
A. Escalón de continuidad 2018-20 -> 2019-21 (Jaccard de vínculos entre papers compartidos: 0.89 -> 0.74).
   - ¿Qué pasa con los vínculos: se pierden o se ganan?
   - ¿Desaparece al excluir los papers de COVID persistente de las dos ventanas?
   - Bootstrap de las referencias (columnas con reemplazo) sobre las ventanas 2017-19 ... 2020-22: ¿queda fuera del ruido?
B. Cohesión alta en 2010-11 frente a 2012+: referencias sobrerrepresentadas en 2010-11 frente a 2013-16.
Salidas: out/o07_summary.json, out/o07_refs_2010_11.csv"""
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

import citnet_big as cb
from o04_network import OUT, build

import citnet as cn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
spec = importlib.util.spec_from_file_location("enrich", os.path.join(ROOT, "scripts", "02_enrich.py"))
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)
SEED = 20261019
COVID = re.compile(r"covid|sars-cov-2|post-acute sequelae|pasc\b|long[- ]haul", re.I)
N_BOOT = 40


def edges(R, sel):
    Bs = R[sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    k = np.asarray(Bs.sum(1)).ravel()
    p, hc, _ = cn.fit_bicm(Bs)
    a, b, s, pv = cb.validated(Bs, p, hc, k > 0)
    return set(zip(sel[a].tolist(), sel[b].tolist()))


def jaccard(E0, S0, E1, S1):
    sh = S0 & S1
    e0 = {e for e in E0 if e[0] in sh and e[1] in sh}
    e1 = {e for e in E1 if e[0] in sh and e[1] in sh}
    return len(e0 & e1) / max(1, len(e0 | e1)), len(e0 & e1), len(e0 - e1), len(e1 - e0)


def main():
    W, R, t = build()
    yr = np.floor(t)
    covid = np.array([bool(COVID.search((w["title"] or "") + " " + (w["abstract"] or ""))) for w in W])
    rng = np.random.default_rng(SEED + 7)
    win = lambda y0: np.where((yr >= y0) & (yr <= y0 + 2))[0]
    S = {}

    # ---------- A. descomposición y control sin COVID
    for y0 in (2017, 2018, 2019, 2020):
        a, b = win(y0), win(y0 + 1)
        j, com, lost, new = jaccard(edges(R, a), set(a.tolist()), edges(R, b), set(b.tolist()))
        a2, b2 = a[~covid[a]], b[~covid[b]]
        j2, com2, lost2, new2 = jaccard(edges(R, a2), set(a2.tolist()), edges(R, b2), set(b2.tolist()))
        S[f"{y0}-{y0+2}->{y0+1}-{y0+3}"] = dict(jaccard=round(j, 3), comunes=com, perdidos=lost, nuevos=new,
                                              jaccard_sin_covid=round(j2, 3), comunes_sin_covid=com2, perdidos_sin_covid=lost2, nuevos_sin_covid=new2)
        print(y0, S[f"{y0}-{y0+2}->{y0+1}-{y0+3}"], flush=True)

    # ---------- A. bootstrap de referencias
    ncol = R.shape[1]
    Rc = R.tocsc()
    boot = []
    for r in range(N_BOOT):
        Rb = Rc[:, rng.integers(0, ncol, ncol)].tocsr()
        E = {y0: edges(Rb, win(y0)) for y0 in range(2016, 2021)}
        boot.append([jaccard(E[y0], set(win(y0).tolist()), E[y0 + 1], set(win(y0 + 1).tolist()))[0] for y0 in range(2016, 2020)])
        if (r + 1) % 10 == 0:
            print(f"  bootstrap {r + 1}/{N_BOOT}", flush=True)
    boot = np.array(boot)           # columnas: 2016-18->17-19, 17-19->18-20, 18-20->19-21, 19-21->20-22
    others = boot[:, [0, 1]].mean(1)
    diff = boot[:, 2] - others
    S["bootstrap"] = dict(transiciones=["2016-18->2017-19", "2017-19->2018-20", "2018-20->2019-21", "2019-21->2020-22"],
                          media=np.round(boot.mean(0), 3).tolist(),
                          ci95=[[round(float(np.quantile(boot[:, i], q)), 3) for q in (0.025, 0.975)] for i in range(4)],
                          diff_2018_20_vs_previas=round(float(diff.mean()), 3),
                          diff_ci95=[round(float(np.quantile(diff, q)), 3) for q in (0.025, 0.975)],
                          frac_lowest=round(float((boot.argmin(1) == 2).mean()), 3))
    print(S["bootstrap"], flush=True)

    # ---------- B. 2010-11
    ref_ids = sorted({r for w in W for r in w["refs"]})
    A = ((yr >= 2010) & (yr <= 2011) & (R.sum(1).A1 > 0))
    B = ((yr >= 2013) & (yr <= 2016) & (R.sum(1).A1 > 0))
    fa, fb = R[A].sum(0).A1 / A.sum(), R[B].sum(0).A1 / B.sum()
    top = np.argsort(-(fa - fb))[:25]
    ids = [ref_ids[j] for j in top]
    titles = {}
    for i in range(0, len(ids), 25):
        d = enrich.cached_get("https://api.openalex.org/works", {"filter": "openalex_id:" + "|".join(ids[i:i + 25]),
                                                                  "select": "id,display_name,publication_year", "per_page": 50})
        for w in (d or {}).get("results", []):
            titles[w["id"].rsplit("/", 1)[-1]] = (w["publication_year"], w["display_name"])
    rows = [dict(ref=ids[n], year=titles.get(ids[n], (None, "?"))[0], title=titles.get(ids[n], (None, "?"))[1],
                 frac_2010_11=round(float(fa[j]), 3), frac_2013_16=round(float(fb[j]), 3)) for n, j in enumerate(top)]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "o07_refs_2010_11.csv"), index=False)
    S["papers_2010_11"] = int(A.sum())
    json.dump(S, open(os.path.join(OUT, "o07_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
