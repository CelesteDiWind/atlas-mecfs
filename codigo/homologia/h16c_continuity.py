"""Paso 16c: ¿es real la caída de continuidad entre 2020-22 y 2021-23?
Continuidad entre ventanas consecutivas, medida sobre los papers que comparten:
    - ARI entre las comunidades de Louvain de cada ventana
    - Jaccard de vínculos: de los vínculos validados entre papers compartidos (en una u otra ventana), cuántos lo están
      en las dos (no depende de Louvain)
Nulo/incertidumbre: bootstrap de las referencias (columnas de la bipartita, con reemplazo, el mismo remuestreo para
todas las ventanas de una réplica). Para cada réplica se recalculan todas las ventanas y se mide la diferencia entre la
transición candidata y la media de las demás: la caída es real si el intervalo del 95 % de esa diferencia excluye el 0.
Variante: lo mismo excluyendo los papers de COVID persistente de los papers compartidos.
Salidas: out/h16c_continuity.csv (por transición), out/h16c_boot.npy, out/h16c_summary.json"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import adjusted_rand_score as ARI

import citnet as cn
from common import SEED, load_papers, out
from h14_compare import WINDOWS
from h16_convergence import COVID

N_BOOT = 100
CAND = 3   # transición 2020-22 -> 2021-23 (índice de la ventana de llegada - 1)


def window_net(R, sel, seed):
    Bs = R[sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    ks = np.asarray(Bs.sum(1)).ravel()
    p, hc, _ = cn.fit_bicm(Bs)
    wa, wb, _, wpv = cn.validated(cn.project(Bs), cn.expected_shared(p, hc), ks > 0)
    ga, gb = sel[wa], sel[wb]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(ga.tolist(), gb.tolist(), -np.log10(np.maximum(wpv, 1e-300))))
    lab = {v: c for c, vs in enumerate(nx.community.louvain_communities(G, weight="weight", seed=seed)) for v in vs}
    return set(zip(ga.tolist(), gb.tolist())), lab


def transitions(nets, sels, exclude=frozenset()):
    res = []
    for k in range(len(nets) - 1):
        (E0, L0), (E1, L1) = nets[k], nets[k + 1]
        shared = (set(sels[k].tolist()) & set(sels[k + 1].tolist())) - exclude
        common = sorted(set(L0) & set(L1) & shared)
        ari = ARI([L0[v] for v in common], [L1[v] for v in common]) if len(common) > 10 else np.nan
        e0 = {e for e in E0 if e[0] in shared and e[1] in shared}
        e1 = {e for e in E1 if e[0] in shared and e[1] in shared}
        jac = len(e0 & e1) / max(1, len(e0 | e1))
        res.append((ari, jac, len(common), len(e0 | e1)))
    return res


def main():
    J = load_papers()
    P = J["papers"]
    covid = {i for i, p in enumerate(P) if COVID.search(p["t"] + " " + (p.get("abs") or "") + " " + " ".join(p.get("kw") or []))}
    yr = np.floor(np.load(out("cit_dates.npz"))["paper"])
    R = load_npz(out("bip_refs.npz")).astype(np.int32).tocsc()
    sels = [np.where((yr >= y0) & (yr <= y1))[0] for y0, y1 in WINDOWS]
    names = [f"{WINDOWS[k][0]}-{WINDOWS[k][1]} -> {WINDOWS[k + 1][0]}-{WINDOWS[k + 1][1]}" for k in range(len(WINDOWS) - 1)]

    # observado (media de 5 semillas de Louvain para el ARI)
    obs = []
    for s in range(5):
        nets = [window_net(R.tocsr(), sel, SEED + s) for sel in sels]
        obs.append((transitions(nets, sels), transitions(nets, sels, frozenset(covid))))
    rows = []
    for k, nm in enumerate(names):
        rows.append(dict(transicion=nm, ari=np.nanmean([o[0][k][0] for o in obs]), jaccard=obs[0][0][k][1],
                         common=obs[0][0][k][2], edges_shared_union=obs[0][0][k][3],
                         ari_sin_covid=np.nanmean([o[1][k][0] for o in obs]), jaccard_sin_covid=obs[0][1][k][1],
                         common_sin_covid=obs[0][1][k][2]))
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)

    # bootstrap de referencias
    path = out("h16c_boot.npy")
    boot = list(np.load(path)) if os.path.exists(path) else []
    rng = np.random.default_rng(SEED + 163 + len(boot))
    ncol = R.shape[1]
    while len(boot) < N_BOOT:
        Rb = R[:, rng.integers(0, ncol, ncol)].tocsr()
        nets = [window_net(Rb, sel, SEED + len(boot)) for sel in sels]
        t, tc = transitions(nets, sels), transitions(nets, sels, frozenset(covid))
        boot.append([[x[0], x[1]] for x in t] + [[x[0], x[1]] for x in tc])
        if len(boot) % 10 == 0:
            np.save(path, np.array(boot))
            print(f"  bootstrap {len(boot)}/{N_BOOT}", flush=True)
    boot = np.array(boot)  # réplicas x (5 transiciones + 5 sin covid) x (ari, jaccard)
    np.save(path, boot)
    nt = len(names)
    summ = {"transiciones": rows, "candidata": names[CAND]}
    for part, off in (("todos", 0), ("sin_covid", nt)):
        for m, mname in enumerate(("ari", "jaccard")):
            X = boot[:, off:off + nt, m]
            others = np.delete(X, CAND, axis=1).mean(1)
            diff = X[:, CAND] - others
            lo_rank = (X.argmin(1) == CAND).mean()
            summ[f"{part}_{mname}"] = dict(
                ci95=[[round(float(np.nanquantile(X[:, k], q)), 3) for q in (0.025, 0.975)] for k in range(nt)],
                diff_mean=round(float(np.nanmean(diff)), 3),
                diff_ci95=[round(float(np.nanquantile(diff, q)), 3) for q in (0.025, 0.975)],
                p_diff_ge0=round(float((diff >= 0).mean()), 3), frac_lowest=round(float(lo_rank), 3))
            print(part, mname, summ[f"{part}_{mname}"], flush=True)
    pd.DataFrame(rows).to_csv(out("h16c_continuity.csv"), index=False)
    json.dump(summ, open(out("h16c_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
