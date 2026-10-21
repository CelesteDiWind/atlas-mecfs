"""Paso 9: papers puente.
Para cada paper (SPECTER2, cuerda completa sin él): cambio del diagrama H1 al quitarlo (bottleneck y
pérdida de persistencia de las barras significativas) y cambio de H0 (persistencia de la barra más larga finita).
Además: intermediación entre clusters (fracción de TODOS los caminos mínimos entre papers de clusters distintos que pasan
por él, en el grafo de 15 vecinos), qué pares de clusters une esos caminos, de qué clusters son sus 15 vecinos, y
pertenencia a ciclos significativos (paso 6). Puntuación final = media de rangos percentiles.
Reanudable (out/loo_progress.npy). Salidas: out/bridges.csv"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from common import MAIN, finite, load_emb, load_papers, out, rips, space_distance

SP = "full"  # en las geodésicas k15/k30 no hay ciclos significativos; los 6 ciclos están en la cuerda completa


def main():
    from persim import bottleneck
    P = load_papers()["papers"]
    cl = np.array([p["c"] for p in P])
    E = load_emb(MAIN)
    n = len(E)
    D = space_distance(E, SP)
    full = rips(D, 1)["dgms"]
    h1 = finite(full[1])
    B = pd.read_csv(out(f"h1_bars_{MAIN}_all_{SP}.csv"))
    sig_pers = B[B.p_gmm < 0.05].pers.sum()
    h0max = finite(full[0])[:, 1].max()

    path = out("loo_progress.npy")
    res = list(np.load(path)) if os.path.exists(path) else []
    for i in range(len(res), n):
        keep = np.delete(np.arange(n), i)
        dg = rips(space_distance(E[keep], SP), 1)["dgms"]
        d1 = finite(dg[1])
        bn = bottleneck(h1, d1) if len(d1) else np.nan
        # persistencia significativa que sobrevive: barras con persistencia >= la menor significativa
        thr = B[B.p_gmm < 0.05].pers.min() if sig_pers > 0 else np.inf
        surv = (d1[:, 1] - d1[:, 0])[(d1[:, 1] - d1[:, 0]) >= thr].sum() if np.isfinite(thr) else 0
        res.append([bn, sig_pers - surv, h0max - finite(dg[0])[:, 1].max()])
        if (i + 1) % 25 == 0 or i + 1 == n:
            np.save(path, np.array(res))
            print(f"  sin paper {i + 1}/{n}", flush=True)
    res = np.array(res)

    # intermediación entre clusters en el grafo de 15 vecinos, exacta: camino mínimo entre TODOS los pares de papers
    # de clusters distintos; para cada paper intermedio se anota qué par de clusters une ese camino
    nb = np.argsort(D, axis=1)[:, 1:16]
    A = np.zeros((n, n))
    rows = np.repeat(np.arange(n), nb.shape[1])
    A[rows, nb.ravel()] = D[rows, nb.ravel()]
    A = np.maximum(A, A.T)
    _, pred = dijkstra(csr_matrix(A), directed=False, return_predecessors=True)
    btw = np.zeros(n)
    via = [Counter() for _ in range(n)]  # via[v][(a, b)] = caminos entre los clusters a y b que pasan por v
    n_cross = 0
    for s in range(n):
        for t in range(s + 1, n):
            if cl[s] == cl[t] or pred[s, t] < 0:
                continue
            n_cross += 1
            key = (int(min(cl[s], cl[t])), int(max(cl[s], cl[t])))
            v = pred[s, t]
            while v != s:
                btw[v] += 1
                via[v][key] += 1
                v = pred[s, v]
    btw /= n_cross
    print(f"caminos entre clusters distintos: {n_cross}")

    def top_pairs(i, k=5):
        tot = sum(via[i].values())
        return [[a, b, round(c / tot, 3)] for (a, b), c in via[i].most_common(k)] if tot else []

    neigh = [{int(k): int(v) for k, v in Counter(cl[nb[i]]).most_common()} for i in range(n)]
    # papel: «puerta» de su tema (caminos que salen o llegan a su propio cluster) frente a puente entre otros dos temas
    gateway = [sum(c for (a, b), c in via[i].items() if cl[i] in (a, b)) / max(1, sum(via[i].values())) for i in range(n)]

    with open(out("cycle_members.json")) as fh:
        cyc = json.load(fh)
    in_cycles = {i: [] for i in range(n)}
    for cid, mem in cyc.items():
        if not cid.startswith(SP):
            continue
        for v in mem["nodes"]:
            in_cycles[v].append(cid)
    pct = lambda x: pd.Series(x).rank(pct=True).values
    score = (pct(np.nan_to_num(res[:, 0])) + pct(res[:, 1]) + pct(btw)) / 3
    df = pd.DataFrame(dict(id=range(n), title=[p["t"] for p in P], year=[p["y"] for p in P], cluster=cl,
                           citations=[p["cit"] for p in P], h1_bottleneck_change=np.round(res[:, 0], 5),
                           sig_persistence_lost=np.round(res[:, 1], 5), h0_change=np.round(res[:, 2], 5),
                           cross_cluster_betweenness=np.round(btw, 5), cycles=[",".join(in_cycles[i]) for i in range(n)],
                           bridge_score=np.round(score, 4),
                           # pares de clusters que une (a, b, fracción de los caminos que pasan por el paper) y vecinos
                           cluster_pairs=[json.dumps(top_pairs(i)) for i in range(n)],
                           own_topic_share=np.round(gateway, 3),
                           neighbor_clusters=[json.dumps(neigh[i]) for i in range(n)])).sort_values("bridge_score", ascending=False)
    df.to_csv(out("bridges.csv"), index=False)
    print(df.head(15)[["id", "title", "year", "cluster", "citations", "bridge_score", "cycles"]].to_string(index=False))


if __name__ == "__main__":
    main()
